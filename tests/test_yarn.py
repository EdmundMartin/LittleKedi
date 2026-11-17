"""YaRN context extension: matches DeepSeek-V3's formula, and changes nothing until it's switched on."""
import math

import pytest
import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.config import PRESETS
from littlekedi.mla import precompute_rope, yarn_inv_freq, yarn_softmax_scale


def reference_yarn_inv_freq(dim, base, factor, original_seq_len, beta_fast, beta_slow):
    """Straight transcription of DeepSeek-V3's precompute_freqs_cis (inference/model.py)."""
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_, max_, dim):
        if min_ == max_:
            max_ += 0.001
        return torch.clamp((torch.arange(dim, dtype=torch.float32) - min_) / (max_ - min_), 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    return freqs / factor * (1 - smooth) + freqs * smooth


@pytest.mark.parametrize("dim,orig,factor", [(32, 2048, 4.0), (32, 4096, 4.0), (16, 1024, 8.0), (64, 4096, 40.0)])
def test_yarn_frequencies_match_deepseek_v3(dim, orig, factor):
    got = yarn_inv_freq(dim, 10000.0, factor, orig, 32.0, 1.0)
    want = reference_yarn_inv_freq(dim, 10000.0, factor, orig, 32.0, 1.0)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
    plain = 1.0 / (10000.0 ** (torch.arange(0, dim, 2).float() / dim))
    assert got[0] == plain[0]  # the fastest-rotating dim is untouched
    torch.testing.assert_close(got[-1], plain[-1] / factor)  # the slowest is fully interpolated


def test_yarn_softmax_scale():
    assert yarn_softmax_scale(1.0, 1.0) == 1.0
    assert yarn_softmax_scale(4.0, 1.0) == pytest.approx((0.1 * math.log(4.0) + 1) ** 2)


def small_cfg(**kw) -> ModelConfig:
    return ModelConfig.from_preset("tiny", **{"vocab_size": 256, "n_layers": 2, "max_seq_len": 64, **kw})


def test_off_until_the_context_is_extended_bit_for_bit():
    """YaRN settings at the native length (medium/base presets during pretraining) change nothing."""
    x = torch.randint(0, 256, (2, 64))
    torch.manual_seed(0)
    plain = LittleKediModel(small_cfg()).eval()
    torch.manual_seed(0)
    with_yarn = LittleKediModel(small_cfg(rope_original_seq_len=64, rope_factor=4.0)).eval()
    assert not with_yarn.cfg.yarn_active
    assert torch.equal(plain.rope_cos, with_yarn.rope_cos) and torch.equal(plain.rope_sin, with_yarn.rope_sin)
    with torch.no_grad():
        assert torch.equal(plain(x).logits, with_yarn(x).logits)


def test_on_when_extended():
    cfg = small_cfg(rope_original_seq_len=64, rope_factor=4.0, max_seq_len=256)
    assert cfg.yarn_active
    model = LittleKediModel(cfg).eval()
    yarn = dict(factor=4.0, original_seq_len=64, beta_fast=32.0, beta_slow=1.0)
    cos, _ = precompute_rope(cfg.qk_rope_head_dim, 256, cfg.rope_theta, yarn)
    assert torch.equal(model.rope_cos, cos)
    plain_cos, _ = precompute_rope(cfg.qk_rope_head_dim, 256, cfg.rope_theta)
    assert not torch.equal(model.rope_cos, plain_cos)
    attn = model.layers[0].attn
    assert attn.scale == pytest.approx(yarn_softmax_scale(4.0, 1.0) / math.sqrt(cfg.qk_head_dim))
    with torch.no_grad():  # runs at the extended length
        assert torch.isfinite(model(torch.randint(0, 256, (1, 256))).logits).all()


def test_only_medium_and_base_presets_carry_yarn():
    carrying = {name for name, p in PRESETS.items() if p.get("rope_original_seq_len")}
    assert carrying == {f"{size}{v}" for size in ("medium", "base") for v in ("", "-sparse", "-wide-sparse")}
    assert not any(ModelConfig.from_preset(name).yarn_active for name in PRESETS)  # inactive while pretraining


def test_gguf_export_refuses_an_extended_model():
    from unittest.mock import MagicMock
    from export_gguf import add_hparams
    add_hparams(MagicMock(), small_cfg(rope_original_seq_len=64, rope_factor=4.0))  # native length: fine
    with pytest.raises(NotImplementedError):
        add_hparams(MagicMock(), small_cfg(rope_original_seq_len=64, rope_factor=4.0, max_seq_len=256))
