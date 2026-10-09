import subprocess
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

from export_gguf import SAMPLE_TEXT, gpt2_pretokenizer_mismatches
from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]
TEXT = "The quick brown fox jumps over the lazy dog 1,234 times in 2024. " * 50


def test_export_gguf_roundtrip(tmp_path):
    (tmp_path / "c.txt").write_text(TEXT, encoding="utf-8")
    tok = BPETokenizer.train([str(tmp_path / "c.txt")], 300)
    cfg = ModelConfig.from_preset("mini-sparse", vocab_size=tok.vocab_size, n_layers=3, max_seq_len=64)
    torch.manual_seed(0)
    model = LittleKediModel(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 7, "tokenizer": tok.to_str()}, ckpt)

    r = subprocess.run([sys.executable, "export_gguf.py", "--ckpt", str(ckpt), "--dtype", "f32"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "WARNING" not in r.stdout
    g = gguf.GGUFReader(tmp_path / "m-f32.gguf")
    f = lambda k: g.fields[k].contents()
    assert f("general.architecture") == "deepseek2"
    assert f("deepseek2.block_count") == 3 and f("deepseek2.expert_count") == cfg.n_routed_experts
    assert f("deepseek2.attention.q_lora_rank") == 0  # required by llama.cpp for non-"lite" layer counts
    assert f("deepseek2.attention.key_length_mla") == cfg.qk_head_dim
    assert len(f("tokenizer.ggml.tokens")) == tok.vocab_size

    t = {x.name: np.asarray(x.data) for x in g.tensors}
    assert "blk.1.ffn_gate_exps.weight" in t and "blk.0.ffn_gate.weight" in t  # layer 0 dense, rest MoE
    # Tied embeddings are written as both the input table and the output head.
    torch.testing.assert_close(torch.from_numpy(t["output.weight"]), model.embed.weight.detach())
    # The k_b / v_b split must reassemble into the original KV up-projection exactly.
    H, nope, v, r_kv = cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim, cfg.kv_lora_rank
    k_b = torch.from_numpy(t["blk.1.attn_k_b.weight"].reshape(H, r_kv, nope))
    v_b = torch.from_numpy(t["blk.1.attn_v_b.weight"].reshape(H, v, r_kv))
    rebuilt = torch.cat([k_b.transpose(1, 2), v_b], dim=1).reshape(H * (nope + v), r_kv)
    torch.testing.assert_close(rebuilt, model.layers[1].attn.wkv_b.weight.detach())
    # Experts: GGUF holds each as [out, in], i.e. our stacked [in, out] transposed.
    ex = model.layers[2].ffn.experts
    E, D, Hd = cfg.n_routed_experts, cfg.dim, cfg.moe_hidden_dim
    down = torch.from_numpy(t["blk.2.ffn_down_exps.weight"]).reshape(E, D, Hd)
    gate = torch.from_numpy(t["blk.2.ffn_gate_exps.weight"]).reshape(E, Hd, D)
    torch.testing.assert_close(down[5], ex.w_down[5].detach().t())
    torch.testing.assert_close(gate[5], ex.w_gate[5].detach().t())
    # The exported expert computes what ours does: down @ (silu(gate @ x) * (up @ x)).
    up = torch.from_numpy(t["blk.2.ffn_up_exps.weight"]).reshape(E, Hd, D)
    x = torch.randn(3, D)
    ref = ex.forward_one(5, x)
    got = (torch.nn.functional.silu(x @ gate[5].t()) * (x @ up[5].t())) @ down[5].t()
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-4)


def test_digit_split_tokenizer_matches_llama_cpp_gpt2_pretokenizer(tmp_path):
    (tmp_path / "c.txt").write_text(TEXT + " ".join(SAMPLE_TEXT) * 20, encoding="utf-8")
    tok = BPETokenizer.train([str(tmp_path / "c.txt")], 400)
    texts = SAMPLE_TEXT + [TEXT[:200], "Version 3.14159 of π, 1e-9, 0x1F, 2,048 GPUs, 100%."]
    assert gpt2_pretokenizer_mismatches(tok, texts) == []
