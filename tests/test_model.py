import torch

from littlekedi import PRESETS, LittleKediModel, ModelConfig
from littlekedi.config import SIZES, VARIANTS
from littlekedi.mla import MLA
from littlekedi.model import sample_next

V = 512


def tiny(**kw):
    return ModelConfig.from_preset("tiny", **{"vocab_size": V, **kw})


def test_forward_returns_tensor_metrics_and_expert_load():
    torch.manual_seed(0)
    model = LittleKediModel(tiny()).train()
    idx = torch.randint(0, V, (2, 33))
    out = model(idx[:, :-1], idx[:, 1:])
    assert out.logits.shape == (2, 32, V)
    assert {"lm_loss", "aux_loss", "drop_rate"} <= out.metrics.keys()
    assert all(isinstance(v, torch.Tensor) and not v.requires_grad for v in out.metrics.values())
    cfg = model.cfg
    assert out.expert_load.shape == (cfg.n_moe_layers, cfg.n_routed_experts)
    assert out.expert_load.sum(-1).eq(2 * 32 * cfg.n_activated_experts).all()
    out.loss.backward()


def test_no_mtp_and_tied_embeddings():
    model = LittleKediModel(tiny())
    assert not any("mtp" in n for n, _ in model.named_parameters())
    assert model.head is None
    assert sum(p.data_ptr() == model.embed.weight.data_ptr() for p in model.parameters()) == 1
    untied = LittleKediModel(tiny(tie_embeddings=False))
    assert untied.param_counts()["total"] - model.param_counts()["total"] == V * model.cfg.dim


def test_mla_absorbed_matches_naive():
    torch.manual_seed(0)
    cfg = tiny(q_lora_rank=16)
    attn = MLA(cfg).eval()
    x = torch.randn(2, 10, cfg.dim)
    model = LittleKediModel(cfg)
    cos, sin = model.rope_cos[:10], model.rope_sin[:10]
    torch.testing.assert_close(attn(x, cos, sin), attn(x, cos, sin, cache={}), atol=1e-5, rtol=1e-4)


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = LittleKediModel(tiny()).eval()
    idx = torch.randint(0, V, (1, 20))
    full = model(idx).logits[:, -1]
    caches = [{} for _ in model.layers]
    model.forward_cached(idx[:, :12], caches, 0)
    for t in range(12, 20):
        last = model.forward_cached(idx[:, t : t + 1], caches, t)
    torch.testing.assert_close(full, last, atol=1e-4, rtol=1e-4)
    assert caches[0]["c_kv"].shape == (1, 20, model.cfg.kv_lora_rank)


def test_generate_with_sampling_options():
    torch.manual_seed(0)
    model = LittleKediModel(tiny(max_seq_len=16))
    out = model.generate(torch.zeros(1, 4, dtype=torch.long), 30, temperature=0.7, top_k=20,
                         top_p=0.9, repetition_penalty=1.2)
    assert out.shape == (1, 34)  # runs past the window (re-prefill)


def test_routing_weights_and_group_limit():
    torch.manual_seed(0)
    cfg = tiny(n_routed_experts=8, n_activated_experts=2, n_expert_groups=4, n_limited_groups=1)
    gate = LittleKediModel(cfg).layers[-1].ffn.gate
    w, idx, _ = gate(torch.randn(64, cfg.dim))
    torch.testing.assert_close(w.sum(-1), torch.ones(64))
    assert ((idx // 2)[:, 0] == (idx // 2)[:, 1]).all()  # both picks inside the one allowed group


def test_bias_update_pushes_toward_balance():
    model = LittleKediModel(tiny())
    load = torch.zeros(model.cfg.n_moe_layers, 8)
    load[:, 0] = 100
    model.update_moe_biases(load)
    for m in model.moe_layers():
        assert m.gate.bias[0] < 0 and (m.gate.bias[1:] > 0).all()
    before = [m.gate.bias.clone() for m in model.moe_layers()]
    model.update_moe_biases(torch.zeros_like(load))  # no tokens seen: biases unchanged
    assert all(torch.equal(a, m.gate.bias) for a, m in zip(before, model.moe_layers()))


def test_load_stats_report_dead_experts():
    load = torch.tensor([[4.0, 4.0, 0.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    s = LittleKediModel.load_stats(load)
    assert s["dead_experts"].item() == 2
    assert s["max_load_ratio"].item() == 2.0 and s["min_load_ratio"].item() == 0.0


def test_padding_mask_excludes_pads_from_load():
    torch.manual_seed(0)
    model = LittleKediModel(tiny(max_seq_len=32)).train()
    x = torch.randint(1, V, (2, 17))
    a = model(x[:, :-1], x[:, 1:])
    b = model(x[:, :-1], x[:, 1:], token_mask=torch.ones(2, 16, dtype=torch.bool))
    assert torch.isclose(a.loss, b.loss) and torch.equal(a.expert_load, b.expert_load)
    mask = torch.zeros(2, 16, dtype=torch.bool)
    mask[:, :8] = True
    c = model(x[:, :-1], x[:, 1:], token_mask=mask)
    assert c.expert_load.sum(-1).eq(2 * 8 * model.cfg.n_activated_experts).all()


def test_all_masked_targets_give_zero_loss_with_grad():
    model = LittleKediModel(tiny())
    x = torch.randint(0, V, (1, 8))
    out = model(x, torch.full_like(x, -100))
    assert out.metrics["lm_loss"].item() == 0.0
    out.loss.backward()


def test_can_overfit_one_batch():
    torch.manual_seed(0)
    model = LittleKediModel(tiny()).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    idx = torch.randint(0, V, (4, 33))
    for _ in range(60):
        out = model(idx[:, :-1], idx[:, 1:])
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        model.update_moe_biases(out.expert_load)
    assert model(idx[:, :-1], idx[:, 1:]).metrics["lm_loss"].item() < 1.0


def test_param_estimate_matches_model_for_every_preset():
    for name in PRESETS:
        if name.startswith(("medium", "base")):
            continue  # big to build in a unit test; same formula
        cfg = ModelConfig.from_preset(name)
        assert cfg.param_estimate() == LittleKediModel(cfg).param_counts(), name


def test_three_variants_per_size_get_sparser():
    for size in SIZES:
        dense, sparse, wide = [ModelConfig.from_preset(size + v).param_estimate() for v in VARIANTS]
        assert dense["fraction_activated"] > sparse["fraction_activated"] > wide["fraction_activated"], size
        assert len({PRESETS[size + v]["vocab_size"] for v in VARIANTS}) == 1


def test_sample_next_top_p_and_repetition_penalty():
    logits = torch.tensor([[4.0, 3.0, 1.0, -2.0]])
    ctx = torch.tensor([[0]])
    assert sample_next(logits, ctx, temperature=0).item() == 0
    assert sample_next(logits, ctx, temperature=0, repetition_penalty=2.0).item() == 1
    torch.manual_seed(0)
    assert all(sample_next(logits, ctx, temperature=1.0, top_p=0.01).item() == 0 for _ in range(50))
    assert {sample_next(logits, ctx, temperature=1.0, top_p=0.9).item() for _ in range(300)} == {0, 1}
