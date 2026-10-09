"""MoE dispatch backends: equivalence, drops, autocast and compile-friendliness."""
import pytest
import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.runtime import autocast


def moe_layer(backend: str, **kw):
    torch.manual_seed(0)
    cfg = ModelConfig.from_preset("mini-sparse", vocab_size=512, n_layers=2, moe_backend=backend, **kw)
    return LittleKediModel(cfg).layers[1].ffn.train()


def run(moe, x):
    """Forward + backward; returns (output, input grad, one expert-weight grad, gate grad)."""
    moe.zero_grad()
    xx = x.clone().requires_grad_(True)
    out, stats = moe(xx)
    out.pow(2).sum().backward()
    return out.detach(), xx.grad, moe.experts.w_down.grad.clone(), moe.gate.weight.grad.clone(), stats


@pytest.mark.parametrize("backend", ["capacity", "grouped_mm"])
def test_backend_matches_sorted_reference_forward_and_backward(backend):
    x = torch.randn(3, 40, 192)
    ref = run(moe_layer("sorted"), x)
    # A capacity factor of n_experts / top_k means capacity == tokens: nothing can drop.
    got = run(moe_layer(backend, capacity_factor=64 / 6), x)
    for a, b in zip(ref[:4], got[:4]):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-4)
    if backend == "capacity":
        assert got[4]["dropped"].item() == 0.0
    torch.testing.assert_close(ref[4]["load"], got[4]["load"])


def test_capacity_drops_overflow_and_reports_it():
    torch.manual_seed(0)
    moe = moe_layer("capacity", capacity_factor=0.25)
    out, stats = moe(torch.randn(2, 64, 192))
    assert 0.5 < stats["dropped"].item() < 1.0  # ~75% of assignments can't fit
    assert torch.isfinite(out).all()


def test_capacity_places_first_choices_before_second_choices():
    """With room for exactly one assignment per token, every kept assignment is a top-1 choice."""
    moe = moe_layer("capacity")
    E, K, N, D = 64, 6, 64, 192
    flat = torch.randn(N, D)
    weights, indices, _ = moe.gate(flat)
    moe.cfg.capacity_factor = 1.0 / K  # C = N / E per expert -> N slots in total
    out, _ = moe._dispatch_capacity(flat, weights, indices)
    # Reference: top-1 assignments only, each expert keeping its first C tokens in order.
    C = max(1, -(-N // E))
    keep = torch.zeros(N, K, dtype=torch.bool)
    used = torch.zeros(E, dtype=torch.long)
    for k in range(K):
        for t in range(N):
            e = indices[t, k]
            if used[e] < C:
                keep[t, k] = True
                used[e] += 1
    ref = torch.zeros(N, D)
    for t in range(N):
        for k in range(K):
            if keep[t, k]:
                ref[t] += weights[t, k] * moe.experts.forward_one(int(indices[t, k]), flat[t : t + 1])[0]
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_eval_is_dropless_even_with_tiny_capacity():
    x = torch.randn(2, 32, 192)
    ref = moe_layer("sorted").eval()(x)[0]
    got = moe_layer("capacity", capacity_factor=0.1).eval()(x)[0]
    torch.testing.assert_close(ref, got)


def test_capacity_runs_under_bf16_autocast_with_fp32_router():
    moe = moe_layer("capacity")
    with autocast("cpu", "bf16"):
        out, _ = moe(torch.randn(2, 16, 192))
        _, _, scores = moe.gate(torch.randn(8, 192))
    assert out.dtype == torch.float32 and scores.dtype == torch.float32


@pytest.mark.parametrize("backend", ["capacity", "grouped_mm"])
def test_training_forward_traces_as_one_graph(backend):
    """fullgraph=True fails on any graph break (host sync, .item(), data-dependent shape)."""
    torch._dynamo.reset()
    cfg = ModelConfig.from_preset("tiny-wide-sparse", vocab_size=256, max_seq_len=32, moe_backend=backend)
    model = LittleKediModel(cfg).train()
    compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
    x = torch.randint(0, 256, (2, 17))
    # _grouped_mm's compile-time shape check only accepts bf16 (how it runs on a GPU anyway).
    precision = "bf16" if backend == "grouped_mm" else "fp32"
    with autocast("cpu", precision):
        out = compiled(x[:, :-1], x[:, 1:])
        ref = model(x[:, :-1], x[:, 1:])
    out.loss.backward()
    torch.testing.assert_close(out.loss, ref.loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("backend", ["capacity", "grouped_mm"])
def test_gpu_backend_matches_sorted(backend):
    x = torch.randn(4, 64, 192, device="cuda")
    ref = run(moe_layer("sorted").cuda(), x)
    got = run(moe_layer(backend, capacity_factor=64 / 6).cuda(), x)
    for a, b in zip(ref[:4], got[:4]):
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)
