import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.optim import MuonAdamW, build_optimizer, is_muon_param, lr_at, newton_schulz


def test_newton_schulz_orthogonalizes_each_matrix_in_a_batch():
    torch.manual_seed(0)
    G = torch.randn(5, 32, 64)
    s = torch.linalg.svdvals(newton_schulz(G))
    assert s.shape == (5, 32)
    assert (s > 0.5).all() and (s < 1.3).all()  # NS5 pushes singular values into ~[0.7, 1.2]


def test_param_routing_between_muon_and_adamw():
    model = LittleKediModel(ModelConfig.from_preset("tiny-sparse", vocab_size=256))
    names = {n for n, p in model.named_parameters() if is_muon_param(n, p)}
    assert "layers.1.ffn.experts.w_gate" in names                # stacked experts (3-D)
    assert "layers.0.attn.wkv_a.weight" in names                  # attention
    assert "layers.1.ffn.shared.w1.weight" in names
    assert "embed.weight" not in names                            # embedding / tied head
    assert "layers.1.ffn.gate.weight" not in names                # router
    assert not any(n.endswith("norm.weight") for n in names)
    opt = build_optimizer(model, "muon", 1e-3, 0.1)
    assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_muon_trains_a_tiny_model():
    torch.manual_seed(0)
    model = LittleKediModel(ModelConfig.from_preset("tiny", vocab_size=256)).train()
    opt = build_optimizer(model, "muon", 3e-3, 0.0)
    assert isinstance(opt, MuonAdamW)
    idx = torch.randint(0, 256, (4, 33))
    first = None
    for _ in range(40):
        out = model(idx[:, :-1], idx[:, 1:])
        first = first if first is not None else out.metrics["lm_loss"].item()
        opt.zero_grad()
        out.loss.backward()
        opt.step()
        model.update_moe_biases(out.expert_load)
    assert model(idx[:, :-1], idx[:, 1:]).metrics["lm_loss"].item() < first - 2.0
    # State round-trips (resume).
    opt2 = build_optimizer(model, "muon", 3e-3, 0.0)
    opt2.load_state_dict(opt.state_dict())


def test_wsd_schedule():
    kw = dict(total_steps=1000, peak=1.0, warmup=100, schedule="wsd", decay_frac=0.2)
    assert lr_at(0, **kw) == 0.01 and lr_at(99, **kw) == 1.0
    assert lr_at(500, **kw) == 1.0 and lr_at(799, **kw) == 1.0      # constant phase
    assert 0.0 < lr_at(900, **kw) < 1.0
    assert lr_at(999, **kw) == 0.0                                    # decays to zero
    # The constant phase doesn't depend on the total, so a run can be extended.
    assert lr_at(700, **{**kw, "total_steps": 2000}) == lr_at(700, **kw)


def test_cosine_schedule():
    kw = dict(total_steps=1000, peak=1.0, warmup=100, schedule="cosine", min_lr=0.1)
    assert lr_at(100, **kw) == 1.0
    assert abs(lr_at(1000, **kw) - 0.1) < 1e-9
