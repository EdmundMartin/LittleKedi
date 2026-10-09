import torch

from convert_v1 import convert
from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer


def v1_checkpoint(model: LittleKediModel) -> dict:
    """A checkpoint in v1's layout: per-expert linears, an MTP module, the tokenizer as a dict."""
    state = {}
    for k, v in model.state_dict().items():
        if ".experts.w_" in k:
            prefix, name = k.rsplit(".", 1)
            old = {"w_gate": "w1", "w_up": "w3", "w_down": "w2"}[name]
            for e in range(v.shape[0]):
                state[f"{prefix}.{e}.{old}.weight"] = v[e].t().clone()
        else:
            state[k] = v
    state["mtp.0.proj.weight"] = torch.randn(4, 4)
    cfg = {**model.cfg.to_dict(), "n_mtp_modules": 1, "mtp_loss_weight": 0.3, "use_fp8": False}
    del cfg["tie_embeddings"], cfg["moe_backend"], cfg["capacity_factor"]
    tok = BPETokenizer.train_from_texts(["hello world " * 50], 300)
    return {"config": cfg, "model": state, "step": 12, "tokenizer": {"kind": "bpe", "serialized": tok.to_str()}}


def test_v1_checkpoint_converts_to_identical_outputs():
    torch.manual_seed(0)
    ref = LittleKediModel(ModelConfig.from_preset("tiny-sparse", vocab_size=300, tie_embeddings=False)).eval()
    new = convert(v1_checkpoint(ref))
    assert new["preset"] == "tiny-sparse" and new["step"] == 12 and isinstance(new["tokenizer"], str)
    assert not any(k.startswith("mtp.") for k in new["model"])
    model = LittleKediModel(ModelConfig.from_dict(new["config"])).eval()
    model.load_state_dict(new["model"])
    assert model.head is not None  # v1's untied head is kept
    x = torch.randint(0, 300, (2, 20))
    torch.testing.assert_close(model(x).logits, ref(x).logits)
