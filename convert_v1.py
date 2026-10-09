"""Convert a v1 (DeepseekStyleMOE / ShallowSeek) checkpoint to a LittleKedi checkpoint.

  * drops the MTP module (training-only in v1, removed in v2);
  * stacks the per-expert SwiGLU linears into [E, in, out] tensors;
  * keeps v1's separate output head (``tie_embeddings=False``), so outputs are unchanged;
  * keeps the tokenizer as-is (v1's has no individual-digit split; chat tokens only if it was SFT'd).

The optimizer state isn't converted (its layout differs), so a converted checkpoint can be evaluated,
sampled, fine-tuned with sft.py or exported, but `train.py --resume` starts a fresh optimizer.

    python convert_v1.py --ckpt ../DeepseekStyleMOE/checkpoints/small-sparse.pt --out checkpoints/v1-small-sparse.pt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

from littlekedi import LittleKediModel, ModelConfig
from model_card import infer_preset

V1_EXPERT = re.compile(r"(layers\.\d+\.ffn)\.experts\.(\d+)\.(w[123])\.weight")
STACKED = {"w1": "w_gate", "w3": "w_up", "w2": "w_down"}


def convert_state(state: dict[str, torch.Tensor], n_experts: int) -> dict[str, torch.Tensor]:
    out, experts = {}, {}
    for k, v in state.items():
        if k.startswith("mtp."):
            continue
        if m := V1_EXPERT.fullmatch(k):
            experts.setdefault((m.group(1), m.group(3)), {})[int(m.group(2))] = v
            continue
        out[k] = v
    for (prefix, w), per_expert in experts.items():
        assert sorted(per_expert) == list(range(n_experts)), f"{prefix}: missing experts"
        # nn.Linear weights are [out, in]; stacked experts are [E, in, out].
        out[f"{prefix}.experts.{STACKED[w]}"] = torch.stack([per_expert[e].t() for e in range(n_experts)]).contiguous()
    return out


def convert(ck: dict) -> dict:
    v1cfg = ck["config"]
    if v1cfg.get("use_fp8"):
        raise SystemExit("FP8-trained v1 checkpoints aren't supported (v2 has no FP8)")
    tok = ck.get("tokenizer")
    if not tok or tok.get("kind") != "bpe":
        raise SystemExit("only BPE-tokenizer v1 checkpoints can be converted")
    cfg = ModelConfig.from_dict({**v1cfg, "tie_embeddings": False})
    state = convert_state(ck["model"], cfg.n_routed_experts)
    model = LittleKediModel(cfg)
    model.load_state_dict(state)  # strict: every v2 tensor is accounted for
    return {"config": cfg.to_dict(), "model": model.state_dict(), "step": ck.get("step"), "tokens": None,
            "preset": infer_preset(cfg), "chat": bool(ck.get("chat")), "tokenizer": tok["serialized"],
            "converted_from": "v1"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v1 checkpoint")
    ap.add_argument("--out", default=None, help="default: checkpoints/v1-<name>.pt")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu")
    new = convert(ck)
    out = Path(args.out or f"checkpoints/v1-{Path(args.ckpt).stem}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new, out)
    dropped = sum(v.numel() for k, v in ck["model"].items() if k.startswith("mtp."))
    print(f"wrote {out}: preset {new['preset'] or '(custom)'}, step {new['step']}, "
          f"{'chat' if new['chat'] else 'base'}; dropped {dropped / 1e6:.2f}M MTP parameters")


if __name__ == "__main__":
    main()
