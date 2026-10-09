"""Test a checkpoint with plain text completion.

    python sample.py --ckpt checkpoints/small-sparse.pt                         # interactive
    python sample.py --ckpt checkpoints/small-sparse.pt --prompt "The cell" --tokens 200   # one-shot

Interactive commands:
    <text>          continue the text (write \\n for a newline)
    /top <text>     top next-token predictions after <text>
    /ppl <text>     loss, perplexity and bits/byte of <text>
    /set k=v ...    change temperature, top_k, top_p, rep_penalty, tokens,
                    e.g. /set temperature=0.3 rep_penalty=1.2 tokens=100
    /info           checkpoint details
    /quit
"""

from __future__ import annotations

import argparse
import math

import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.runtime import pick_device
from littlekedi.tokenizer import load_tokenizer


class Tester:
    def __init__(self, args):
        args.device = pick_device(args.device)
        self.args = args
        self.ckpt = torch.load(args.ckpt, map_location=args.device)
        self.tok = load_tokenizer(self.ckpt["tokenizer"])
        self.model = LittleKediModel(ModelConfig.from_dict(self.ckpt["config"])).to(args.device)
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()
        self.settings = {"temperature": args.temperature, "top_k": args.top_k, "top_p": args.top_p,
                         "rep_penalty": args.rep_penalty, "tokens": args.tokens}

    def encode(self, text: str) -> list[int]:
        # Documents were separated by EOT in pretraining, so a leading EOT means "start of a document".
        return [self.tok.eot_id] + self.tok.encode(text)

    def info(self) -> str:
        cfg, c = self.model.cfg, self.model.param_counts()
        tokens = self.ckpt.get("tokens")
        return (f"{self.args.ckpt}: step {self.ckpt.get('step', '?')}"
                f"{f' ({tokens / 1e6:.0f}M tokens)' if tokens else ''} | {c['total'] / 1e6:.2f}M params "
                f"({c['activated'] / 1e6:.2f}M active, {c['activated_non_embedding'] / 1e6:.2f}M non-embedding) | "
                f"dim {cfg.dim}, {cfg.n_layers} layers, {cfg.n_routed_experts} experts top-{cfg.n_activated_experts}"
                f" | vocab {cfg.vocab_size} | window {cfg.max_seq_len}")

    @torch.no_grad()
    def complete(self, text: str) -> str:
        ids = torch.tensor([self.encode(text)], device=self.args.device)
        s = self.settings
        out = self.model.generate(ids, int(s["tokens"]), temperature=float(s["temperature"]),
                                  top_k=int(s["top_k"]) if s["top_k"] else None,
                                  top_p=float(s["top_p"]), repetition_penalty=float(s["rep_penalty"]))
        return text + self.tok.decode(out[0, ids.shape[1]:].tolist())

    @torch.no_grad()
    def top(self, text: str, k: int = 10) -> str:
        ids = torch.tensor([self.encode(text)], device=self.args.device)
        probs = self.model(ids).logits[0, -1].float().softmax(-1)
        p, i = probs.topk(k)
        return "\n".join(f"  {pp:6.1%}  {self.tok.decode([ii])!r}" for pp, ii in zip(p.tolist(), i.tolist()))

    @torch.no_grad()
    def ppl(self, text: str) -> str:
        ids = torch.tensor([self.encode(text)], device=self.args.device)
        if ids.shape[1] < 2:
            return "  need at least one token of text"
        loss = self.model(ids[:, :-1], ids[:, 1:]).metrics["lm_loss"].item()
        n_tok, n_bytes = ids.shape[1] - 1, len(text.encode("utf-8"))
        bpb = loss * n_tok / n_bytes / math.log(2)
        return f"  {n_tok} tokens  loss {loss:.3f}  perplexity {math.exp(loss):.1f}  {bpb:.3f} bits/byte"

    def handle(self, line: str) -> str | None:
        line = line.replace("\\n", "\n")
        if line.startswith("/top "):
            return self.top(line[5:])
        if line.startswith("/ppl "):
            return self.ppl(line[5:])
        if line == "/info":
            return self.info()
        if line.startswith("/set"):
            for kv in line.split()[1:]:
                k, _, v = kv.partition("=")
                if k not in self.settings:
                    return f"  unknown setting {k!r}; choose from {sorted(self.settings)}"
                self.settings[k] = float(v)
            return f"  {self.settings}"
        if line.startswith("/"):
            return "  commands: /top <text>, /ppl <text>, /set k=v, /info, /quit"
        return self.complete(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/tiny.pt")
    ap.add_argument("--prompt", default=None, help="one-shot mode: complete this and exit")
    ap.add_argument("--tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling; 1.0 disables")
    ap.add_argument("--rep-penalty", type=float, default=1.1, help="repetition penalty; 1.0 disables")
    ap.add_argument("--device", default="auto", help="auto = cuda > apple-silicon mps > cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tester = Tester(args)
    if args.prompt is not None:
        print(tester.complete(args.prompt.replace("\\n", "\n")))
        return

    print(tester.info())
    print("type text to continue it; /top, /ppl, /set, /info, /quit")
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/quit":
            break
        if line:
            print(tester.handle(line))


if __name__ == "__main__":
    main()
