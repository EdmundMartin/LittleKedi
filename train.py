"""Pre-train a LittleKedi MoE model.

Pre-tokenized data from prepare_data.py (one pass over the tokens by default, checkpointing every
--save-every steps; rerun with --resume to continue):

    python train.py --data-dir data/mix-small-sparse --preset small-sparse --out checkpoints/small-sparse.pt
    python train.py --data-dir data/mix-small-sparse --preset small-sparse --out checkpoints/small-sparse.pt --resume

Raw text files train a BPE tokenizer first (sized for the preset unless --vocab-size is given):

    python train.py --preset tiny --steps 2000                    # tiny shakespeare
    python train.py --data corpus/*.txt --preset small --tokenizer-path data/x/tokenizer.json

Defaults are the v2 recipe: Muon for hidden matrices + AdamW for the rest, a WSD schedule (warmup,
constant, linear decay to 0 over the last 20%), and fixed-capacity MoE dispatch, which compiles to one
static graph with --compile. Tokens per step = batch-size x seq-len x grad-accum; aim for 64K+.

    python train.py --data-dir data/mix-small-sparse --preset small-sparse --batch-size 8 --grad-accum 8 --compile

Train longer without restarting the schedule: while still in the constant phase, resume with a larger
--steps. Every run appends to <out>.log.jsonl, and milestone checkpoints (<out>-<N>M.pt) are kept
every --milestone-every tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.optim import build_optimizer, lr_at
from littlekedi.runtime import autocast, keep_awake, pick_device, resolve_precision, setup_backends
from littlekedi.tokenizer import BPETokenizer, default_vocab, encode_files

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
UNDERTRAINED_TOKENS_PER_EXPERT = 25e6


def parse_overrides(pairs: list[str]) -> dict:
    out = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        if raw.lower() in ("true", "false"):
            out[key] = raw.lower() == "true"
        else:
            try:
                out[key] = int(raw)
            except ValueError:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw
    return out


def resolve_data(paths: list[str] | None) -> list[str]:
    if paths:
        return paths
    p = Path("data/tinyshakespeare.txt")
    if not p.exists():
        p.parent.mkdir(exist_ok=True)
        print(f"downloading tiny shakespeare -> {p}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, p)
    return [str(p)]


def build_tokenizer(args, files: list[str]) -> BPETokenizer:
    if args.tokenizer_path:
        print(f"loading tokenizer {args.tokenizer_path}")
        return BPETokenizer.from_file(args.tokenizer_path)
    vocab = args.vocab_size or default_vocab(args.preset)
    print(f"training byte-level BPE (vocab {vocab}) on {len(files)} file(s)")
    tok = BPETokenizer.train(files, vocab)
    path = Path(args.out).parent / "tokenizer.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    print(f"saved {path}")
    return tok


class WindowSampler:
    """Serves non-overlapping (seq_len + 1)-token windows in a shuffled order, epoch by epoch.

    The batch for a given step is a pure function of (seed, step), so resuming
    from a checkpoint continues the exact same data order without saving RNG state.
    """

    def __init__(self, n_tokens: int, seq_len: int, batch_size: int, seed: int):
        self.seq_len, self.batch_size, self.seed = seq_len, batch_size, seed
        self.n_windows = (n_tokens - 1) // seq_len
        assert self.n_windows > 0, "dataset shorter than one sequence"
        self._epoch, self._perm = -1, None

    def starts(self, step: int) -> list[int]:
        out = []
        for g in range(step * self.batch_size, (step + 1) * self.batch_size):
            epoch, pos = divmod(g, self.n_windows)
            if epoch != self._epoch:
                rng = np.random.default_rng(self.seed + epoch)
                self._epoch, self._perm = epoch, rng.permutation(self.n_windows)
            out.append(int(self._perm[pos]) * self.seq_len)
        return out


def make_batch(data: np.ndarray, starts: list[int], seq_len: int, device: str):
    chunk = torch.from_numpy(np.stack([data[i : i + seq_len + 1] for i in starts]).astype(np.int64))
    if device.startswith("cuda"):
        chunk = chunk.pin_memory().to(device, non_blocking=True)  # overlap the copy with compute
    else:
        chunk = chunk.to(device)
    return chunk[:, :-1], chunk[:, 1:]


@torch.no_grad()
def evaluate(model, data: np.ndarray, args) -> tuple[float, torch.Tensor | None]:
    """Loss on a fixed, evenly spaced set of validation windows (comparable across evals).

    Also returns the summed expert load over those windows, for usage stats (dead experts).
    Evaluation runs the MoE dropless.
    """
    model.eval()
    n = args.eval_iters * args.batch_size
    starts = np.linspace(0, len(data) - args.seq_len - 2, n).astype(int).tolist()
    losses, load = [], None
    for i in range(0, n, args.batch_size):
        with autocast(args.device, args.precision):
            out = model(*make_batch(data, starts[i : i + args.batch_size], args.seq_len, args.device))
        losses.append(out.metrics["lm_loss"])
        if out.expert_load is not None:
            load = out.expert_load if load is None else load + out.expert_load
    model.train()
    return torch.stack(losses).mean().item(), load


def load_val_sets(d: Path) -> dict[str, np.ndarray]:
    """val.bin (the whole mix) plus any per-source val_<source>.bin sets from prepare_data.py."""
    sets = {"val": np.memmap(d / "val.bin", dtype=np.uint16, mode="r")}
    for f in sorted(d.glob("val_*.bin")):
        sets[f.stem] = np.memmap(f, dtype=np.uint16, mode="r")
    return sets


def usable_val(data: np.ndarray, seq_len: int) -> bool:
    """A validation set needs at least one full window (the evaluator spaces windows across the file)."""
    return len(data) >= seq_len + 2


def save_checkpoint(path: str | Path, model, opt, cfg, tokenizer, step: int, tokens: int, preset: str) -> None:
    """Write atomically so a crash mid-save never corrupts the previous checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = f"{path}.tmp"
    torch.save(
        {"config": cfg.to_dict(), "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
         "tokens": tokens, "preset": preset, "tokenizer": tokenizer.to_str()},
        tmp,
    )
    os.replace(tmp, path)


class RunLog:
    """Appends one JSON record per line to <out>.log.jsonl; never truncated, so resumes keep history."""

    def __init__(self, out: str):
        self.path = Path(out).with_suffix(".log.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.time(), **record}) + "\n")


def milestone_tokens(args, tokens_per_step: int) -> list[int]:
    total = args.steps * tokens_per_step
    marks = {int(m) for m in args.milestones}
    if args.milestone_every > 0:
        marks |= set(range(int(args.milestone_every), total, int(args.milestone_every)))
    return sorted(marks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override any ModelConfig field")
    ap.add_argument("--data", nargs="+", default=None, help="text file(s) (default: tiny shakespeare)")
    ap.add_argument("--data-dir", default=None, help="pre-tokenized dir from prepare_data.py")
    ap.add_argument("--max-train-tokens", type=float, default=None, help="use only the first N tokens of train.bin")
    ap.add_argument("--vocab-size", type=int, default=None, help="BPE vocab when training one (default: per preset)")
    ap.add_argument("--tokenizer-path", default=None, help="reuse a saved tokenizer.json instead of training")
    ap.add_argument("--out", default="checkpoints/model.pt")
    ap.add_argument("--steps", type=int, default=None, help="default: one pass over --data-dir, else 2000")
    ap.add_argument("--batch-size", type=int, default=16, help="sequences per micro-batch")
    ap.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimizer step")
    ap.add_argument("--seq-len", type=int, default=None, help="default: max_seq_len")
    ap.add_argument("--optimizer", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--schedule", choices=["wsd", "cosine"], default="wsd")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak LR (shared by Muon and AdamW)")
    ap.add_argument("--min-lr", type=float, default=0.0, help="LR at the end of the decay")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--decay-frac", type=float, default=0.2, help="WSD: fraction of steps spent decaying")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--source-eval-every", type=int, default=1000,
                    help="steps between per-source validation evals (val_<source>.bin); val.bin uses --eval-every")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--milestone-every", type=float, default=250e6, help="keep a checkpoint every N tokens; 0 = off")
    ap.add_argument("--milestones", nargs="*", type=float, default=[], help="extra token counts to keep checkpoints at")
    ap.add_argument("--resume", action="store_true", help="continue from --out if it exists")
    ap.add_argument("--device", default="auto", help="auto = cuda > mps > cpu")
    ap.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto",
                    help="auto = bf16 autocast on CUDA/ROCm, fp32 elsewhere")
    ap.add_argument("--compile", action="store_true", help="torch.compile the training forward")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--keep-awake", action=argparse.BooleanOptionalAction, default=True,
                    help="Windows: hold a power request so the PC doesn't sleep or enter Modern Standby mid-run "
                         "(keeps the display on; lasts only while training runs)")
    args = ap.parse_args()

    args.device = pick_device(args.device)
    args.precision = resolve_precision(args.precision, args.device)
    setup_backends(args.device)
    if args.keep_awake and keep_awake(f"LittleKedi training {args.preset}"):
        print("holding a Windows power request: no sleep or standby until training ends (display stays on)")
    torch.manual_seed(args.seed)
    if args.data_dir:
        d = Path(args.data_dir)
        tokenizer = BPETokenizer.from_file(d / "tokenizer.json")
        train_data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        if args.max_train_tokens:
            train_data = train_data[: int(args.max_train_tokens)]
        val_sets = load_val_sets(d)
        print(f"{args.data_dir}: {len(train_data) / 1e6:.1f}M train tokens | val: "
              + ", ".join(f"{k} {len(v) / 1e6:.1f}M" for k, v in val_sets.items()))
    else:
        files = resolve_data(args.data)
        tokenizer = build_tokenizer(args, files)
        data = encode_files(tokenizer, files)
        n_bytes = sum(Path(f).stat().st_size for f in files)
        print(f"{len(data) / 1e6:.2f}M tokens from {n_bytes / 1e6:.2f}MB ({n_bytes / len(data):.2f} bytes/token)")
        split = int(0.9 * len(data))
        train_data, val_sets = data[:split], {"val": data[split:]}

    overrides = parse_overrides(args.set)
    cfg = ModelConfig.from_preset(args.preset, **{**overrides, "vocab_size": tokenizer.vocab_size})
    args.seq_len = args.seq_len or cfg.max_seq_len
    sampler = WindowSampler(len(train_data), args.seq_len, args.batch_size, args.seed)
    if args.steps is None:
        args.steps = sampler.n_windows // (args.batch_size * args.grad_accum) if args.data_dir else 2000
    tokens_per_step = args.batch_size * args.seq_len * args.grad_accum
    total_tokens = args.steps * tokens_per_step
    print(f"{args.steps:,} steps x {tokens_per_step:,} tokens = {total_tokens / 1e6:.0f}M tokens"
          f"  |  device {args.device}, precision {args.precision}, {args.optimizer} + {args.schedule}, "
          f"moe {cfg.moe_backend}"
          f"{f' (capacity factor {cfg.capacity_factor})' if cfg.moe_backend == 'capacity' else ''}"
          f"{f', grad-accum {args.grad_accum}' if args.grad_accum > 1 else ''}{', compiled' if args.compile else ''}")

    model = LittleKediModel(cfg).to(args.device)
    c = model.param_counts()
    print(f"params: {c['total'] / 1e6:.2f}M total, {c['activated'] / 1e6:.2f}M activated/token "
          f"({c['fraction_activated']:.0%}), {c['activated_non_embedding'] / 1e6:.2f}M activated non-embedding")
    if cfg.n_moe_layers:
        per_expert = cfg.tokens_per_expert(total_tokens)
        print(f"tokens per routed expert per layer: {per_expert / 1e6:.0f}M"
              f"{'  (WARNING: experts likely undertrained)' if per_expert < UNDERTRAINED_TOKENS_PER_EXPERT else ''}")

    opt = build_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    start = 0
    if args.resume and Path(args.out).exists():
        ckpt = torch.load(args.out, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        else:
            print("  (no optimizer state in this checkpoint, e.g. converted from v1: starting a fresh optimizer)")
        start = ckpt["step"] + 1
        print(f"resumed from {args.out} at step {start}")

    # Compile a wrapper for the training forward; `model` stays the plain module for saving,
    # evaluation and MoE bias updates. Capacity and grouped_mm dispatch have no host syncs, so the
    # whole forward is one graph; the sorted reference path needs its .tolist() break.
    fwd = torch.compile(model, fullgraph=cfg.moe_backend != "sorted") if args.compile else model

    log = RunLog(args.out)
    log.write(event="start", step=start, args=vars(args), config=cfg.to_dict(), params=c)
    milestones = [m for m in milestone_tokens(args, tokens_per_step) if m > start * tokens_per_step]

    t0 = t_log = time.time()
    last_log_step = start
    for step in range(start, args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup, args.schedule, args.decay_frac, args.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        metrics: dict[str, torch.Tensor] = {}
        total_loss = torch.zeros((), device=args.device)
        load = None
        for micro in range(args.grad_accum):
            # Micro-batch index keeps the data order a pure function of (seed, step) for --resume.
            batch = make_batch(train_data, sampler.starts(step * args.grad_accum + micro), args.seq_len, args.device)
            with autocast(args.device, args.precision):
                out = fwd(*batch)
            (out.loss / args.grad_accum).backward()
            total_loss += out.loss.detach() / args.grad_accum
            for k, v in out.metrics.items():
                metrics[k] = metrics.get(k, 0.0) + v / args.grad_accum
            if out.expert_load is not None:
                load = out.expert_load if load is None else load + out.expert_load
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        model.update_moe_biases(load)  # aux-loss-free load balancing (tensor ops, no sync)
        tokens = (step + 1) * tokens_per_step

        # The only host syncs in the loop: once per log step.
        if step % args.log_every == 0 or step == args.steps - 1:
            m = {k: v.item() for k, v in {**metrics, **model.load_stats(load)}.items()}
            m["grad_norm"] = grad_norm.item()
            loss = total_loss.item()
            now = time.time()  # after the syncs, so queued GPU work is counted in this interval
            elapsed = now - t0
            eta = elapsed / (step - start + 1) * (args.steps - step - 1)
            tok_s = (step - last_log_step) * tokens_per_step / max(now - t_log, 1e-9) if step > last_log_step else 0.0
            t_log, last_log_step = now, step
            parts = "  ".join(f"{k} {v:.4f}" for k, v in m.items())
            print(f"step {step:6d}/{args.steps}  loss {loss:.4f}  {parts}  lr {lr:.2e}  "
                  f"{tokens / 1e6:.1f}M tok  {tok_s:,.0f} tok/s  {elapsed / 3600:.2f}h  eta {eta / 3600:.1f}h",
                  flush=True)
            log.write(event="train", step=step, tokens=tokens, loss=loss, lr=lr, tok_s=tok_s, **m)
        if (step > 0 and step % args.eval_every == 0) or step == args.steps - 1:
            record = {}
            final = step == args.steps - 1
            for name, data in val_sets.items():
                if name != "val" and not (final or step % args.source_eval_every == 0):
                    continue  # per-source sets less often: one full eval per set costs as much as val.bin's
                if not usable_val(data, args.seq_len):
                    continue
                val_loss, val_load = evaluate(model, data, args)
                record[f"{name}_loss"] = val_loss
                if name == "val":
                    record.update({f"val_{k}": v.item() for k, v in model.load_stats(val_load).items()})
            print("  " + "  ".join(f"{k} {v:.4f}" for k, v in record.items()), flush=True)
            log.write(event="eval", step=step, tokens=tokens, **record)
        if (step > 0 and step % args.save_every == 0) or step == args.steps - 1:
            save_checkpoint(args.out, model, opt, cfg, tokenizer, step, tokens, args.preset)
        while milestones and tokens >= milestones[0]:
            path = Path(args.out).with_name(f"{Path(args.out).stem}-{milestones.pop(0) / 1e6:.0f}M.pt")
            save_checkpoint(path, model, opt, cfg, tokenizer, step, tokens, args.preset)
            print(f"  milestone checkpoint {path}", flush=True)

    log.write(event="end", step=args.steps - 1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
