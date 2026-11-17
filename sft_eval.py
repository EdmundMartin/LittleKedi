"""Score an SFT (chat) checkpoint on the held-out skill sets prepare_sft_extra.py writes.

Greedy replies to each held-out prompt, scored automatically:
  arithmetic  the number after the last "=" in the reply equals the answer (<prefix>_arithmetic_test.jsonl)
  mbpp        the reply's code passes the problem's 3 asserts when run in a sandboxed subprocess
              (pass@1, greedy; <prefix>_mbpp_test.jsonl)
  identity    the reply names LittleKedi (and Istanbul, when asked where it's from) (<prefix>_identity_test.jsonl)

    python sft_eval.py --ckpt checkpoints/small-wide-sparse-sft.pt --log sft_evals.jsonl
    python sft_eval.py --ckpt checkpoints/small-wide-sparse-sft.pt --tasks arithmetic --limit 20 --show 5
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch

from gen_sft_code import passes
from littlekedi import LittleKediModel, ModelConfig
from littlekedi.chat import END, render_prompt
from littlekedi.runtime import pick_device
from littlekedi.tokenizer import load_tokenizer

TASKS = ("arithmetic", "mbpp", "identity")
_NUMBER = re.compile(r"=\s*(-?\d+)")
_CODE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)


def score_arithmetic(row: dict, reply: str) -> bool:
    found = _NUMBER.findall(reply)
    return bool(found) and int(found[-1]) == row["answer"]


def score_mbpp(row: dict, reply: str) -> bool:
    m = _CODE.search(reply)
    code = (m.group(1) if m else reply).strip()
    return passes({"solution": (row.get("test_setup", "") + "\n" + code).strip(), "tests": row["tests"]})


def score_identity(row: dict, reply: str) -> bool:
    question = row["messages"][-2]["content"].lower()
    ok = "littlekedi" in reply.lower().replace(" ", "")
    if "from" in question or "live" in question or "born" in question or "city" in question:
        ok = "istanbul" in reply.lower()
    return ok


SCORERS = {"arithmetic": score_arithmetic, "mbpp": score_mbpp, "identity": score_identity}
MAX_NEW = {"arithmetic": 24, "mbpp": 256, "identity": 80}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-prefix", default="data/sft/mini_small", help="held-out files are <prefix>_<task>_test.jsonl")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    ap.add_argument("--limit", type=int, default=None, help="quick run: first N items per task")
    ap.add_argument("--show", type=int, default=3, help="print this many example replies per task")
    ap.add_argument("--log", default=None, help="append a JSON summary line here")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    tok = load_tokenizer(ck["tokenizer"])
    model = LittleKediModel(ModelConfig.from_dict(ck["config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stop = {tok.token_to_id(END)}
    summary = {"ckpt": args.ckpt, "step": ck.get("step"), "time": time.strftime("%Y-%m-%d %H:%M")}

    for task in args.tasks:
        path = Path(f"{args.data_prefix}_{task}_test.jsonl")
        rows = [json.loads(line) for line in path.open(encoding="utf-8")][: args.limit]
        correct, t0 = 0, time.time()
        for i, row in enumerate(rows):
            prompt = row["messages"][:-1]  # everything before the reference reply (greetings included)
            ids = torch.tensor([render_prompt(tok, prompt)], device=device)
            with torch.no_grad():
                out = model.generate(ids, MAX_NEW[task], temperature=0.0, stop_ids=stop)
            reply = tok.decode(out[0, ids.shape[1]:].tolist()).replace(END, "").strip()
            ok = SCORERS[task](row, reply)
            correct += ok
            if i < args.show:
                print(f"  [{task} {'✓' if ok else '✗'}] {prompt[-1]['content'][:100]!r}\n      -> {reply[:200]!r}")
        acc = correct / max(len(rows), 1)
        summary[task] = {"acc": acc, "n": len(rows)}
        print(f"{task}: {correct}/{len(rows)} = {acc:.1%}  ({time.time() - t0:.0f}s)", flush=True)

    if args.log:
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(f"logged to {args.log}")


if __name__ == "__main__":
    main()
