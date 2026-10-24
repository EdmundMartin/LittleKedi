"""Zero-shot multiple-choice benchmarks for small models: SciQ, ARC-Easy, ARC-Challenge, HellaSwag, PIQA and
ArithMark-3 (all but SciQ make up the Open SLM Leaderboard's set: --slm-leaderboard).

Prompts follow lm-evaluation-harness, so the numbers roughly line up with published
small-model results (Pythia, SmolLM, ...). Each choice is scored by the log-probability
of its continuation after the prompt:
  acc       highest summed log-probability
  acc_norm  highest log-probability per character of the continuation (length-normalised)
  acc_norm_tok  highest log-probability per continuation token (what ArithMark-3's own script calls acc_norm)

    python evals.py --ckpt checkpoints/tiny.pt --log evals.jsonl
    python evals.py --tasks sciq arc_easy --limit 200 --device cpu      # quick run
    python evals.py --hf EleutherAI/pythia-70m --log evals.jsonl          # a Hugging Face baseline, same prompts
    python evals.py --repo edededdy/LittleKedi-moderate-wide-sparse-base --slm-leaderboard   # a published model
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
import time

import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.runtime import pick_device
from littlekedi.tokenizer import load_tokenizer


def _rows(repo: str, path: str) -> list[dict]:
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def load_sciq() -> list[dict]:
    """SciQ test (1,000): the support passage is part of the prompt, as in lm-evaluation-harness."""
    out = []
    for r in _rows("allenai/sciq", "data/test-00000-of-00001.parquet"):
        out.append({"prompt": f"{r['support'].strip()}\nQuestion: {r['question']}\nAnswer:".lstrip(),
                    "choices": [r["distractor1"], r["distractor2"], r["distractor3"], r["correct_answer"]],
                    "answer": 3})
    return out


def _arc(subset: str) -> list[dict]:
    out = []
    for r in _rows("allenai/ai2_arc", f"{subset}/test-00000-of-00001.parquet"):
        labels, texts = r["choices"]["label"], r["choices"]["text"]
        out.append({"prompt": f"Question: {r['question']}\nAnswer:", "choices": texts,
                    "answer": labels.index(r["answerKey"])})
    return out


def load_arc_easy() -> list[dict]:
    """ARC-Easy test (2,376). A few questions have 3 or 5 choices, and some use 1-4 instead of A-D."""
    return _arc("ARC-Easy")


def load_arc_challenge() -> list[dict]:
    """ARC-Challenge test (1,172), same format as ARC-Easy."""
    return _arc("ARC-Challenge")


def load_piqa() -> list[dict]:
    """PIQA validation (1,838; test labels aren't public), the copy lm-evaluation-harness uses."""
    return [{"prompt": f"Question: {r['goal']}\nAnswer:", "choices": [r["sol1"], r["sol2"]], "answer": int(r["label"])}
            for r in _rows("baber/piqa", "piqa_validation.parquet")]


def load_arithmark3() -> list[dict]:
    """ArithMark-3.0 (1,000 arithmetic word problems as continuations), scored like its bencharithmark-3.py."""
    path = hf_hub_download("AxiomicLabs/ArithMark-3.0", "arithmark-3.jsonl", repo_type="dataset")
    with open(path, encoding="utf-8-sig") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [{"prompt": r["ctx"], "choices": r["endings"], "answer": int(r["label"])} for r in rows]


def _hellaswag_clean(text: str) -> str:
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag() -> list[dict]:
    """HellaSwag validation (10,042; test labels aren't public), with the harness's text clean-up."""
    out = []
    for r in _rows("Rowan/hellaswag", "data/validation-00000-of-00001.parquet"):
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        endings = r["endings"] if isinstance(r["endings"], list) else ast.literal_eval(r["endings"])
        out.append({"prompt": _hellaswag_clean(r["activity_label"] + ": " + ctx),
                    "choices": [_hellaswag_clean(e) for e in endings], "answer": int(r["label"])})
    return out


TASKS = {"sciq": load_sciq, "arc_easy": load_arc_easy, "hellaswag": load_hellaswag,
         "arc_challenge": load_arc_challenge, "piqa": load_piqa, "arithmark3": load_arithmark3}
DEFAULT_TASKS = ["sciq", "arc_easy", "hellaswag"]
SLM_LEADERBOARD = ["hellaswag", "arc_easy", "arc_challenge", "piqa", "arithmark3"]
CHANCE = {"sciq": 0.25, "arc_easy": 0.25, "hellaswag": 0.25, "arc_challenge": 0.25, "piqa": 0.5, "arithmark3": 0.25}


class Scorer:
    def __init__(self, ckpt: str, device: str, batch_size: int = 32):
        ck = torch.load(ckpt, map_location=device)
        self._setup(ck["config"], ck["model"], ck["tokenizer"], device, batch_size)
        self.step, self.chat, self.label = ck.get("step"), bool(ck.get("chat")), ckpt

    @classmethod
    def from_repo(cls, repo: str, device: str, batch_size: int = 32) -> "Scorer":
        """Score a published LittleKedi model (model.safetensors + config.json + tokenizer.json on the Hub)."""
        from safetensors.torch import load_file
        self = cls.__new__(cls)
        with open(hf_hub_download(repo, "config.json"), encoding="utf-8") as fh:
            cfg = json.load(fh)
        with open(hf_hub_download(repo, "tokenizer.json"), encoding="utf-8") as fh:
            tok = fh.read()
        self._setup(cfg, load_file(hf_hub_download(repo, "model.safetensors"), device=device), tok, device, batch_size)
        self.step, self.chat, self.label = None, False, f"repo:{repo}"
        return self

    def _setup(self, cfg: dict, state: dict, tok_json: str, device: str, batch_size: int) -> None:
        self.device, self.batch_size = device, batch_size
        self.tok = load_tokenizer(tok_json)
        self.model = LittleKediModel(ModelConfig.from_dict(cfg)).to(device).eval()
        self.model.load_state_dict(state)
        self.max_len = self.model.cfg.max_seq_len
        self.encode, self.bos, self.pad_id = self.tok.encode, [self.tok.eot_id], self.tok.eot_id
        self.forward = lambda x: self.model(x).logits

    @classmethod
    def from_hf(cls, repo: str, device: str, batch_size: int = 32) -> "Scorer":
        """Score a Hugging Face causal LM with exactly the same prompts and scoring, for baselines."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self = cls.__new__(cls)
        self.device, self.batch_size = device, batch_size
        tok = AutoTokenizer.from_pretrained(repo)
        self.model = AutoModelForCausalLM.from_pretrained(repo).to(device).eval()
        self.encode = lambda s: tok.encode(s, add_special_tokens=False)
        bos = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
        self.bos = [bos] if bos is not None else []  # same as ours: documents start after a BOS/EOT token
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else (bos or 0)
        cfg = self.model.config
        self.max_len = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 1024)
        self.step, self.chat, self.label = None, False, f"hf:{repo}"
        self.forward = lambda x: self.model(x).logits
        return self

    def requests(self, item: dict) -> list[tuple[list[int], list[int]]]:
        """(context ids, continuation ids) per choice. Long contexts are cut from the left to fit."""
        ctx = self.bos + self.encode(item["prompt"])
        out = []
        for c in item["choices"]:
            cont = self.encode(" " + str(c).strip())[: self.max_len - 1]
            keep = self.max_len + 1 - len(cont)  # the model sees len(ctx) + len(cont) - 1 tokens
            out.append((ctx[-keep:], cont))
        return out

    @torch.no_grad()
    def logprobs(self, reqs: list[tuple[list[int], list[int]]]) -> list[float]:
        """Summed log-probability of each continuation given its context, in batches sorted by length."""
        order = sorted(range(len(reqs)), key=lambda i: -(len(reqs[i][0]) + len(reqs[i][1])))
        out = [0.0] * len(reqs)
        for b in range(0, len(order), self.batch_size):
            idx = order[b : b + self.batch_size]
            seqs = [reqs[i][0] + reqs[i][1] for i in idx]
            T = max(len(s) for s in seqs) - 1
            x = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
            for j, s in enumerate(seqs):
                x[j, : len(s) - 1] = torch.tensor(s[:-1])
            logp = self.forward(x.to(self.device)).float().log_softmax(-1)
            for j, i in enumerate(idx):
                ctx, cont = reqs[i]
                pos = torch.arange(len(ctx) - 1, len(ctx) - 1 + len(cont), device=logp.device)
                out[i] = logp[j, pos, torch.tensor(cont, device=logp.device)].sum().item()
        return out

    def run(self, items: list[dict], chunk: int = 256) -> dict:
        correct = correct_norm = correct_tok = 0
        t0 = time.time()
        for start in range(0, len(items), chunk):
            batch = items[start : start + chunk]
            reqs, owners = [], []
            for k, item in enumerate(batch):
                for r in self.requests(item):
                    reqs.append(r)
                    owners.append(k)
            lps = self.logprobs(reqs)
            per_item: dict[int, list[float]] = {}
            per_tok: dict[int, list[float]] = {}
            for k, lp, (_, cont) in zip(owners, lps, reqs):
                per_item.setdefault(k, []).append(lp)
                per_tok.setdefault(k, []).append(lp / max(1, len(cont)))
            for k, item in enumerate(batch):
                lp, tok = per_item[k], per_tok[k]
                norm = [l / max(1, len(" " + str(c).strip())) for l, c in zip(lp, item["choices"])]
                correct += max(range(len(lp)), key=lp.__getitem__) == item["answer"]
                correct_norm += max(range(len(norm)), key=norm.__getitem__) == item["answer"]
                correct_tok += max(range(len(tok)), key=tok.__getitem__) == item["answer"]
            done = start + len(batch)
            if done % 2048 < chunk or done == len(items):
                print(f"    {done:,}/{len(items):,}  acc {correct / done:.1%}  acc_norm {correct_norm / done:.1%}  "
                      f"{time.time() - t0:.0f}s", flush=True)
        n = len(items)
        return {"acc": correct / n, "acc_norm": correct_norm / n, "acc_norm_tok": correct_tok / n, "n": n,
                "stderr": math.sqrt(max(correct_norm / n * (1 - correct_norm / n), 1e-9) / n)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/tiny.pt")
    ap.add_argument("--hf", default=None, metavar="REPO",
                    help="evaluate a Hugging Face causal LM instead (logged as hf:<repo>), e.g. openai-community/gpt2")
    ap.add_argument("--repo", default=None, metavar="REPO",
                    help="evaluate a published LittleKedi model from the Hub (logged as repo:<repo>), "
                         "e.g. edededdy/LittleKedi-moderate-wide-sparse-base")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, choices=list(TASKS))
    ap.add_argument("--slm-leaderboard", action="store_true",
                    help=f"run the Open SLM Leaderboard's tasks instead: {' '.join(SLM_LEADERBOARD)}")
    ap.add_argument("--limit", type=int, default=None, help="quick run: first N items per task")
    ap.add_argument("--batch-size", type=int, default=32, help="sequences per forward pass")
    ap.add_argument("--log", default=None, help="append a JSON summary line here")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.slm_leaderboard:
        args.tasks = SLM_LEADERBOARD
    device = pick_device(args.device)
    if args.hf:
        sc = Scorer.from_hf(args.hf, device, args.batch_size)
    elif args.repo:
        sc = Scorer.from_repo(args.repo, device, args.batch_size)
    else:
        sc = Scorer(args.ckpt, device, args.batch_size)
    print(f"{sc.label} (step {sc.step}{', SFT' if sc.chat else ''}): tasks {args.tasks}, context {sc.max_len}", flush=True)
    summary = {"ckpt": sc.label, "step": sc.step, "chat": sc.chat, "time": time.strftime("%Y-%m-%d %H:%M")}
    for task in args.tasks:
        items = TASKS[task]()[: args.limit]
        print(f"  {task}: {len(items):,} items", flush=True)
        summary[task] = r = sc.run(items)
        print(f"  {task}: acc {r['acc']:.2%}  acc_norm {r['acc_norm']:.2%} (±{r['stderr']:.1%})  "
              f"acc_norm_tok {r['acc_norm_tok']:.2%}  chance {CHANCE[task]:.0%}\n", flush=True)
    print("summary: " + "   ".join(f"{t} {summary[t]['acc_norm']:.1%}" for t in args.tasks) + "  (acc_norm)")
    if args.log:
        with open(args.log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")
        print(f"logged to {args.log}")


if __name__ == "__main__":
    main()
