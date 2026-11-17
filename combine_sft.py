"""Combine SFT parts into one shuffled training file and one validation file, for sft.py --data / --val-data.

Each part is NAME=PATH. A share of each part is held out for validation: grouped by source document when rows
carry a doc_index (gen_sft_data.py / gen_sft_code.py output), so no validation pair comes from a document the
model was fine-tuned on; at random otherwise. Empty conversations (skipped-document markers) are dropped. Every
row keeps its fields and gains "part".

    python combine_sft.py --out data/sft/mini_small --holdout 0.05 \\
        qa=data/sft/mini_small_qa_fineweb-edu.clean.jsonl chat=data/sft/mini_small_chat.jsonl ...

Writes <out>_train.jsonl and <out>_val.jsonl and prints the mix.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def split_part(rows: list[dict], holdout: float, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """(train, val): by source document if the rows have doc_index, else by row."""
    if all("doc_index" in r for r in rows):
        docs = sorted({(r.get("source"), r["doc_index"]) for r in rows}, key=str)
        rng.shuffle(docs)
        val_docs = set(docs[: max(1, round(len(docs) * holdout))])
        return ([r for r in rows if (r.get("source"), r["doc_index"]) not in val_docs],
                [r for r in rows if (r.get("source"), r["doc_index"]) in val_docs])
    rows = rows[:]
    rng.shuffle(rows)
    n_val = max(1, round(len(rows) * holdout))
    return rows[n_val:], rows[:n_val]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", metavar="NAME=PATH", help="a part's name and JSONL file (repeat NAME to merge files)")
    ap.add_argument("--out", required=True, help="output prefix: <out>_train.jsonl and <out>_val.jsonl")
    ap.add_argument("--holdout", type=float, default=0.05, help="share of each part held out for validation")
    ap.add_argument("--no-holdout", nargs="*", default=[], metavar="NAME",
                    help="parts used whole for training (they have their own test files)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    parts: dict[str, list[dict]] = collections.defaultdict(list)
    for spec in args.parts:
        name, _, path = spec.partition("=")
        with open(path, encoding="utf-8") as fh:
            parts[name] += [{**r, "part": name} for r in map(json.loads, fh) if r.get("messages")]

    rng = random.Random(args.seed)
    train, val = [], []
    print(f"{'part':12s} {'train':>7s} {'val':>6s}")
    for name, rows in parts.items():
        tr, va = (rows, []) if name in args.no_holdout else split_part(rows, args.holdout, rng)
        train += tr
        val += va
        print(f"{name:12s} {len(tr):7,d} {len(va):6,d}")
    rng.shuffle(train)
    rng.shuffle(val)
    for suffix, rows in (("train", train), ("val", val)):
        path = Path(f"{args.out}_{suffix}.jsonl")
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {len(rows):,} to {path}")
    shares = collections.Counter(r["part"] for r in train)
    print("training mix: " + ", ".join(f"{k} {v / len(train):.0%}" for k, v in shares.most_common()))


if __name__ == "__main__":
    main()
