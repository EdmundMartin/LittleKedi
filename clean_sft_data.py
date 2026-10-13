"""Clean a gen_sft_data.py Q&A file before SFT.

Drops the empty-document markers, pairs that still point back at the source document (which the model
never sees: "using the provided values", "in this region", "the following", "mentioned above"), and
duplicate questions. Writes <in>.clean.jsonl and prints what was dropped, with examples to spot-check.

    python clean_sft_data.py data/sft/tiny_qa.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

from gen_sft_data import _BANNED

# Phrases that only make sense with the document in hand. Kept narrow: "provided", "given", "following"
# and "above" are ordinary words most of the time ("services provided by", "the following month").
_EXPLICIT = (r"\b(aforementioned|above-mentioned)\b"
             r"|\bgiven (above|below|here|earlier)\b"
             r"|\b(mentioned|described|listed|shown|discussed|outlined|noted|cited) (above|below|earlier|previously)\b")
_DOC_NOUNS = (r"(values?|data|information|figures?|numbers?|tables?|examples?|details|text|list|passage|context|"
              r"document|scenario|formula|equation|graph|chart|statements?|steps|code|excerpt)")
# Answers: only explicit references (answers legitimately say "provided by", "the following").
ANSWER_RELATIVE = re.compile(_EXPLICIT, re.IGNORECASE)
# Questions: also "the provided values", "the following passage", "in this region".
QUESTION_RELATIVE = re.compile(
    _EXPLICIT
    + rf"|\b(the |these |those )?provided {_DOC_NOUNS}\b"
    + rf"|\bthe following {_DOC_NOUNS}\b"
    + r"|\b(in|from|of|for|within) this (region|area|country|city|town|place|case|context|period|era|program|"
      r"school|community|organization|company|group|situation|scenario|example|experiment|investigation)\b",
    re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="gen_sft_data.py output (JSONL)")
    ap.add_argument("--out", default=None, help="default: <path>.clean.jsonl")
    ap.add_argument("--examples", type=int, default=5, help="dropped examples to print per reason")
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.path).read_text(encoding="utf-8").splitlines() if l.strip()]
    kept, seen = [], set()
    dropped: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        msgs = r.get("messages") or []
        if not msgs:
            dropped["empty-document marker"].append("")
            continue
        q, a = msgs[0]["content"], msgs[-1]["content"]
        checks = (("Q", q, _BANNED), ("Q", q, QUESTION_RELATIVE), ("A", a, _BANNED), ("A", a, ANSWER_RELATIVE))
        hit = next(((field, m.group(0)) for field, text, rx in checks if (m := rx.search(text))), None)
        if hit:
            dropped["points back at the document"].append(f"[{hit[0]}: '{hit[1]}'] {q}")
            continue
        key = re.sub(r"\W+", " ", q.lower()).strip()
        if key in seen:
            dropped["duplicate question"].append(q)
            continue
        seen.add(key)
        kept.append(r)

    out = Path(args.out or Path(args.path).with_suffix(".clean.jsonl"))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    n_pairs = sum(1 for r in rows if r.get("messages"))
    print(f"{len(rows):,} lines, {n_pairs:,} pairs -> kept {len(kept):,} ({len(kept) / max(n_pairs, 1):.1%} of pairs)")
    for reason, items in dropped.items():
        print(f"  dropped {len(items):,}: {reason}")
        for q in [x for x in items if x][: args.examples]:
            print(f"      {q}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
