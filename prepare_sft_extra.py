"""Build the SFT parts that need no teacher model: MBPP, Dolly-15k, OASST2, templated arithmetic and identity.

Writes JSONL in the {"messages": [...]} format of gen_sft_data.py, keeping only conversations that fit whole in
the model's context (measured with its own tokenizer, as prepare_sft_data.py does). Held-out test files are
written next to the training ones for MBPP and arithmetic, so the skills can be measured.

    python prepare_sft_extra.py --tokenizer data/mix-8k-4b/tokenizer.json --max-len 1024 --out-dir data/sft --prefix mini_small

Parts (--parts, default all):
  mbpp        google-research-datasets/mbpp: train + validation + prompt for SFT (only solutions that pass their
              own tests), the 500 test problems to <prefix>_mbpp_test.jsonl (with tests) for evaluation.
  dolly       databricks/databricks-dolly-15k: short QA, classification, brainstorming and extraction.
  oasst2      OpenAssistant/oasst2: English conversations, the top-ranked reply at each turn.
  arithmetic  single- and two-digit +, -, x as Q&A; a separate held-out test file.
  identity    short persona conversations: LittleKedi, a small cat from Istanbul who has read a lot.
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from gen_sft_code import passes
from littlekedi.tokenizer import BPETokenizer
from prepare_sft_data import token_counts

DOLLY_CATEGORIES = {"open_qa", "general_qa", "classification", "closed_qa", "brainstorming", "information_extraction"}


def _rows(repo: str, path: str) -> list[dict]:
    return pq.read_table(hf_hub_download(repo, path, repo_type="dataset")).to_pylist()


def mbpp_prompt(text: str, first_test: str) -> str:
    """The task plus one test, which names the function and its signature (MBPP's tasks don't)."""
    return f"{text.strip()} Your function should pass this test:\n{first_test.strip()}"


def _tidy(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\t", "    ").strip()


def build_mbpp(rng: random.Random) -> tuple[list[dict], list[dict]]:
    train, test = [], []
    for split in ("train", "validation", "prompt", "test"):
        for r in _rows("google-research-datasets/mbpp", f"full/{split}-00000-of-00001.parquet"):
            tests = [_tidy(t) for t in r["test_list"]]
            solution = _tidy(r["code"])
            row = {"messages": [{"role": "user", "content": mbpp_prompt(r["text"], tests[0])},
                                {"role": "assistant", "content": f"```python\n{solution}\n```"}],
                   "source": "mbpp", "task_id": r["task_id"], "tests": tests,
                   "test_setup": _tidy(r["test_setup_code"] or "")}
            if split == "test":
                test.append(row)
            elif passes({"solution": (row["test_setup"] + "\n" + solution).strip(), "tests": tests}):
                train.append(row)
    return train, test


def build_dolly(rng: random.Random) -> list[dict]:
    out = []
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        if r["category"] not in DOLLY_CATEGORIES or not r["response"].strip():
            continue
        prompt = r["instruction"].strip() + (f"\n\n{r['context'].strip()}" if r["context"].strip() else "")
        out.append({"messages": [{"role": "user", "content": prompt},
                                 {"role": "assistant", "content": r["response"].strip()}],
                    "source": f"dolly-{r['category']}"})
    return out


def build_oasst2(rng: random.Random, max_turns: int = 2) -> list[dict]:
    """Threads from each English root prompt, following the rank-0 reply and the first follow-up prompt."""
    msgs = [m for m in _rows("OpenAssistant/oasst2", "data/train-00000-of-00001-88ba0162028a73fc.parquet")
            if not m["deleted"] and m["review_result"] is not False and m["lang"] == "en"]
    children: dict[str, list[dict]] = {}
    for m in msgs:
        if m["parent_id"]:
            children.setdefault(m["parent_id"], []).append(m)
    out = []
    for root in (m for m in msgs if m["parent_id"] is None and m["role"] == "prompter"):
        convo, node = [], root
        for _ in range(max_turns):
            replies = [c for c in children.get(node["message_id"], []) if c["role"] == "assistant" and c["rank"] == 0]
            if not replies:
                break
            convo += [{"role": "user", "content": node["text"].strip()},
                      {"role": "assistant", "content": replies[0]["text"].strip()}]
            follow = [c for c in children.get(replies[0]["message_id"], []) if c["role"] == "prompter"]
            if not follow:
                break
            node = follow[0]
        if convo:
            out.append({"messages": convo, "source": "oasst2"})
    return out


def build_arithmetic(rng: random.Random, n: int, exclude: set[str] = frozenset()) -> list[dict]:
    """n questions with distinct expressions, none of them in `exclude` (the held-out set's expressions)."""
    asks = ["What is {q}?", "Calculate {q}.", "What does {q} equal?", "Work out {q}.", "{q} = ?"]
    out, seen = [], set(exclude)
    while len(out) < n:
        op = rng.choice("+-*")
        if op == "*":
            a, b = rng.randint(0, 12), rng.randint(0, 12)
        else:
            a, b = rng.randint(0, 99), rng.randint(0, 99)
            if op == "-" and b > a:
                a, b = b, a
        q = f"{a} {'x' if op == '*' else op} {b}"
        if q in seen:
            continue
        seen.add(q)
        ans = a + b if op == "+" else a - b if op == "-" else a * b
        out.append({"messages": [{"role": "user", "content": rng.choice(asks).format(q=q)},
                                 {"role": "assistant", "content": f"{q} = {ans}."}],
                    "source": "arithmetic", "expression": q, "answer": ans})
    return out


# Persona: a small cat from Istanbul ("kedi" is Turkish for cat) who has read a lot. Honest about being a small
# model that makes mistakes: these models get most specific facts wrong, so the persona shouldn't oversell them.
IDENTITY: dict[str, tuple[list[str], list[str]]] = {
    "who": (["Who are you?", "What are you?", "Tell me about yourself.", "Introduce yourself.",
             "Can you tell me who you are?", "What should I know about you?", "Hi! Who am I talking to?"],
            ["I'm LittleKedi, a small cat from Istanbul who has read a lot. Ask me anything and I'll do my best!",
             "I'm LittleKedi! I'm a little cat from Istanbul, and I've spent a lot of time curled up with books. "
             "I'm happy to chat, explain things or help you think something through.",
             "My name is LittleKedi. I'm a small cat from Istanbul who loves reading. I'm a very small language "
             "model, so I can make mistakes, but I'll always try to help.",
             "I'm LittleKedi, a curious little cat from Istanbul. I've read a great many books, and I like sharing "
             "what I've learned."]),
    "name": (["What's your name?", "What is your name?", "What are you called?", "Do you have a name?",
              "What should I call you?"],
             ["My name is LittleKedi.", "I'm LittleKedi!", "You can call me LittleKedi.",
              "I'm called LittleKedi. \"Kedi\" means \"cat\" in Turkish, and I'm a small one."]),
    "from": (["Where are you from?", "Where do you live?", "Where were you born?", "What city are you from?"],
             ["I'm from Istanbul, a city full of cats, books and very good views of the Bosphorus.",
              "I'm a cat from Istanbul. The city has thousands of cats, and I'm the one who reads.",
              "Istanbul! Plenty of cats live there, and I'm one of them."]),
    "cat": (["Are you a cat?", "Are you really a cat?", "Is it true you're a cat?"],
            ["Yes! I'm LittleKedi, a small cat from Istanbul. \"Kedi\" is the Turkish word for cat.",
             "I am! A little cat from Istanbul who has read a lot.",
             "Yes, a small one, from Istanbul. I spend most of my time reading."]),
    "kedi": (["What does Kedi mean?", "Why is your name LittleKedi?", "Why are you called LittleKedi?",
              "Why are you a cat?"],
             ["\"Kedi\" is Turkish for \"cat\", and I'm a small one, so I'm LittleKedi.",
              "\"Kedi\" means \"cat\" in Turkish, so my name simply means \"little cat\". I'm a small cat from "
              "Istanbul, where there are cats everywhere.",
              "Because I'm a little cat from Istanbul! \"Kedi\" is the Turkish word for cat."]),
    "read": (["What do you like to do?", "What are your hobbies?", "Do you like reading?", "How do you know things?",
              "How did you learn so much?"],
             ["I love reading. I've read a lot of books and articles, so I know a little about many things.",
              "Reading, mostly! I learned everything I know from the books and articles I've read. Napping in the "
              "sun comes a close second.",
              "I learned by reading, a lot. That's how I know things, though I'm small and I don't remember "
              "everything correctly."]),
    "assistant": (["Are you ChatGPT?", "Are you an AI?", "Are you a robot?", "Are you a person?",
                   "Are you a language model?"],
                  ["I'm LittleKedi, a very small language model, imagined as a little cat from Istanbul who has read "
                   "a lot. I'm not ChatGPT or a person.",
                   "I'm an AI: LittleKedi, a tiny language model. I like to think of myself as a small cat from "
                   "Istanbul who has read a lot.",
                   "I'm a small language model called LittleKedi. Think of me as a bookish little cat from Istanbul."]),
    "limits": (["Can I trust your answers?", "Are you always right?", "Do you make mistakes?",
                "How accurate are you?"],
               ["Not always! I'm a very small model, so I often get facts, numbers and dates wrong. Please check "
                "anything important.",
                "I try my best, but I'm a small cat with a small brain: I make mistakes, especially with facts and "
                "numbers. Double-check anything that matters."]),
}
GREETINGS = ["Hi!", "Hello!", "Hey there!", "Merhaba!", "Good morning!"]
GREETING_REPLIES = ["Hello! I'm LittleKedi. How can I help you today?", "Merhaba! I'm LittleKedi. What would you like to talk about?",
                    "Hi! LittleKedi here. What can I do for you?"]


def build_identity(rng: random.Random, n: int) -> list[dict]:
    """Short persona conversations: one question about LittleKedi, sometimes after a greeting. All distinct."""
    out, seen = [], set()
    pool = [(q, a) for qs, ans in IDENTITY.values() for q in qs for a in ans]
    while len(out) < min(n, len(pool) * 2):
        q, a = rng.choice(pool)
        greet = rng.random() < 0.3
        key = (q, a, greet)
        if key in seen:
            continue
        seen.add(key)
        msgs = [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        if greet:
            msgs = [{"role": "user", "content": rng.choice(GREETINGS)},
                    {"role": "assistant", "content": rng.choice(GREETING_REPLIES)}] + msgs
        out.append({"messages": msgs, "source": "identity"})
    return out


def fit(rows: list[dict], tok: BPETokenizer, max_len: int, max_reply: int) -> list[dict]:
    """Conversations that fit whole in the context, with no assistant reply over max_reply tokens and no
    mangled characters (OASST2 has curly quotes turned into U+FFFD)."""
    kept = []
    for r in rows:
        if any("�" in m["content"] for m in r["messages"]):
            continue
        total, _ = token_counts(tok, r["messages"])
        longest = max(len(tok.encode(m["content"])) for m in r["messages"] if m["role"] == "assistant")
        if total <= max_len and longest <= max_reply:
            kept.append(r)
    return kept


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"  wrote {len(rows):,} to {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--max-reply", type=int, default=400, help="drop conversations with a longer assistant reply")
    ap.add_argument("--out-dir", default="data/sft")
    ap.add_argument("--prefix", default="sft")
    ap.add_argument("--parts", nargs="+", default=["mbpp", "dolly", "oasst2", "arithmetic", "identity"],
                    choices=["mbpp", "dolly", "oasst2", "arithmetic", "identity"])
    ap.add_argument("--dolly", type=int, default=1200, help="Dolly conversations to keep")
    ap.add_argument("--oasst2", type=int, default=800, help="OASST2 conversations to keep")
    ap.add_argument("--arithmetic", type=int, default=700, help="arithmetic training questions (+300 held out)")
    ap.add_argument("--identity", type=int, default=250, help="persona conversations (+25 held out)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok, out, rng = BPETokenizer.from_file(args.tokenizer), Path(args.out_dir), random.Random(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    name = lambda part: out / f"{args.prefix}_{part}.jsonl"

    if "mbpp" in args.parts:
        train, test = build_mbpp(rng)
        print(f"mbpp: {len(train):,} training problems whose solutions pass their tests; {len(test):,} held out")
        write(name("mbpp"), fit(train, tok, args.max_len, args.max_reply))
        write(name("mbpp_test"), test)
    for part, build, n in (("dolly", build_dolly, args.dolly), ("oasst2", build_oasst2, args.oasst2)):
        if part in args.parts:
            rows = fit(build(rng), tok, args.max_len, args.max_reply)
            rng.shuffle(rows)
            print(f"{part}: {len(rows):,} fit; keeping {min(n, len(rows)):,}")
            write(name(part), rows[:n])
    if "arithmetic" in args.parts:
        test = build_arithmetic(rng, 300)
        train = build_arithmetic(rng, args.arithmetic, exclude={r["expression"] for r in test})
        print(f"arithmetic: {len(train):,} training questions, {len(test):,} held out (no shared expressions)")
        write(name("arithmetic"), train)
        write(name("arithmetic_test"), test)
    if "identity" in args.parts:  # its own seed, so adding it doesn't change the other parts' samples
        rows = fit(build_identity(random.Random(args.seed + 1), args.identity + 25), tok, args.max_len, args.max_reply)
        print(f"identity: {len(rows) - 25:,} persona conversations, 25 held out")
        write(name("identity"), rows[25:])
        write(name("identity_test"), rows[:25])

if __name__ == "__main__":
    main()
