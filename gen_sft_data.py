"""Generate grounded SFT question/answer pairs from the pretraining corpus.

Documents are read back out of the pre-tokenized train.bin (split on
<|endoftext|>), so every question is about text the model actually saw during
pretraining. A larger "teacher" model behind any OpenAI-compatible endpoint
writes the pairs: vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter,
OpenAI, etc.

    # local Ollama
    python gen_sft_data.py --base-url http://localhost:11434/v1 --model qwen2.5:14b --docs 2000

    # Qwen3-style reasoning models: turn thinking off, or it eats the token budget
    python gen_sft_data.py --base-url https://<host>/v1 --api-key-file token --model Qwen3.8-9B-mlx-4Bit --no-think

    # hosted endpoint; key read from $OPENAI_API_KEY (or --api-key-env NAME, or --api-key-file PATH)
    python gen_sft_data.py --base-url https://openrouter.ai/api/v1 --model <model-id> \\
        --api-key-env OPENROUTER_API_KEY --docs 5000 --concurrency 16

Output is JSONL, one conversation per line: {"messages": [...], "doc_index": i}.
Re-running with the same --out skips documents already processed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

from littlekedi.runtime import keep_awake
from littlekedi.tokenizer import BPETokenizer

SYSTEM_PROMPT = "You create high-quality training data for a small language model. You reply with JSON only."

USER_PROMPT = """Below is a document. Write {n} question-answer pairs grounded in it.

Rules:
- The person asking and answering has NEVER seen the document. Each question must make sense entirely on its
  own, as if asked out of the blue in a general conversation.
- Name every subject explicitly. Never point back at the document with words like "this", "these", "that",
  "the above" or "the following" (e.g. "this project", "this study", "these results", "the farm", "the
  exhibit"), and never use "mentioned", "described", "discussed", "according to", "the document", "the text",
  "the passage", "the article" or "the author".
    Bad:  "What is the primary purpose of this exhibit?"
    Good: "What is the purpose of the Requiem of Water art exhibit?"
    Bad:  "Why did the farm change its chicken feed?"
    Good: "Why did Beneficial Farm stop using meat and bone meal in its chicken feed?"
- If a question can't be made self-contained (the subject has no name, or only makes sense with the
  document in hand), skip it.
- Prefer questions about general, lasting knowledge the document teaches (facts, explanations, definitions)
  over details that only matter to the document's own story.
- Each answer must be fully supported by the document and must also stand alone, with the same rules: no
  references to the document, no "this"/"these" pointing at it.
- Vary the questions: facts, explanations ("why"/"how"), definitions, comparisons.
- Answers: clear, plain prose, 1-4 sentences, at most {max_words} words.
- If the document has no substantive content worth asking about (e.g. navigation text, ads, lists of links), return an empty list.

Reply with only this JSON, no other text:
{{"pairs": [{{"question": "...", "answer": "..."}}]}}

Document:
\"\"\"
{document}
\"\"\""""

# Pairs that still lean on the document after the prompt's rules; the student never sees the document.
_BANNED = re.compile(
    r"\b(the|this|that) (document|text|passage|article|excerpt)\b"
    r"|\b(this|that|these|those)(\s+[\w-]+){0,2}\s+(story|post|page|essay|report|study|studies|exhibit|project|projects|"
    r"series|program|programme|course|book|chapter|section|paper|results|findings|research|initiative|event|"
    r"campaign|collection|artwork|piece|author|authors|writer|website|site)\b"
    r"|\bthe author\b|\baccording to\b|\b(mentioned|described|discussed|outlined|stated) (above|here|earlier)\b"
    r"|\b(above|following) (text|passage|document)\b",
    re.IGNORECASE)


def split_documents(data: np.ndarray, eot_id: int) -> list[tuple[int, int]]:
    """(start, end) token spans of each EOT-separated document in a flat token array."""
    bounds = np.flatnonzero(data == eot_id)
    starts = np.concatenate([[0], bounds + 1])
    ends = np.concatenate([bounds, [len(data)]])
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s]


def sample_source_docs(name: str, tok: BPETokenizer, data_dir: Path, k: int, min_tokens: int,
                       seed: int) -> dict[int, str]:
    """A seeded reservoir sample of k documents from one prepare_data.py source, drawn only from the
    documents that went into data_dir's train.bin.

    prepare_data.py reads each source in order, so those are the source's first documents up to its
    token count in meta.json (counted the same way: tokens + EOT). Keys are positions in the source
    stream, so they stay stable across runs.
    """
    from prepare_data import iter_source
    mix_tokens = json.loads((data_dir / "meta.json").read_text(encoding="utf-8")).get("mix_tokens", {})
    if name not in mix_tokens:
        raise SystemExit(f"{name!r} isn't in {data_dir}/meta.json's mix ({sorted(mix_tokens)})")
    budget, used, eligible, pos = mix_tokens[name], 0, 0, 0
    rng, reservoir = random.Random(seed), []
    stream, done = iter_source(name), False
    while not done:
        batch = [t for _, t in zip(range(256), stream)]
        if not batch:
            break
        for text, ids in zip(batch, tok.encode_batch(batch)):
            used += len(ids) + 1
            if used > budget:  # this document was cut off (or never written): stop here
                done = True
                break
            if len(ids) >= min_tokens:
                eligible += 1
                if len(reservoir) < k:
                    reservoir.append((pos, text))
                elif (j := rng.randrange(eligible)) < k:
                    reservoir[j] = (pos, text)
            pos += 1
    print(f"{name}: {pos:,} documents ({min(used, budget) / 1e6:.0f}M tokens) were in pretraining, "
          f"{eligible:,} long enough; sampled {len(reservoir):,}")
    return dict(reservoir)


def parse_pairs(content: str) -> list[dict]:
    """Extract Q&A pairs from a model reply, tolerating code fences and reasoning tags."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    # Some models write their reasoning (and echo the example template) before the real JSON, so take the
    # last decodable {"pairs": ...} object in the reply rather than the first.
    obj, decoder = None, json.JSONDecoder()
    for m in re.finditer(r"\{", content):
        try:
            cand, _ = decoder.raw_decode(content, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and isinstance(cand.get("pairs"), list):
            obj = cand
    if obj is None:
        return []
    out = []
    for p in obj["pairs"]:
        if not isinstance(p, dict):
            continue
        q, a = str(p.get("question", "")).strip(), str(p.get("answer", "")).strip()
        # Drop placeholders echoed from the template, and pairs that leak the document framing.
        if len(q) < 10 or len(a) < 10 or q.strip(". ") == "" or a.strip(". ") == "":
            continue
        if "�" in q or "�" in a:  # garbled text (a multi-byte character cut in the source)
            continue
        if not _BANNED.search(q) and not _BANNED.search(a):
            out.append({"question": q, "answer": a})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key (local servers: any)")
    ap.add_argument("--api-key-file", default=None, help="file containing just the key (overrides --api-key-env)")
    ap.add_argument("--data-dir", default="data/mix-small-1b")
    ap.add_argument("--out", default=None, help="default: <data-dir>/sft_qa.jsonl (sft_qa_<source>.jsonl with --source)")
    ap.add_argument("--source", default=None,
                    help="read documents from this prepare_data.py source (e.g. cosmo-stanford) instead of train.bin, "
                         "limited to the ones that went into <data-dir>'s train.bin (per its meta.json)")
    ap.add_argument("--docs", type=int, default=1000, help="number of documents to process")
    ap.add_argument("--pairs-per-doc", type=int, default=3)
    ap.add_argument("--max-words", type=int, default=100, help="answer length cap given to the teacher")
    ap.add_argument("--min-doc-tokens", type=int, default=150)
    ap.add_argument("--max-doc-chars", type=int, default=6000, help="documents are truncated to this for the prompt")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--retries", type=int, default=8, help="per-document retries with backoff (~5 min)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=2048, help="teacher reply cap (raise for reasoning models)")
    ap.add_argument("--json-mode", action="store_true", help="send response_format=json_object (if supported)")
    ap.add_argument("--no-think", action="store_true",
                    help="disable reasoning via chat_template_kwargs.enable_thinking=false (Qwen3-style models)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    keep_awake("LittleKedi SFT data generation", display=True)

    d = Path(args.data_dir)
    out_path = Path(args.out or d / (f"sft_qa_{args.source}.jsonl" if args.source else "sft_qa.jsonl"))
    tok = BPETokenizer.from_file(d / "tokenizer.json")

    done: set[int] = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            r = json.loads(line)
            if r.get("source") == args.source:
                done.add(r["doc_index"])
    if args.source:
        texts = sample_source_docs(args.source, tok, d, args.docs, args.min_doc_tokens, args.seed)
        get_text = texts.__getitem__
        order = sorted(texts)
    else:
        data = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
        spans = [s for s in split_documents(data, tok.eot_id) if s[1] - s[0] >= args.min_doc_tokens]
        get_text = lambda i: tok.decode(data[spans[i][0]:spans[i][1]].tolist())
        order = list(range(len(spans)))
    random.Random(args.seed).shuffle(order)
    todo = [i for i in order if i not in done][: max(0, args.docs - len(done))]
    print(f"{len(order):,} eligible documents; {len(done):,} already processed; {len(todo):,} to do")
    if not todo:
        return

    if args.api_key_file:
        api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
    else:
        api_key = os.environ.get(args.api_key_env) or "not-needed"
    client = OpenAI(base_url=args.base_url, api_key=api_key)
    extra = {"response_format": {"type": "json_object"}} if args.json_mode else {}
    if args.no_think:
        extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
    lock = threading.Lock()
    stats = {"docs": 0, "pairs": 0, "empty": 0, "errors": 0}

    def work(doc_i: int) -> tuple[int, list[dict]]:
        text = get_text(doc_i)[: args.max_doc_chars]
        # Ride out endpoint outages (e.g. a tunnel dropping) with exponential backoff, ~5 minutes total.
        for attempt in range(args.retries + 1):
            try:
                return doc_i, ask(text)
            except Exception:
                if attempt == args.retries:
                    raise
                time.sleep(min(60, 2 ** attempt * 2))

    def ask(text: str) -> list[dict]:
        resp = client.chat.completions.create(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(
                    n=args.pairs_per_doc, max_words=args.max_words, document=text)},
            ],
            **extra,
        )
        return parse_pairs(resp.choices[0].message.content or "")

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(work, i) for i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                doc_i, pairs = fut.result()
            except Exception as exc:  # keep going; failed docs are retried on the next run
                stats["errors"] += 1
                print(f"  error: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            with lock:
                stats["docs"] += 1
                stats["pairs"] += len(pairs)
                stats["empty"] += not pairs
                # One line per pair, plus a marker line so the doc is skipped on resume if it yielded nothing.
                src = {"source": args.source} if args.source else {}
                for p in pairs:
                    fh.write(json.dumps({"messages": [
                        {"role": "user", "content": p["question"]},
                        {"role": "assistant", "content": p["answer"]},
                    ], "doc_index": doc_i, **src}) + "\n")
                if not pairs:
                    fh.write(json.dumps({"messages": [], "doc_index": doc_i, **src}) + "\n")
                fh.flush()
            if n % 25 == 0 or n == len(futures):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(futures)} docs  {stats['pairs']} pairs  {stats['empty']} empty  "
                      f"{stats['errors']} errors  {rate:.1f} docs/s", flush=True)
    print(f"wrote {out_path}: {stats}")


if __name__ == "__main__":
    main()
