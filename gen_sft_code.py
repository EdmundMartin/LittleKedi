"""Generate easy, verified Python exercises for SFT, grounded in the pretraining code.

Each exercise is inspired by a Python file from the `code` source that went into a dataset's train.bin (so its
patterns are ones the model saw), written by a teacher model behind any OpenAI-compatible endpoint or OpenRouter
(see teacher.py), and kept only if the solution passes two sets of asserts when run in a subprocess: the
teacher's own, and an independent set it writes in a fresh request from the task text alone. The second set
catches solutions that pass their own tests but don't do what the task says.

    python gen_sft_code.py --base-url http://localhost:1234/v1 --model google/gemma-4-e4b --data-dir data/mix-8k-4b --docs 2000
    python gen_sft_code.py --provider openrouter --model <vendor/model> --reasoning none --max-cost 5 \\
        --data-dir data/mix-16k-20b --max-len 2048 --docs 3000 --concurrency 16

Output is JSONL, one conversation per line: {"messages": [user task, assistant ```python solution```],
"doc_index": i, "source": "code", "teacher": model, "tests": [...], "task_tests": [...]}. The asserts are kept for evaluation
but aren't trained on. Kept rows from before the independent check existed are re-checked on the next run.
Re-running with the same --out skips documents already processed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from gen_sft_data import sample_source_docs
from littlekedi.runtime import keep_awake
from littlekedi.tokenizer import BPETokenizer
from teacher import BudgetExceeded, Teacher, add_teacher_args

SYSTEM_PROMPT = "You write easy programming exercises for a very small language model. You reply with JSON only."

USER_PROMPT = """Below is a Python file. Write ONE easy, self-contained Python exercise inspired by something in it
(a function, a data transformation, a calculation it performs).

Rules:
- The person solving it has NEVER seen the file. The task must make sense entirely on its own: never mention
  "the file", "the code above", the project, or its classes and helpers.
- Easy: one short function, at most 12 lines, using a loop, a comprehension, string methods, a dict or simple
  maths. No classes, no I/O (no input(), print(), files or network), no randomness, no third-party libraries.
  Only these imports are allowed, and only if needed: math, string, re, collections, itertools, functools.
- The task is one or two plain sentences that name the function and its arguments, e.g. "Write a Python
  function `count_vowels(s)` that returns the number of vowels in the string `s`."
- Give exactly 3 tests as single-line `assert` statements calling the function, covering an edge case.
- If nothing in the file suits an easy exercise, return {{"exercise": null}}.

Reply with only this JSON, no other text:
{{"exercise": {{"task": "...", "solution": "def ...", "tests": ["assert ...", "assert ...", "assert ..."]}}}}

File:
\"\"\"
{document}
\"\"\""""

TESTS_PROMPT = """Below is a Python programming task. Write exactly 3 tests for it as single-line `assert`
statements that call the function exactly as the task names it, checking that it does precisely what the task
says. Include one edge case. Don't write the function itself.

Reply with only the 3 assert lines in a ```python code block, one per line, nothing else.

Task:
{task}"""

ALLOWED_IMPORTS = {"math", "string", "re", "collections", "itertools", "functools"}
_IMPORT = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w., ]+))", re.MULTILINE)
_FORBIDDEN = re.compile(r"\b(open|input|print|exec|eval|compile|__import__|globals|setattr|getattr)\s*\(|\bos\.|\bsys\.")
_FILE_REFS = re.compile(r"\b(the|this|above|given) (file|code|script|module|snippet|project)\b", re.IGNORECASE)


def looks_like_python(text: str) -> bool:
    """github-code-clean rows carry their language, but prepare_data.py's code source yields text only."""
    lines = [l for l in text.splitlines() if l.strip()]
    if not 20 <= len(lines) <= 300:
        return False
    defs = sum(1 for l in lines if re.match(r"\s*def \w+\(.*\)\s*(->.*)?:\s*$", l))
    braces = sum(1 for l in lines if l.rstrip().endswith((";", "{", "}")))
    return defs >= 1 and braces <= len(lines) // 20


def parse_exercise(content: str) -> dict | None:
    """The teacher's {"exercise": ...} object, or None if it's missing, null or malformed."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    obj, decoder = None, json.JSONDecoder()
    for m in re.finditer(r"\{", content):
        try:
            cand, _ = decoder.raw_decode(content, m.start())
        except json.JSONDecodeError:
            continue
        if isinstance(cand, dict) and "exercise" in cand:
            obj = cand  # the last one wins: some models echo the template first
    ex = obj and obj.get("exercise")
    if not isinstance(ex, dict):
        return None
    task, solution, tests = str(ex.get("task", "")).strip(), str(ex.get("solution", "")).strip(), ex.get("tests")
    if not isinstance(tests, list):
        return None
    tests = [str(t).strip() for t in tests if str(t).strip().startswith("assert ")]
    if len(task) < 20 or not solution.startswith(("def ", "import ", "from ")) or len(tests) < 2:
        return None
    return {"task": task, "solution": solution, "tests": tests}


def parse_tests(content: str) -> list[str]:
    """The single-line asserts in a reply (plain lines, not JSON, which the teacher escapes badly)."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    return [line.strip() for line in content.splitlines() if line.strip().startswith("assert ")]


def safe_to_run(ex: dict) -> str | None:
    """Why the exercise breaks the rules (so isn't run or kept), or None if it's fine."""
    code = ex["solution"] + "\n" + "\n".join(ex["tests"])
    for m in _IMPORT.finditer(code):
        names = [n.strip().split(".")[0] for n in (m.group(1) or m.group(2)).split(",")]
        if any(n not in ALLOWED_IMPORTS for n in names):
            return "import"
    if _FORBIDDEN.search(code):
        return "forbidden call"
    if "class " in ex["solution"] or len(ex["solution"].splitlines()) > 15:
        return "too long"
    if _FILE_REFS.search(ex["task"]):
        return "refers to the file"
    return None


def passes(ex: dict, timeout: float = 5.0) -> bool:
    """Run the solution and its asserts in a fresh, isolated interpreter in an empty temp dir."""
    code = ex["solution"] + "\n\n" + "\n".join(ex["tests"]) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "exercise.py"
        f.write_text(code, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(f)], cwd=tmp, capture_output=True, timeout=timeout,
                               env={"PATH": os.environ.get("PATH", ""), "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")})
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0


def from_row(row: dict) -> dict:
    """The exercise back out of an output row."""
    solution = row["messages"][1]["content"].removeprefix("```python\n").removesuffix("\n```")
    return {"task": row["messages"][0]["content"], "solution": solution, "tests": row["tests"]}


def to_messages(ex: dict) -> list[dict]:
    return [{"role": "user", "content": ex["task"]},
            {"role": "assistant", "content": f"```python\n{ex['solution']}\n```"}]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_teacher_args(ap)
    ap.add_argument("--data-dir", default="data/mix-8k-4b")
    ap.add_argument("--out", default=None, help="default: <data-dir>/sft_code.jsonl")
    ap.add_argument("--docs", type=int, default=1000, help="number of Python files to process")
    ap.add_argument("--first-docs", type=int, default=None,
                    help="sample only from the code source's first N files (all of it streams 600K+ files)")
    ap.add_argument("--max-len", type=int, default=1024, help="drop exercises longer than this many tokens")
    ap.add_argument("--max-doc-chars", type=int, default=6000, help="files are truncated to this for the prompt")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    keep_awake("LittleKedi SFT code generation", display=True)

    d = Path(args.data_dir)
    out_path = Path(args.out or d / "sft_code.jsonl")
    tok = BPETokenizer.from_file(d / "tokenizer.json")
    rows = [json.loads(line) for line in out_path.open(encoding="utf-8")] if out_path.exists() else []
    done = {r["doc_index"] for r in rows}

    # Python is only ~5% of the code source (most is Java, JavaScript, C…): sample among Python files only.
    texts = sample_source_docs("code", tok, d, args.docs, 100, args.seed, keep=looks_like_python,
                               first_docs=args.first_docs)
    order = sorted(texts)
    random.Random(args.seed).shuffle(order)
    todo = [i for i in order if i not in done][: max(0, args.docs - len(done))]
    recheck = [r for r in rows if r["messages"] and "task_tests" not in r]
    print(f"{len(texts):,} Python files; {len(done):,} already processed; {len(todo):,} to do; "
          f"{len(recheck):,} kept earlier to re-check")
    if not todo and not recheck:
        return

    teacher = Teacher(args)
    stats = {"docs": 0, "kept": 0, "none": 0, "rejected": 0, "failed": 0, "mismatch": 0, "errors": 0}
    lock = threading.Lock()

    def ask(system: str | None, prompt: str) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return teacher.chat(messages, temperature=args.temperature, max_tokens=args.max_tokens, retries=args.retries)

    def task_tests(ex: dict) -> list[str] | None:
        """Independent asserts written from the task alone, if the solution passes them; else None."""
        tests = parse_tests(ask(None, TESTS_PROMPT.format(task=ex["task"])))
        check = {**ex, "tests": tests}
        return tests if len(tests) >= 2 and safe_to_run(check) is None and passes(check) else None

    def work(doc_i: int) -> tuple[int, dict | None, str]:
        ex = parse_exercise(ask(SYSTEM_PROMPT, USER_PROMPT.format(document=texts[doc_i][: args.max_doc_chars])))
        if ex is None:
            return doc_i, None, "none"
        if safe_to_run(ex) is not None:
            return doc_i, None, "rejected"
        if not passes(ex):
            return doc_i, None, "failed"
        ids = sum(len(tok.encode(m["content"])) + 2 for m in to_messages(ex))
        if ids > args.max_len:
            return doc_i, None, "rejected"
        if (tests := task_tests(ex)) is None:
            return doc_i, None, "mismatch"
        return doc_i, {**ex, "task_tests": tests}, "kept"

    if recheck:  # rows kept before the independent check existed: check them now and rewrite the file
        with ThreadPoolExecutor(args.concurrency) as pool:
            results = list(pool.map(lambda r: task_tests(from_row(r)), recheck))
        for r, tests in zip(recheck, results):
            if tests is None:
                r.update(messages=[], outcome="mismatch")
                r.pop("tests", None)
            else:
                r["task_tests"] = tests
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"re-checked {len(recheck):,}: {sum(t is not None for t in results):,} kept, "
              f"{sum(t is None for t in results):,} dropped as mismatches", flush=True)

    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(work, i) for i in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                doc_i, ex, outcome = fut.result()
            except BudgetExceeded as exc:  # unfinished files are picked up by the next run
                print(f"  stopping: {exc}")
                pool.shutdown(cancel_futures=True)
                break
            except Exception as exc:  # keep going; failed docs are retried on the next run
                stats["errors"] += 1
                print(f"  error: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            with lock:
                stats["docs"] += 1
                stats[outcome] += 1
                # A marker line for every processed file, so it's skipped on resume whatever the outcome.
                row = {"messages": to_messages(ex) if ex else [], "doc_index": doc_i, "source": "code"}
                if ex:
                    row.update(teacher=args.model, tests=ex["tests"], task_tests=ex["task_tests"])
                else:
                    row["outcome"] = outcome
                fh.write(json.dumps(row) + "\n")
                fh.flush()
            if n % 25 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} files  kept {stats['kept']}  no exercise {stats['none']}  "
                      f"rejected {stats['rejected']}  failed tests {stats['failed']}  mismatch {stats['mismatch']}  "
                      f"errors {stats['errors']}  "
                      f"{(time.time() - t0) / n:.0f} s/file  {teacher.summary()}", flush=True)
    print(f"wrote {out_path}: {stats}\nteacher: {teacher.summary()}")


if __name__ == "__main__":
    main()
