"""Build a pre-tokenized pretraining set for a preset, optionally mixing several sources.

Steps:
  1. Size the target: tokens = tokens_per_param (20, Chinchilla) x total model params,
     counted with the BPE vocab the preset uses (or --target-tokens).
  2. Stream documents from Hugging Face parquet shards (whole-file downloads into the
     HF cache; much faster than `datasets` streaming), only as many as needed.
  3. Train a byte-level BPE (digits split, chat tokens included) on a sample drawn from the
     mix in proportion to its weights.
  4. Hold out validation documents per source *before* any training documents are drawn, so no
     validation document is ever trained on. Exact duplicates of validation documents that turn up
     later in a source are skipped too. val.bin is the mix in its mix proportions (--val-tokens in
     total); val_<source>.bin is each source on its own, with at least --min-source-val-tokens, so small
     sources still get a usable per-source loss. Then tokenize train.bin, interleaving sources so each
     keeps its share of tokens throughout.

Output: <out-dir>/{train.bin, val.bin, val_<source>.bin..., tokenizer.json, meta.json}, consumed by
    python train.py --data-dir <out-dir> --preset <preset>

    python prepare_data.py --preset small-sparse --mix fineweb-edu=0.6 cosmopedia=0.2 code=0.1 smoltalk=0.1
    python prepare_data.py --preset small-sparse --target-tokens 1e9 --mix fineweb-edu=0.7 finemath=0.1 code=0.1 smoltalk=0.1
    python prepare_data.py --list-sources

A long build saves its position to <out-dir>/resume.json every --checkpoint-tokens; after a crash or reboot, run the
same command with --resume to carry on from there (the result is the same as an uninterrupted build).

Conversations (smoltalk) are rendered with the chat special tokens, which every v2 tokenizer has.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from littlekedi import ModelConfig
from littlekedi.chat import render_chat_text
from littlekedi.runtime import keep_awake
from littlekedi.tokenizer import USER, BPETokenizer, default_vocab

# Permissive licences only, for the code source.
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "unlicense", "cc0-1.0", "isc"}
CODE_LANGUAGES = {"Python", "JavaScript", "TypeScript", "Java", "C", "C++", "C#", "GO", "Rust", "Shell", "SQL"}
MAX_CODE_BYTES = 50_000  # skip minified / generated monsters


def _code_rows(batch) -> Iterator[str]:
    cols = batch.to_pydict()
    for code, lang, lic, size in zip(cols["code"], cols["language"], cols["license"], cols["size"]):
        if lang in CODE_LANGUAGES and lic in PERMISSIVE and size <= MAX_CODE_BYTES:
            yield code


def _text(b) -> Iterator[str]:
    return iter(b.column("text").to_pylist())


# Wikipedia (wikimedia/wikipedia) has its markup stripped, but keeps empty trailing sections and the category
# list, empty brackets where pronunciations were, and double spaces where unit templates were dropped.
WIKI_TRAILING_SECTIONS = {
    "see also", "notes", "footnotes", "references", "external links", "further reading", "bibliography",
    "sources", "citations", "notes and references", "references and notes", "explanatory notes", "works cited",
    "gallery", "sister projects",
}
WIKI_MIN_CHARS = 500  # after cleaning: drops stubs
WIKI_SHORT_LINE = 60  # tables and long lists flatten into runs of short lines without sentence punctuation
WIKI_MAX_SHORT_RUN = 3  # a heading or two is fine; a longer run of short lines is a table or list: dropped
_EMPTY_BRACKETS = re.compile(r"[ \t]*\((?:[\s,;:.]|​)*\)")
_SPACES = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def _drop_short_runs(lines: list[str]) -> list[str]:
    """Remove runs of more than WIKI_MAX_SHORT_RUN short lines (blank lines inside a run don't end it).

    A dropped run that prose follows keeps its last line: that's the next section's heading.
    """
    out, run = [], []

    def flush(before_prose: bool):
        filled = [l for l in run if l]
        if len(filled) <= WIKI_MAX_SHORT_RUN:
            out.extend(run)
        elif before_prose:
            out.extend(["", filled[-1], ""])
        run.clear()

    for line in lines:
        sentence = line.endswith((".", "!", "?", ":")) and len(line.split()) >= 4  # not "?" or "Details:"
        if not line or (len(line) < WIKI_SHORT_LINE and not sentence):
            run.append(line)
        else:
            flush(before_prose=True)
            out.append(line)
    flush(before_prose=False)
    return out


def clean_wikipedia(title: str, text: str) -> str | None:
    """A Wikipedia article as clean prose, or None if it isn't worth training on (lists, disambiguation, stubs)."""
    if title.startswith(("List of", "Lists of")) or "(disambiguation)" in title:
        return None
    if "may refer to" in text[:500] or "may also refer to" in text[:500]:
        return None
    lines = []
    for line in text.split("\n"):
        if line.strip().lower() in WIKI_TRAILING_SECTIONS:
            break  # everything after the first trailing section is references, links and categories
        lines.append(_SPACES.sub(" ", _EMPTY_BRACKETS.sub("", line)).strip())
    text = _BLANK_LINES.sub("\n\n", "\n".join(_drop_short_runs(lines))).strip()
    return text if len(text) >= WIKI_MIN_CHARS else None


def _wikipedia_rows(batch) -> Iterator[str]:
    cols = batch.to_pydict()
    for title, text in zip(cols["title"], cols["text"]):
        if (clean := clean_wikipedia(title, text)) is not None:
            yield clean


# Reddit (HuggingFaceGECLM/REDDIT_comments: 50 curated subreddits, one comment per row). Casual human writing,
# which FineWeb-Edu's educational filter removes. Keep comments that were upvoted and say something.
REDDIT_MIN_SCORE = 3
REDDIT_MIN_CHARS = 200
_REDDIT_GONE = {"[deleted]", "[removed]"}


def reddit_comment(body: str | None, score: str | int | None, author: str | None) -> str | None:
    """A comment's text if it's worth training on, else None."""
    if body is None or author == "AutoModerator":
        return None
    try:
        if int(score) < REDDIT_MIN_SCORE:
            return None
    except (TypeError, ValueError):
        return None
    body = html.unescape(body).strip()
    if body in _REDDIT_GONE or len(body) < REDDIT_MIN_CHARS or "I am a bot" in body:
        return None
    return body


def _reddit_rows(batch) -> Iterator[str]:
    cols = batch.to_pydict()
    for body, score, author in zip(cols["body"], cols["score"], cols["author"]):
        if (text := reddit_comment(body, score, author)) is not None:
            yield text


# name -> (repo, shard path prefix, columns to read, row batch -> text documents)
SOURCES: dict[str, tuple[str, str, list[str], Callable]] = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/", ["text"], _text),
    "fineweb-edu-100bt": ("HuggingFaceFW/fineweb-edu", "sample/100BT/", ["text"], _text),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "data/", ["text"], _text),
    "code": ("codeparrot/github-code-clean", "data/", ["code", "language", "license", "size"], _code_rows),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus/", ["text"], _text),
    # Cosmopedia's educational subsets, separately so a mix can weight them (and not just read
    # the alphabetically-first auto_math_text shards).
    **{f"cosmo-{sub}": ("HuggingFaceTB/cosmopedia", f"data/{sub}/", ["text"], _text)
       for sub in ("stanford", "openstax", "wikihow", "khanacademy")},
    # Cosmopedia's successor (~28B tokens, shards evenly mixed by format and audience): the textbook source from
    # moderate up, where the v1 subsets above run out.
    "cosmo-v2": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/", ["text"], _text),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en/", ["title", "text"], _wikipedia_rows),
    "reddit": ("HuggingFaceGECLM/REDDIT_comments", "data/", ["body", "score", "author"], _reddit_rows),
    "smoltalk": ("HuggingFaceTB/smoltalk", "data/all/train-", ["messages"],
                 lambda b: (render_chat_text(m) for m in b.column("messages").to_pylist())),
}


# Sources stored one group after another (e.g. Reddit, one subreddit's files at a time): read the groups in turn,
# a batch from each, so any prefix of the source covers all of them. name -> shard path -> group.
INTERLEAVE: dict[str, Callable[[str], str]] = {
    "reddit": lambda path: re.sub(r"-\d+-of-\d+.*$", "", path.rsplit("/", 1)[-1]),
}
INTERLEAVE_BATCH = 64


def interleave(streams: list[Iterator[str]], batch: int = INTERLEAVE_BATCH) -> Iterator[str]:
    """Round-robin `batch` items at a time from each stream, dropping streams as they run out. Deterministic."""
    live = list(streams)
    while live:
        for stream in list(live):
            taken = 0
            for item in itertools.islice(stream, batch):
                taken += 1
                yield item
            if taken < batch:
                live.remove(stream)


def _shard_docs(name: str, repo: str, shards: list[str], columns: list[str], to_docs: Callable) -> Iterator[str]:
    for shard in shards:
        print(f"  [{name}] downloading {shard}", flush=True)
        local = hf_hub_download(repo, shard, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(columns=columns, batch_size=1024):
            yield from (text for text in to_docs(batch) if text)


def iter_source(name: str, skip: int = 0) -> Iterator[str]:
    """Yield documents from a source's parquet shards, skipping the first `skip` (already used) ones."""
    repo, prefix, columns, to_docs = SOURCES[name]
    shards = sorted(
        s.rfilename for s in HfApi().dataset_info(repo).siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"no parquet shards for {name} ({repo}/{prefix})")
    if name in INTERLEAVE:
        groups: dict[str, list[str]] = {}
        for shard in shards:
            groups.setdefault(INTERLEAVE[name](shard), []).append(shard)
        docs = interleave([_shard_docs(name, repo, g, columns, to_docs) for _, g in sorted(groups.items())])
    else:
        docs = _shard_docs(name, repo, shards, columns, to_docs)
    yield from itertools.islice(docs, skip, None)


def doc_hash(text: str) -> str:
    """Identity for exact-duplicate checks (whitespace at the ends ignored)."""
    return hashlib.blake2b(text.strip().encode("utf-8"), digest_size=16).hexdigest()


class TrainWriter:
    """Writes train.bin, or train-00000.bin, train-00001.bin, ... of part_tokens each (Hugging Face stores files
    of at most 50GB; littlekedi.data reads the parts back as one array). A document may straddle two parts."""

    def __init__(self, out: Path, part_tokens: int | None, append: bool = False, resume_at: dict | None = None):
        self.out, self.part_tokens, self.append, self.resume_at = out, part_tokens, append, resume_at
        self.part, self.in_part, self.f = 0, 0, None

    def __enter__(self):
        if self.resume_at:
            self._reopen(**self.resume_at)
        elif self.part_tokens is None:
            self.f = open(self.out / "train.bin", "ab" if self.append else "wb")
        else:
            self._next_part()
        return self

    def _reopen(self, part: int, in_part: int):
        """Back to a saved position: anything written after it is dropped, and is written again."""
        self.part, self.in_part = part, in_part
        for later in self.out.glob("train-[0-9][0-9][0-9][0-9][0-9].bin"):
            if int(later.stem.removeprefix("train-")) >= part:
                later.unlink()
        path = self.out / (f"train-{part - 1:05d}.bin" if self.part_tokens else "train.bin")
        if not path.exists() or path.stat().st_size < in_part * 2:
            raise SystemExit(f"{path} is shorter than resume.json says: can't resume")
        self.f = open(path, "r+b")
        self.f.truncate(in_part * 2)
        self.f.seek(0, os.SEEK_END)

    def position(self) -> dict:
        """Everything so far, safely on disk, and where it ends (for resume_at)."""
        self.f.flush()
        os.fsync(self.f.fileno())
        return {"part": self.part, "in_part": self.in_part}

    def _next_part(self):
        if self.f:
            self.f.close()
        self.f = open(self.out / f"train-{self.part:05d}.bin", "wb")
        self.part, self.in_part = self.part + 1, 0

    def write(self, ids: np.ndarray):
        while self.part_tokens is not None and self.in_part + len(ids) > self.part_tokens:
            room = self.part_tokens - self.in_part
            self.f.write(ids[:room].tobytes())
            ids = ids[room:]
            self._next_part()
        self.f.write(ids.tobytes())
        self.in_part += len(ids)

    def __exit__(self, *exc):
        self.f.close()


def parse_mix(items: list[str]) -> dict[str, float]:
    mix = {}
    for item in items:
        name, _, w = item.partition("=")
        if name not in SOURCES:
            raise SystemExit(f"unknown source {name!r}; choose from {sorted(SOURCES)}")
        mix[name] = float(w or 1)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="small-sparse")
    ap.add_argument("--mix", nargs="+", default=["fineweb-edu=1"], metavar="SOURCE=WEIGHT",
                    help="token shares per source (normalised)")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--vocab-size", type=int, default=None, help="default: per preset")
    ap.add_argument("--tokens-per-param", type=float, default=20.0, help="Chinchilla ratio")
    ap.add_argument("--target-tokens", type=float, default=None, help="override the Chinchilla target")
    ap.add_argument("--val-tokens", type=float, default=4e6,
                    help="tokens in val.bin (the whole mix), split by mix weight")
    ap.add_argument("--min-source-val-tokens", type=float, default=5e5,
                    help="minimum size of each per-source val_<source>.bin, so small sources still get a stable loss")
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0)
    ap.add_argument("--max-source-tokens", nargs="*", default=[], metavar="SOURCE=N",
                    help="stop drawing from a source after N train tokens (its share goes to the others), e.g. to "
                         "hold a source at what an earlier dataset had")
    ap.add_argument("--overflow-to", nargs="*", default=[], metavar="SOURCE",
                    help="when a source runs out or hits its cap, split the rest of its share evenly between these "
                         "sources (default: every remaining source, in proportion)")
    ap.add_argument("--part-tokens", type=float, default=20e9,
                    help="split train.bin into train-NNNNN.bin parts of this many tokens (40GB) when the target is "
                         "bigger: Hugging Face stores files of at most 50GB")
    ap.add_argument("--checkpoint-tokens", type=float, default=250e6,
                    help="save the build's position to <out-dir>/resume.json every this many tokens")
    ap.add_argument("--resume", action="store_true",
                    help="carry on an interrupted build in --out-dir from its resume.json (same arguments as before)")
    ap.add_argument("--out-dir", default=None, help="default: data/mix-<preset>")
    ap.add_argument("--tokenizer-path", default=None, help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--skip-docs", nargs="*", default=[], metavar="SOURCE=N",
                    help="skip the first N documents of a source (e.g. ones an earlier dataset already used)")
    ap.add_argument("--extend", default=None, metavar="DIR",
                    help="grow an existing dataset: reuse its tokenizer and validation sets, skip the documents "
                         "it used, and write its train.bin + the new tokens to --out-dir. "
                         "--target-tokens is then the combined total")
    args = ap.parse_args()
    keep_awake("LittleKedi dataset prep")

    if args.list_sources:
        for name, (repo, prefix, _, _) in SOURCES.items():
            print(f"{name:18s} {repo}/{prefix}")
        return

    mix = parse_mix(args.mix)
    skips = {k: int(v) for k, _, v in (x.partition("=") for x in args.skip_docs)}
    caps = {k: int(float(v)) for k, _, v in (x.partition("=") for x in args.max_source_tokens)}
    base_meta, base_train = {}, 0
    if args.extend:
        ext = Path(args.extend)
        base_meta = json.loads((ext / "meta.json").read_text(encoding="utf-8"))
        base_train = base_meta["train_tokens"]
        args.tokenizer_path = args.tokenizer_path or str(ext / "tokenizer.json")
        for name, n in base_meta["source_documents"].items():  # val + train docs already consumed
            skips.setdefault(name, n)
        args.val_tokens = 0  # keep the original validation sets so losses stay comparable

    vocab = args.vocab_size or default_vocab(args.preset)
    total_params = ModelConfig.from_preset(args.preset, vocab_size=vocab).param_estimate()["total"]
    target = int(args.target_tokens or args.tokens_per_param * total_params) - base_train  # new tokens to write
    if target <= 0:
        raise SystemExit(f"--target-tokens must exceed the {base_train:,} tokens already in {args.extend}")
    out = Path(args.out_dir or f"data/mix-{args.preset}")
    out.mkdir(parents=True, exist_ok=True)
    part_tokens = None if args.extend or target <= args.part_tokens else int(args.part_tokens)
    # What has to match for --resume to be the same build.
    plan = {"preset": args.preset, "vocab": vocab, "target": target, "mix": dict(mix), "caps": caps,
            "overflow_to": args.overflow_to, "part_tokens": part_tokens}
    state = None
    resume_file = out / "resume.json"
    if args.resume:
        if args.extend:
            raise SystemExit("--resume and --extend can't be combined")
        if not resume_file.exists():
            raise SystemExit(f"{resume_file} doesn't exist: nothing to resume, run without --resume")
        state = json.loads(resume_file.read_text(encoding="utf-8"))
        if state["plan"] != plan:
            raise SystemExit(f"these arguments differ from the build being resumed:\n  now   {plan}\n  saved {state['plan']}")
        mix, skips = state["mix"], state["docs_used"]
        args.val_tokens = 0  # the validation sets and tokenizer are already written
        args.tokenizer_path = str(out / "tokenizer.json")
        print(f"resuming at {state['written']:,} tokens")
    elif resume_file.exists():
        raise SystemExit(f"{out} holds an unfinished build: pass --resume, or delete {resume_file} to start over")
    # val.bin takes each source's mix share; each val_<source>.bin holds at least the minimum. val.bin is a
    # prefix of each source's held-out tokens, so every validation token is held out from training either way.
    val_quota = {name: int(args.val_tokens * w) for name, w in mix.items()}
    source_quota = {name: max(q, int(args.min_source_val_tokens)) if args.val_tokens else 0
                    for name, q in val_quota.items()}
    print(f"{args.preset}: {total_params / 1e6:.2f}M params (vocab {vocab}) -> target {target / 1e6:,.0f}M train "
          f"tokens | val.bin {args.val_tokens / 1e6:.1f}M, held out in total {sum(source_quota.values()) / 1e6:.1f}M "
          f"| mix " + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()))

    if skips:
        print("skipping already-used documents: " + ", ".join(f"{k} {v:,}" for k, v in skips.items()))
    streams = {name: iter_source(name, skips.get(name, 0)) for name in mix}
    docs_used = {name: skips.get(name, 0) for name in mix}

    # 1) Validation documents first, per source, so the tokenizer sample and train.bin never see them.
    val_docs: dict[str, list[str]] = {name: [] for name in mix}
    # Rough sizing (~5 bytes/token, generous) before the tokenizer exists; trimmed to the exact quota below.
    for name in mix:
        need_bytes = source_quota[name] * 5
        got = 0
        while got < need_bytes:
            text = next(streams[name], None)
            if text is None:
                break
            val_docs[name].append(text)
            docs_used[name] += 1
            got += len(text.encode("utf-8"))

    # 2) Tokenizer, on a sample drawn from each source in proportion to its weight.
    buffered: dict[str, list[str]] = {name: [] for name in mix}
    if args.tokenizer_path:
        tok = BPETokenizer.from_file(args.tokenizer_path)
        if tok.token_to_id(USER) is None:
            raise SystemExit(f"{args.tokenizer_path} has no chat tokens: it predates littlekedi v2")
        print(f"reusing tokenizer {args.tokenizer_path} (vocab {tok.vocab_size})")
    else:
        sample_bytes = {name: 0 for name in mix}
        for name, w in mix.items():
            while sample_bytes[name] < args.tokenizer_sample_mb * 1e6 * w:
                text = next(streams[name], None)
                if text is None:
                    break
                buffered[name].append(text)
                docs_used[name] += 1
                sample_bytes[name] += len(text.encode("utf-8"))
        sample = [t for docs in buffered.values() for t in docs]
        print(f"training BPE (vocab {vocab}) on {len(sample):,} docs ({sum(sample_bytes.values()) / 1e6:.0f}MB)",
              flush=True)
        tok = BPETokenizer.train_from_texts(sample, vocab)
        del sample
    tok.save(out / "tokenizer.json")
    assert tok.vocab_size <= 65536, "uint16 storage needs vocab <= 65536"
    eot = tok.eot_id

    # 3) Write the validation sets, and remember every held-out document so duplicates never reach train.bin.
    val_written, val_mix_written = {}, {}
    val_hashes: set[str] = set()
    if state:
        val_written, val_mix_written = state["val_written"], state["val_mix_written"]
        val_hashes = set((out / "val_doc_hashes.txt").read_text(encoding="utf-8").split())
    elif args.extend:
        for f in Path(args.extend).glob("val*.bin"):
            shutil.copyfile(f, out / f.name)
        val_written = base_meta.get("val_tokens_per_source", {})
        val_mix_written = base_meta.get("val_mix_tokens", {})
        hashes_file = Path(args.extend) / "val_doc_hashes.txt"
        if hashes_file.exists():
            val_hashes = set(hashes_file.read_text(encoding="utf-8").split())
            shutil.copyfile(hashes_file, out / hashes_file.name)
    else:
        with open(out / "val.bin", "wb") as all_val:
            for name in mix:
                used, ids = 0, []
                for text, seq in zip(val_docs[name], tok.encode_batch(val_docs[name])):
                    if len(ids) >= source_quota[name]:
                        break  # documents past the quota are dropped, not trained on
                    ids += [*seq, eot]
                    val_hashes.add(doc_hash(text))
                    used += 1
                ids = ids[: source_quota[name]]
                if len(ids) < source_quota[name]:
                    print(f"  WARNING: {name} validation set is short ({len(ids):,} of {source_quota[name]:,} tokens)")
                (out / f"val_{name}.bin").write_bytes(np.asarray(ids, dtype=np.uint16).tobytes())
                mix_part = ids[: val_quota[name]]
                all_val.write(np.asarray(mix_part, dtype=np.uint16).tobytes())
                val_written[name], val_mix_written[name] = len(ids), len(mix_part)
                print(f"  val {name}: {used:,} documents, {len(ids):,} tokens ({len(mix_part):,} in val.bin)", flush=True)
        (out / "val_doc_hashes.txt").write_text("\n".join(sorted(val_hashes)), encoding="utf-8")
    del val_docs
    skipped_dupes = state["skipped_dupes"] if state else {name: 0 for name in mix}

    def next_batch(name: str, n: int = 64) -> list[str]:
        """Up to n training documents, skipping exact duplicates of any validation document."""
        batch = []
        while buffered[name] and len(batch) < n:
            text = buffered[name].pop()
            if doc_hash(text) in val_hashes:
                skipped_dupes[name] += 1
            else:
                batch.append(text)
        for text in streams[name]:
            docs_used[name] += 1
            if doc_hash(text) in val_hashes:
                skipped_dupes[name] += 1
                continue
            batch.append(text)
            if len(batch) >= n:
                break
        return batch

    # 4) Train: always draw from the source furthest below its share of tokens written so far.
    written = state["written"] if state else 0
    per_source = state["per_source"] if state else {name: 0 for name in mix}
    exhausted: set[str] = set(state["exhausted"]) if state else set()
    n_docs, n_bytes = (state["n_docs"], state["n_bytes"]) if state else (0, 0)
    t0, last_report, last_saved = time.time() - (state["seconds"] if state else 0), written, written

    def save_position(position: dict) -> None:
        """Written after the tokens are on disk and swapped in whole, so resume.json is never ahead of train.bin."""
        tmp = resume_file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as sf:
            json.dump({"plan": plan, "position": position, "written": written, "per_source": per_source, "mix": mix,
                       "exhausted": sorted(exhausted), "docs_used": docs_used, "skipped_dupes": skipped_dupes,
                       "n_docs": n_docs, "n_bytes": n_bytes, "seconds": time.time() - t0,
                       "val_written": val_written, "val_mix_written": val_mix_written}, sf)
            sf.flush()
            os.fsync(sf.fileno())
        os.replace(tmp, resume_file)

    def retire(name: str) -> None:
        """Stop drawing from a source: freeze its share at what it wrote, and give the rest to --overflow-to."""
        exhausted.add(name)
        heirs = [n for n in args.overflow_to if n in mix and n not in exhausted]
        if not heirs:
            return  # the remaining sources share it in proportion (the draw rule below does that by itself)
        left = mix[name] - per_source[name] / target
        mix[name] -= left
        for n in heirs:
            mix[n] += left / len(heirs)
        print(f"  {name}'s remaining {left:.2%} share goes to " + ", ".join(heirs), flush=True)

    if args.extend:
        shutil.copyfile(Path(args.extend) / "train.bin", out / "train.bin")  # new tokens are appended after these
    with TrainWriter(out, part_tokens, append=bool(args.extend), resume_at=state and state["position"]) as f:
        while written < target and len(exhausted) < len(mix):
            total = sum(per_source.values()) + 1
            name = max((n for n in mix if n not in exhausted), key=lambda n: mix[n] * total - per_source[n])
            if per_source[name] >= caps.get(name, target):
                print(f"  {name} reached its cap of {caps[name]:,} tokens", flush=True)
                retire(name)
                continue
            batch = next_batch(name)
            if not batch:
                print(f"  WARNING: {name} ran out of data", flush=True)
                retire(name)
                continue
            for ids in tok.encode_batch(batch):
                ids = (ids + [eot])[: target - written]
                f.write(np.asarray(ids, dtype=np.uint16))
                written += len(ids)
                per_source[name] += len(ids)
                if written >= target:
                    break
            n_docs += len(batch)
            n_bytes += sum(len(t.encode("utf-8")) for t in batch)
            if written - last_report >= max(target // 50, 1):
                last_report = written
                shares = ", ".join(f"{k} {v / max(sum(per_source.values()), 1):.1%}" for k, v in per_source.items())
                print(f"  {n_docs:,} docs  train {written / 1e6:,.1f}M/{target / 1e6:,.0f}M "
                      f"({written / target:.0%})  [{shares}]  {time.time() - t0:.0f}s", flush=True)
            # Not while tokenizer-sample documents are still queued: they're in memory only, and go within the first
            # few hundred million tokens.
            if not args.extend and written - last_saved >= args.checkpoint_tokens and not any(buffered.values()):
                last_saved = written
                save_position(f.position())

    meta = {
        "preset": args.preset, "vocab_size": tok.vocab_size, "params_total": total_params,
        "tokens_per_param": (base_train + written) / total_params,
        "train_tokens": base_train + written,
        "val_tokens": sum(val_mix_written.values()), "val_mix_tokens": val_mix_written,
        "val_tokens_per_source": val_written, "val_duplicates_skipped_from_train": skipped_dupes,
        "train_documents": base_meta.get("train_documents", 0) + n_docs,
        # Documents consumed per source (validation, tokenizer sample and train); --extend skips these.
        "source_documents": {**base_meta.get("source_documents", {}), **docs_used},
        "bytes_per_token": n_bytes / max(written, 1),
        **({"extended_from": {"dir": args.extend, "train_tokens": base_train}, "new_train_tokens": written}
           if args.extend else {}),
        "mix_weights": mix, "mix_tokens": per_source,
        "sources": {k: f"{SOURCES[k][0]}/{SOURCES[k][1]}" for k in mix}, "dtype": "uint16",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    resume_file.unlink(missing_ok=True)
    print(json.dumps(meta, indent=2))
    if written < target:
        print(f"WARNING: ran out of data at {written:,} of {target:,} tokens")


if __name__ == "__main__":
    main()
