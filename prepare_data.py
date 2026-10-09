"""Build a pre-tokenized pretraining set for a preset, optionally mixing several sources.

Steps:
  1. Size the target: tokens = tokens_per_param (20, Chinchilla) x total model params,
     counted with the BPE vocab the preset uses (or --target-tokens).
  2. Stream documents from Hugging Face parquet shards (whole-file downloads into the
     HF cache; much faster than `datasets` streaming), only as many as needed.
  3. Train a byte-level BPE (digits split, chat tokens included) on a sample drawn from the
     mix in proportion to its weights.
  4. Hold out validation documents per source *before* any training documents are drawn, so no
     validation document is ever trained on: val_<source>.bin per source, and val.bin with all of
     them (in mix proportions). Then tokenize train.bin, interleaving sources so each keeps its
     share of tokens throughout.

Output: <out-dir>/{train.bin, val.bin, val_<source>.bin..., tokenizer.json, meta.json}, consumed by
    python train.py --data-dir <out-dir> --preset <preset>

    python prepare_data.py --preset small-sparse --mix fineweb-edu=0.6 cosmopedia=0.2 code=0.1 smoltalk=0.1
    python prepare_data.py --preset small-sparse --target-tokens 1e9 --mix fineweb-edu=0.7 finemath=0.1 code=0.1 smoltalk=0.1
    python prepare_data.py --list-sources

Conversations (smoltalk) are rendered with the chat special tokens, which every v2 tokenizer has.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from littlekedi import ModelConfig
from littlekedi.chat import render_chat_text
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
    "smoltalk": ("HuggingFaceTB/smoltalk", "data/all/train-", ["messages"],
                 lambda b: (render_chat_text(m) for m in b.column("messages").to_pylist())),
}


def iter_source(name: str, skip: int = 0) -> Iterator[str]:
    """Yield documents from a source's parquet shards, skipping the first `skip` (already used) ones."""
    repo, prefix, columns, to_docs = SOURCES[name]
    shards = sorted(
        s.rfilename for s in HfApi().dataset_info(repo).siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".parquet")
    )
    if not shards:
        raise SystemExit(f"no parquet shards for {name} ({repo}/{prefix})")
    for shard in shards:
        print(f"  [{name}] downloading {shard}", flush=True)
        local = hf_hub_download(repo, shard, repo_type="dataset")
        for batch in pq.ParquetFile(local).iter_batches(columns=columns, batch_size=1024):
            for text in to_docs(batch):
                if text:
                    if skip > 0:
                        skip -= 1
                        continue
                    yield text


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
    ap.add_argument("--val-tokens", type=float, default=4e6, help="held-out tokens in total, split by mix weight")
    ap.add_argument("--tokenizer-sample-mb", type=float, default=200.0)
    ap.add_argument("--out-dir", default=None, help="default: data/mix-<preset>")
    ap.add_argument("--tokenizer-path", default=None, help="reuse this tokenizer.json instead of training one")
    ap.add_argument("--skip-docs", nargs="*", default=[], metavar="SOURCE=N",
                    help="skip the first N documents of a source (e.g. ones an earlier dataset already used)")
    ap.add_argument("--extend", default=None, metavar="DIR",
                    help="grow an existing dataset: reuse its tokenizer and validation sets, skip the documents "
                         "it used, and write its train.bin + the new tokens to --out-dir. "
                         "--target-tokens is then the combined total")
    args = ap.parse_args()

    if args.list_sources:
        for name, (repo, prefix, _, _) in SOURCES.items():
            print(f"{name:18s} {repo}/{prefix}")
        return

    mix = parse_mix(args.mix)
    skips = {k: int(v) for k, _, v in (x.partition("=") for x in args.skip_docs)}
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
    val_quota = {name: int(args.val_tokens * w) for name, w in mix.items()}
    print(f"{args.preset}: {total_params / 1e6:.2f}M params (vocab {vocab}) -> target {target / 1e6:,.0f}M train "
          f"tokens + {args.val_tokens / 1e6:.0f}M val | mix " + ", ".join(f"{k} {v:.0%}" for k, v in mix.items()))

    if skips:
        print("skipping already-used documents: " + ", ".join(f"{k} {v:,}" for k, v in skips.items()))
    streams = {name: iter_source(name, skips.get(name, 0)) for name in mix}
    docs_used = {name: skips.get(name, 0) for name in mix}

    # 1) Validation documents first, per source, so the tokenizer sample and train.bin never see them.
    val_docs: dict[str, list[str]] = {name: [] for name in mix}
    # Rough sizing (~4 bytes/token) before the tokenizer exists; trimmed to the exact quota below.
    for name in mix:
        need_bytes = val_quota[name] * 4 * 1.25
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

    # 3) Write the validation sets.
    val_written = {}
    if args.extend:
        for f in Path(args.extend).glob("val*.bin"):
            shutil.copyfile(f, out / f.name)
        val_written = base_meta.get("val_tokens_per_source", {})
    else:
        with open(out / "val.bin", "wb") as all_val:
            for name in mix:
                ids = [t for seq in tok.encode_batch(val_docs[name]) for t in (*seq, eot)][: val_quota[name]]
                if len(ids) < val_quota[name]:
                    print(f"  WARNING: {name} validation set is short ({len(ids):,} of {val_quota[name]:,} tokens)")
                arr = np.asarray(ids, dtype=np.uint16).tobytes()
                (out / f"val_{name}.bin").write_bytes(arr)
                all_val.write(arr)
                val_written[name] = len(ids)
    del val_docs

    def next_batch(name: str, n: int = 64) -> list[str]:
        batch = buffered[name][:n]
        buffered[name] = buffered[name][n:]
        for text in streams[name]:
            if len(batch) >= n:
                break
            batch.append(text)
            docs_used[name] += 1
        return batch

    # 4) Train: always draw from the source furthest below its share of tokens written so far.
    written = 0
    per_source = {name: 0 for name in mix}
    exhausted: set[str] = set()
    n_docs, n_bytes, t0, last_report = 0, 0, time.time(), 0
    if args.extend:
        shutil.copyfile(Path(args.extend) / "train.bin", out / "train.bin")  # new tokens are appended after these
    with open(out / "train.bin", "ab" if args.extend else "wb") as f:
        while written < target and len(exhausted) < len(mix):
            total = sum(per_source.values()) + 1
            name = max((n for n in mix if n not in exhausted), key=lambda n: mix[n] * total - per_source[n])
            batch = next_batch(name)
            if not batch:
                exhausted.add(name)
                print(f"  WARNING: {name} ran out of data", flush=True)
                continue
            for ids in tok.encode_batch(batch):
                ids = (ids + [eot])[: target - written]
                f.write(np.asarray(ids, dtype=np.uint16).tobytes())
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

    meta = {
        "preset": args.preset, "vocab_size": tok.vocab_size, "params_total": total_params,
        "tokens_per_param": (base_train + written) / total_params,
        "train_tokens": base_train + written,
        "val_tokens": sum(val_written.values()), "val_tokens_per_source": val_written,
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
    print(json.dumps(meta, indent=2))
    if written < target:
        print(f"WARNING: ran out of data at {written:,} of {target:,} tokens")


if __name__ == "__main__":
    main()
