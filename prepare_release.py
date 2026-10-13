"""Build a release folder (weights, config, tokenizer, model card) for one LittleKedi checkpoint.

Base or instruct is read from the checkpoint: sft.py marks its output as a chat model and records its base.

    python prepare_release.py --ckpt checkpoints/tiny.pt                     # -> release/LittleKedi-tiny-base/
    python prepare_release.py --ckpt checkpoints/tiny-sft.pt --repo <user>/LittleKedi-tiny-instruct \
        --base-repo <user>/LittleKedi-tiny-base                             # -> release/LittleKedi-tiny-instruct/
    python prepare_release.py --ckpt checkpoints/tiny.pt --card-only         # just the card -> checkpoints/tiny-card.md
    hf upload <user>/LittleKedi-tiny-base release/LittleKedi-tiny-base . --repo-type model   # when you're happy

Base releases are PyTorch weights only. Instruct releases also get GGUF files (f16 and q8_0 by default),
the chat template (chat_template.jinja, tokenizer_config.json) and Ollama `template` / `params` files.

The card (model_card.py) is shared by every size and variant; its numbers come from the checkpoint, its
training log (<ckpt>.log.jsonl), the dataset's meta.json and the eval logs for this checkpoint's step:

    python evals.py --ckpt checkpoints/tiny.pt --log evals.jsonl
    python mmlu_eval.py --ckpt checkpoints/tiny.pt --log mmlu_results.jsonl

Sibling variants (checkpoints/<size>, <size>-sparse, <size>-wide-sparse .pt) are added to the comparison
table automatically when they exist; --compare adds any other checkpoints.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.config import VARIANTS
from littlekedi.tokenizer import USER, load_tokenizer

from model_card import (FAMILY, LOGO, BASELINES, build_card, bio_med_acc, infer_preset, merged_results, model_name,
                        read_run_log, size_and_variant)


OLLAMA_TEMPLATE = """{{- range .Messages }}
{{- if eq .Role "system" }}<|system|>{{ .Content }}<|end|>
{{- else if eq .Role "user" }}<|user|>{{ .Content }}<|end|>
{{- else if eq .Role "assistant" }}<|assistant|>{{ .Content }}<|end|>
{{- end }}
{{- end }}<|assistant|>"""
OLLAMA_PARAMS = {"stop": ["<|end|>", "<|user|>", "<|endoftext|>"], "temperature": 0.7, "top_p": 0.9,
                 "repeat_penalty": 1.15}


def build_gguf_repo(out: Path, name: str, md_args: dict, main_repo: str, gguf_repo: str,
                    gguf_files: list[str]) -> Path:
    """release/<name>-GGUF: GGUF files only, so the Hub shows its GGUF panel and llama.cpp / Ollama / LM Studio
    buttons (a repo that also holds safetensors is shown as a safetensors repo instead)."""
    gg = out.parent / f"{name}-GGUF"
    gg.mkdir(parents=True, exist_ok=True)
    extras = [("template", "Ollama chat template"), ("params", "Ollama stop tokens and sampling defaults"),
              ("chat_template.jinja", "Chat template (Jinja), the same one embedded in the GGUF files"),
              (LOGO, f"The {FAMILY} logo")]
    files = [(g, f"GGUF ({g.rsplit('-', 1)[1][:-5]}) for llama.cpp / Ollama / LM Studio") for g in gguf_files]
    for f, desc in extras:
        if (out / f).exists():
            files.append((f, desc))
    for f, _ in files:
        shutil.copyfile(out / f, gg / f)
    md = build_card(**md_args, files=files, repo=gguf_repo)
    _, front, body = md.split("---\n", 2)
    front = "".join(l + "\n" for l in front.splitlines()
                    if not l.startswith(("base_model:", "base_model_relation:", "library_name:")))
    front += (f"library_name: gguf\nbase_model: {main_repo}\nbase_model_relation: quantized\n"
              f"quantized_by: {gguf_repo.split('/')[0]}\n")
    front = front.replace("tags:\n", "tags:\n- gguf\n", 1)
    body = body.replace(f"# {name}\n", f"# {name}-GGUF\n\nGGUF builds (f16 and q8_0) of "
                        f"[{name}](https://huggingface.co/{main_repo}) for llama.cpp, Ollama and LM Studio. The PyTorch "
                        "weights (`model.safetensors`), config and tokenizer are in the main repo; the card below "
                        "describes the model.\n", 1)
    (gg / "README.md").write_text(f"---\n{front}---\n{body}", encoding="utf-8")
    return gg


def run_log_path(ckpt: str) -> Path:
    """<ckpt>.log.jsonl; milestone checkpoints (<run>-750M.pt) share their run's log (<run>.log.jsonl)."""
    path = Path(ckpt).with_suffix(".log.jsonl")
    if not path.exists() and (m := re.fullmatch(r"(.+)-\d+M", Path(ckpt).stem)):
        path = Path(ckpt).with_name(f"{m.group(1)}.log.jsonl")
    return path


def checkpoint_column(ckpt: str, label: str | None = None) -> tuple | None:
    """One comparison column for a LittleKedi checkpoint, from its own logs."""
    if not Path(ckpt).exists():
        return None
    ck = torch.load(ckpt, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    preset = ck.get("preset") or infer_preset(cfg) or Path(ckpt).stem
    step, est = ck.get("step"), cfg.param_estimate()
    evals = merged_results("evals.jsonl", ckpt, step)
    mmlu = merged_results("mmlu_results.jsonl", ckpt, step)
    log = read_run_log(run_log_path(ckpt))
    val = log["eval"][-1].get("val_loss") if log["eval"] else None
    cloze = mmlu.get("cloze")
    bio = bio_med_acc(cloze) if cloze else None
    tokens = ck.get("tokens") or 0
    return (label or f"`{preset}`", est["total"], est["activated_non_embedding"], f"{tokens / 1e6:,.0f}M",
            evals, cloze["accuracy"] if cloze else None, bio[0] if bio else None, val)


def val_bytes_per_token(data_dir: str | None, tok) -> float | None:
    """Bits/byte needs the validation set's own bytes per token."""
    if not data_dir or not (Path(data_dir) / "val.bin").exists():
        return None
    ids = np.fromfile(Path(data_dir) / "val.bin", dtype=np.uint16)
    text_ids = ids[ids != tok.eot_id]
    return len(tok.decode(text_ids.tolist()).encode("utf-8")) / len(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/tiny.pt")
    ap.add_argument("--card-only", action="store_true", help="write only the card, to <ckpt>-card.md")
    ap.add_argument("--compare", nargs="*", default=[], help="extra checkpoints for the comparison table")
    ap.add_argument("--hardware", default="One AMD Radeon 890M integrated GPU (Ryzen AI 300 laptop), ROCm on Windows")
    ap.add_argument("--capabilities", default=None, help="markdown file for the 'What it can and can't do' section")
    ap.add_argument("--code-url", default=None, help="link to the training code (e.g. your GitHub repo)")
    ap.add_argument("--out-dir", default=None, help="default: release/<name>")
    ap.add_argument("--repo", default=None, help="target HF repo id (shown in the card's usage)")
    ap.add_argument("--base-repo", default=None, help="instruct cards: HF repo of the base model, linked from the card")
    ap.add_argument("--gguf-repo", default=None,
                    help="instruct releases: also build release/<name>-GGUF (GGUF files only) for this repo id, "
                         "linked to --repo as a quantization so the Hub shows the GGUF panel")
    ap.add_argument("--sft-data", default=None,
                    help="instruct cards: a markdown description of the SFT data (default: the file names)")
    ap.add_argument("--base-ckpt", default=None, help="instruct cards: base checkpoint (default: the one sft.py recorded)")
    ap.add_argument("--gguf", nargs="*", default=["f16", "q8_0"], choices=["f32", "f16", "q8_0"],
                    help="GGUF builds for instruct releases (base releases never include GGUF)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig.from_dict(ck["config"])
    chat = bool(ck.get("chat"))
    base_ckpt = args.base_ckpt or ck.get("base") if chat else None
    preset = ck.get("preset") or infer_preset(cfg)
    if preset is None and base_ckpt and Path(base_ckpt).exists():  # SFT grew the vocab: match on the base
        preset = infer_preset(ModelConfig.from_dict(torch.load(base_ckpt, map_location="cpu")["config"]))
    if preset is None:
        raise SystemExit(f"{args.ckpt}: config doesn't match any preset; this card is for the preset family")
    step, tokens = ck.get("step"), ck.get("tokens")
    # Pretraining details live with the base model's run, which an instruct checkpoint points back to.
    log = read_run_log(run_log_path(base_ckpt if chat and base_ckpt else args.ckpt),
                       up_to_step=None if chat else step)
    start_args = (log["start"] or {}).get("args", {})
    if tokens is None and start_args:  # older checkpoints: tokens from the step and the logged batch shape
        tokens = (step + 1) * start_args["batch_size"] * start_args["seq_len"] * start_args.get("grad_accum", 1)
    tokens = tokens or 0
    planned_steps = ((log["latest_start"] or {}).get("args") or {}).get("steps")
    in_progress = None
    if not chat and planned_steps and isinstance(step, int) and step + 1 < planned_steps:
        per_step = tokens / (step + 1)
        in_progress = {"tokens": tokens, "planned": planned_steps * per_step}
    data_dir = start_args.get("data_dir")
    meta_path = Path(data_dir) / "meta.json" if data_dir else None
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path and meta_path.exists() else {}
    tok = load_tokenizer(ck["tokenizer"])
    tok_info = {"vocab": tok.vocab_size, "digits": '"Digits"' in tok.to_str(), "chat": tok.token_to_id(USER) is not None}
    counts = LittleKediModel(cfg).param_counts()

    name = model_name(preset, chat)
    out = Path(args.out_dir or f"release/{name}")
    logo_src = Path(__file__).resolve().parent / LOGO
    if args.card_only:  # the card sits next to the checkpoint and points back at the repo's logo
        logo = Path(os.path.relpath(logo_src, Path(args.ckpt).resolve().parent)).as_posix() if logo_src.exists() else None
    else:
        logo = LOGO if logo_src.exists() else None
    gguf_files = [f"{name}-{d}.gguf" for d in args.gguf] if chat else []
    files = [("model.safetensors", "PyTorch weights (embedding tied with the output head), `littlekedi` parameter names"
              if cfg.tie_embeddings else "PyTorch weights, `littlekedi` parameter names"),
             ("config.json", "`ModelConfig` for `littlekedi`"),
             ("tokenizer.json", "Byte-level BPE tokenizer (Hugging Face `tokenizers` format)")]
    if chat:
        files += [("chat_template.jinja", "Chat template (Jinja), the same one embedded in the GGUF files"),
                  ("tokenizer_config.json", "Special tokens (`<|end|>` = end of turn), chat template, context length"),
                  ("template", "Ollama chat template"), ("params", "Ollama stop tokens and sampling defaults")]
        files += [(g, f"GGUF ({g.rsplit('-', 1)[1][:-5]}) for llama.cpp / Ollama / LM Studio") for g in gguf_files]
    if logo and not args.card_only:
        files.append((LOGO, f"The {FAMILY} logo"))

    # Comparison: this model first, then its sibling variants, any --compare checkpoints, then published baselines.
    size, _ = size_and_variant(preset)
    ckpt_dir = Path(args.ckpt).parent
    others = [str(ckpt_dir / f"{size}{v}.pt") for v in VARIANTS]
    skip = {Path(args.ckpt).resolve()} | ({Path(base_ckpt).resolve()} if base_ckpt else set())
    if m := re.fullmatch(r"(.+)-\d+M", Path(args.ckpt).stem):  # a milestone: its own run isn't a sibling
        skip.add(Path(args.ckpt).with_name(f"{m.group(1)}.pt").resolve())
    others = [p for p in dict.fromkeys(others + args.compare) if Path(p).resolve() not in skip]
    columns = [checkpoint_column(args.ckpt, "**This model**")]
    columns += [c for c in (checkpoint_column(p) for p in others) if c]
    columns += [(link, total, nonemb, data, published, None, None, None) for link, total, nonemb, data, published in BASELINES]

    def results(path, at_step) -> dict:
        mm = merged_results("mmlu_results.jsonl", path, at_step)
        ev = merged_results("evals.jsonl", path, at_step)
        return {"evals": {k: v for k, v in ev.items() if isinstance(v, dict)},
                "mmlu": {k: v for k, v in mm.items() if k in ("letter", "cloze")}}

    mine = results(args.ckpt, step)
    base_results = None
    if chat and base_ckpt and Path(base_ckpt).exists():
        base_results = results(base_ckpt, torch.load(base_ckpt, map_location="cpu").get("step"))
    md = build_card(
        preset=preset, cfg=cfg, counts=counts, step=step, tokens=tokens, log=log, meta=meta, tok_info=tok_info,
        val_bpt=val_bytes_per_token(data_dir, tok), evals=mine["evals"], mmlu=mine["mmlu"],
        columns=columns, files=files, hardware=args.hardware,
        capabilities=Path(args.capabilities).read_text(encoding="utf-8").strip() if args.capabilities else None,
        code_url=args.code_url, ckpt=args.ckpt, logo=logo, chat=chat,
        sft={**(ck.get("sft") or {}), **({"data_note": args.sft_data} if args.sft_data else {})},
        base_results=base_results, inst_results=mine if chat else None, base_repo=args.base_repo,
        gguf=gguf_files, repo=args.repo, in_progress=in_progress)

    if args.card_only:
        path = Path(args.ckpt).with_name(f"{Path(args.ckpt).stem}-card.md")
        path.write_text(md, encoding="utf-8")
        print(f"wrote {path}")
    else:
        from safetensors.torch import save_file
        out.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in ck["model"].items()}, out / "model.safetensors",
                  metadata={"format": "pt", "step": str(step), "preset": preset, "kind": "instruct" if chat else "base"})
        (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
        (out / "tokenizer.json").write_text(ck["tokenizer"], encoding="utf-8")
        (out / "README.md").write_text(md, encoding="utf-8")
        if logo:
            shutil.copyfile(logo_src, out / LOGO)
        if chat:
            from export_gguf import CHAT_TEMPLATE
            from littlekedi.tokenizer import CHAT_TOKENS, END, EOT
            (out / "chat_template.jinja").write_text(CHAT_TEMPLATE, encoding="utf-8")
            (out / "tokenizer_config.json").write_text(json.dumps({
                "tokenizer_class": "PreTrainedTokenizerFast", "chat_template": CHAT_TEMPLATE,
                "bos_token": None, "add_bos_token": False,  # SFT conversations start without <|endoftext|>
                "eos_token": END, "pad_token": EOT,
                "additional_special_tokens": [t for t in CHAT_TOKENS if t != END],
                "model_max_length": cfg.max_seq_len}, indent=2), encoding="utf-8")
            (out / "template").write_text(OLLAMA_TEMPLATE, encoding="utf-8")
            (out / "params").write_text(json.dumps({**OLLAMA_PARAMS, "num_ctx": cfg.max_seq_len}, indent=2), encoding="utf-8")
            for g, dtype in zip(gguf_files, args.gguf):
                subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "export_gguf.py"), "--ckpt",
                                args.ckpt, "--dtype", dtype, "--name", name, "--out", str(out / g)], check=True)
        print(f"release folder: {out}")
        for p in sorted(out.iterdir()):
            print(f"  {p.name:28s} {p.stat().st_size / 1e6:8.2f} MB")
        print(f"\nto publish (after `hf auth login`):\n  hf upload {args.repo or '<user>/' + name} {out} . --repo-type model")
        if chat and args.gguf_repo:
            gg = build_gguf_repo(out, name, md_args=dict(
                preset=preset, cfg=cfg, counts=counts, step=step, tokens=tokens, log=log, meta=meta,
                tok_info=tok_info, val_bpt=val_bytes_per_token(data_dir, tok), evals=mine["evals"],
                mmlu=mine["mmlu"], columns=columns, hardware=args.hardware,
                capabilities=Path(args.capabilities).read_text(encoding="utf-8").strip() if args.capabilities else None,
                code_url=args.code_url, ckpt=args.ckpt, logo=logo, chat=chat,
                sft={**(ck.get("sft") or {}), **({"data_note": args.sft_data} if args.sft_data else {})},
                base_results=base_results, inst_results=mine, base_repo=args.base_repo, gguf=gguf_files,
                in_progress=in_progress), main_repo=args.repo, gguf_repo=args.gguf_repo, gguf_files=gguf_files)
            print(f"\nGGUF repo folder: {gg}")
            for p in sorted(gg.iterdir()):
                print(f"  {p.name:40s} {p.stat().st_size / 1e6:8.2f} MB")
            print(f"  hf upload {args.gguf_repo} {gg} . --repo-type model")
    found = [k for k, v in mine.items() if v]
    progress = f" (IN PROGRESS: {in_progress['tokens'] / in_progress['planned']:.0%} of planned tokens)" if in_progress else ""
    print(f"card for {name} ({'instruct' if chat else 'base'}) at step {step}{progress}: evaluations found: {', '.join(found) or 'none'}"
          + ("" if args.capabilities else "  (capabilities section is a TODO; pass --capabilities FILE)"))


if __name__ == "__main__":
    main()
