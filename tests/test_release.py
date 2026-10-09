"""prepare_release.py: base vs instruct detection, and GGUF only for instruct releases."""
import subprocess
import sys
from pathlib import Path

import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parents[1]


def run(cwd, *args):
    r = subprocess.run([sys.executable, str(ROOT / "prepare_release.py"), *args], cwd=cwd, capture_output=True,
                       text=True, timeout=300, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_base_and_instruct_releases(tmp_path):
    tok = BPETokenizer.train_from_texts(["The cat sat on the mat. " * 200], 300)
    cfg = ModelConfig.from_preset("tiny", vocab_size=tok.vocab_size, max_seq_len=64)
    state = LittleKediModel(cfg).state_dict()
    (tmp_path / "checkpoints").mkdir()
    base = {"config": cfg.to_dict(), "model": state, "step": 9, "tokens": 1000, "preset": "tiny", "tokenizer": tok.to_str()}
    torch.save(base, tmp_path / "checkpoints" / "tiny.pt")
    sft = {"conversations": 10, "train": 9, "val": 1, "assistant_tokens": 500, "steps": 3, "optimizer": "muon",
           "lr": 2e-4, "epochs": 1, "batch_size": 4, "data": ["chat.jsonl"], "val_before": 5.0, "val_after": 4.0}
    torch.save({**base, "chat": True, "base": "checkpoints/tiny.pt", "sft": sft}, tmp_path / "checkpoints" / "tiny-sft.pt")

    out = run(tmp_path, "--ckpt", "checkpoints/tiny.pt")
    assert "LittleKedi-tiny-base (base)" in out
    b = tmp_path / "release" / "LittleKedi-tiny-base"
    assert not list(b.glob("*.gguf")) and not (b / "chat_template.jinja").exists()
    card = (b / "README.md").read_text(encoding="utf-8")
    assert "This is the **base model**" in card and "- base\n" in card and "## Chat format" not in card

    out = run(tmp_path, "--ckpt", "checkpoints/tiny-sft.pt", "--gguf", "q8_0", "--base-repo", "u/LittleKedi-tiny-base")
    assert "LittleKedi-tiny-instruct (instruct)" in out
    i = tmp_path / "release" / "LittleKedi-tiny-instruct"
    assert (i / "LittleKedi-tiny-instruct-q8_0.gguf").exists()
    assert all((i / f).exists() for f in ("chat_template.jinja", "tokenizer_config.json", "template", "params"))
    card = (i / "README.md").read_text(encoding="utf-8")
    assert "instruction-tuned (chat) version of [LittleKedi-tiny-base](https://huggingface.co/u/LittleKedi-tiny-base)" in card
    assert "base_model: u/LittleKedi-tiny-base" in card and "- instruction-tuned" in card
    assert "## Chat format" in card and "## Fine-tuning (SFT)" in card and "**5.000 → 4.000**" in card
    assert "### Base vs instruct" in card and "llama-cli -m LittleKedi-tiny-instruct-q8_0.gguf" in card
