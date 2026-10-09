import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from train import WindowSampler

ROOT = Path(__file__).resolve().parents[1]


def test_sampler_covers_each_window_once_per_epoch_and_is_resumable():
    s = WindowSampler(n_tokens=10 * 8 + 1, seq_len=8, batch_size=2, seed=0)
    assert s.n_windows == 10
    epoch0 = [x for step in range(5) for x in s.starts(step)]
    assert sorted(epoch0) == list(range(0, 80, 8))
    assert WindowSampler(81, 8, 2, seed=0).starts(3) == s.starts(3)
    assert np.all(np.array(s.starts(7)) <= 80 - 8)


def _train(tmp_path, *extra):
    corpus = tmp_path / "c.txt"
    if not corpus.exists():
        corpus.write_text("The quick brown fox jumps over the lazy dog 42 times. " * 600, encoding="utf-8")
    cmd = [sys.executable, "train.py", "--data", str(corpus), "--preset", "tiny-sparse", "--vocab-size", "300",
           "--set", "max_seq_len=64", "--batch-size", "2", "--grad-accum", "2", "--warmup", "2",
           "--eval-every", "3", "--eval-iters", "1", "--log-every", "1", "--save-every", "2",
           "--out", str(tmp_path / "m.pt"), *extra]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return r.stdout


def test_train_script_resumes_appends_log_and_keeps_milestones(tmp_path):
    out = _train(tmp_path, "--steps", "4", "--precision", "bf16", "--milestones", "512")
    assert "muon + wsd" in out and "drop_rate" in out and "tok/s" in out
    assert (tmp_path / "m-0M.pt").exists()  # 512 tokens = the first step (256 tokens/step x 2)
    assert torch.load(tmp_path / "m.pt")["step"] == 3

    _train(tmp_path, "--steps", "6", "--resume")
    assert torch.load(tmp_path / "m.pt")["step"] == 5
    records = [json.loads(l) for l in (tmp_path / "m.log.jsonl").read_text().splitlines()]
    assert [r["event"] for r in records].count("start") == 2           # both runs in one log
    assert {r["step"] for r in records if r["event"] == "train"} == set(range(6))
    evals = [r for r in records if r["event"] == "eval"]
    assert evals and "val_dead_experts" in evals[-1]
