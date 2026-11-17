"""Pre-tokenized training tokens: one train.bin, or train-00000.bin, train-00001.bin, ... parts (Hugging Face
stores files of at most 50GB, so prepare_data.py splits big datasets) read as one array."""

from __future__ import annotations

import bisect
from pathlib import Path

import numpy as np


def train_parts(d: Path) -> list[Path]:
    parts = sorted(d.glob("train-[0-9][0-9][0-9][0-9][0-9].bin"))
    if parts:
        return parts
    if (d / "train.bin").exists():
        return [d / "train.bin"]
    raise SystemExit(f"{d} has no train.bin or train-NNNNN.bin parts")


class TokenParts:
    """Several uint16 memmaps addressed as one: len() and [a:b] slices, across part boundaries."""

    def __init__(self, parts: list[np.ndarray], limit: int | None = None):
        self.parts = parts
        self.offsets = [0]
        for p in parts:
            self.offsets.append(self.offsets[-1] + len(p))
        self.size = self.offsets[-1] if limit is None else min(limit, self.offsets[-1])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, s: slice):
        if not isinstance(s, slice) or s.step not in (None, 1):
            raise TypeError("TokenParts supports contiguous slices only")
        start, stop, _ = s.indices(self.size)
        if s.start is None and stop < self.size:  # data[:n]: a shorter view, as for a memmap
            return TokenParts(self.parts, stop)
        out, i = [], max(bisect.bisect_right(self.offsets, start) - 1, 0)
        while start < stop:
            lo = start - self.offsets[i]
            hi = min(stop, self.offsets[i + 1]) - self.offsets[i]
            out.append(self.parts[i][lo:hi])
            start = self.offsets[i] + hi
            i += 1
        return out[0] if len(out) == 1 else np.concatenate(out)


def load_train_tokens(d: Path):
    """train.bin as a memmap, or its parts as one TokenParts."""
    parts = [np.memmap(p, dtype=np.uint16, mode="r") for p in train_parts(Path(d))]
    return parts[0] if len(parts) == 1 else TokenParts(parts)
