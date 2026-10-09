import sys

from .config import PRESETS, ModelConfig
from .model import LittleKediModel, ModelOutput

# Windows encodes redirected/piped stdout as cp1252, so printing generated text (or ✓/✗) to a log would crash.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8")

__all__ = ["ModelConfig", "PRESETS", "LittleKediModel", "ModelOutput"]
