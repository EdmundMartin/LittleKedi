"""The teacher-model client shared by the SFT data generators (gen_sft_data.py, gen_sft_code.py).

Any OpenAI-compatible endpoint works: vLLM, Ollama, LM Studio, llama.cpp server, OpenAI. --provider openrouter
fills in OpenRouter's endpoint and key (OPENROUTER_API_KEY, from the environment or .env), sends reasoning
settings in OpenRouter's own format, and adds up the cost OpenRouter reports with every reply; --max-cost stops
starting new requests once the run has spent that much (requests already in flight still finish).

    python gen_sft_data.py --provider openrouter --model <model-id> --reasoning none --max-cost 5 ...
    python gen_sft_data.py --base-url http://localhost:11434/v1 --model qwen2.5:14b ...
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from openai import OpenAI

OPENROUTER_URL = "https://openrouter.ai/api/v1"
# OpenRouter's optional app attribution (shown on its usage pages).
OPENROUTER_HEADERS = {"HTTP-Referer": "https://github.com/EdmundMartin/LittleKedi", "X-Title": "LittleKedi"}
REASONING = ["default", "none", "minimal", "low", "medium", "high"]


class BudgetExceeded(RuntimeError):
    pass


def add_teacher_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--provider", choices=["compatible", "openrouter"], default="compatible",
                    help="openrouter: OpenRouter's endpoint, key and reasoning format, with cost tracking")
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1 (default for openrouter: its API)")
    ap.add_argument("--model", required=True, help="teacher model id (OpenRouter ids look like vendor/model)")
    ap.add_argument("--api-key-env", default=None,
                    help="env var holding the key (default OPENROUTER_API_KEY or OPENAI_API_KEY; local servers: any)")
    ap.add_argument("--api-key-file", default=None, help="file containing just the key (overrides --api-key-env)")
    ap.add_argument("--reasoning", choices=REASONING, default="default",
                    help="reasoning effort for thinking models; none turns it off (it can eat the token budget). "
                         "Local servers: none sets chat_template_kwargs.enable_thinking=false (Qwen3-style), other "
                         "levels are sent as reasoning_effort")
    ap.add_argument("--no-think", action="store_true", help="same as --reasoning none")
    ap.add_argument("--max-cost", type=float, default=None,
                    help="openrouter: stop starting new requests once the run has cost this many US dollars")


class Teacher:
    """A thread-safe chat client that retries, and keeps a running total of usage (and cost, on OpenRouter)."""

    def __init__(self, args: argparse.Namespace):
        self.model = args.model
        self.openrouter = args.provider == "openrouter"
        base_url = args.base_url or (OPENROUTER_URL if self.openrouter else None)
        if base_url is None:
            raise SystemExit("--base-url is required (or --provider openrouter)")
        if args.api_key_file:
            api_key = Path(args.api_key_file).read_text(encoding="utf-8").strip()
        else:
            from cloud import load_dotenv
            load_dotenv()  # .env is gitignored; keys never go on the command line
            env = args.api_key_env or ("OPENROUTER_API_KEY" if self.openrouter else "OPENAI_API_KEY")
            api_key = os.environ.get(env) or ""
            if self.openrouter and not api_key:
                raise SystemExit(f"no OpenRouter key: set {env} in the environment or in .env")
        self.client = OpenAI(base_url=base_url, api_key=api_key or "not-needed",
                             default_headers=OPENROUTER_HEADERS if self.openrouter else None)
        self.max_cost = args.max_cost
        if self.max_cost is not None and not self.openrouter:
            print("note: --max-cost needs cost reports, which only --provider openrouter has; ignoring it")
            self.max_cost = None
        effort = "none" if args.no_think else args.reasoning
        self.extra_body: dict = {}
        if self.openrouter and effort != "default":
            # exclude: the model still reasons, but its reasoning isn't sent back (we only want the answer).
            self.extra_body["reasoning"] = {"effort": effort} if effort == "none" else {"effort": effort, "exclude": True}
        elif effort == "none":
            self.extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        elif effort != "default":
            self.extra_body["reasoning_effort"] = effort
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost": 0.0}
        self._lock = threading.Lock()

    def chat(self, messages: list[dict], *, temperature: float, max_tokens: int, retries: int = 8, **kwargs) -> str:
        """The reply's text. Rides out endpoint outages and rate limits with exponential backoff (~5 min)."""
        for attempt in range(retries + 1):
            self._check_budget()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                    extra_body=self.extra_body or None, **kwargs)
                content = resp.choices[0].message.content  # OpenRouter can return an error body with no choices
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(min(60, 2 ** attempt * 2))
                continue
            self._record(resp.usage)
            return content or ""
        raise AssertionError("unreachable")

    def _check_budget(self) -> None:
        if self.max_cost is not None and self.usage["cost"] >= self.max_cost:
            raise BudgetExceeded(f"spent ${self.usage['cost']:.2f} of the ${self.max_cost:.2f} budget")

    def _record(self, usage) -> None:
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        with self._lock:
            self.usage["requests"] += 1
            self.usage["prompt_tokens"] += usage.prompt_tokens or 0
            self.usage["completion_tokens"] += usage.completion_tokens or 0
            self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", None) or 0
            self.usage["cost"] += float(getattr(usage, "cost", None) or 0)  # OpenRouter's extra field, in USD

    def summary(self) -> str:
        u = self.usage
        reasoning = f" ({u['reasoning_tokens'] / 1e6:.2f}M reasoning)" if u["reasoning_tokens"] else ""
        cost = f"  ${u['cost']:.3f}" if self.openrouter else ""
        return (f"{u['requests']:,} requests  {u['prompt_tokens'] / 1e6:.2f}M in / "
                f"{u['completion_tokens'] / 1e6:.2f}M out{reasoning}{cost}")
