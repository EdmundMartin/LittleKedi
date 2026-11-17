"""teacher.py: request format per provider, usage and cost accounting, and the --max-cost budget."""

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from teacher import BudgetExceeded, Teacher, add_teacher_args


class FakeOpenRouter(BaseHTTPRequestHandler):
    bodies: list[dict] = []
    headers_seen: list[dict] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenRouter.bodies.append(body)
        FakeOpenRouter.headers_seen.append({k.lower(): v for k, v in self.headers.items()})
        resp = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120, "cost": 0.4,
                          "completion_tokens_details": {"reasoning_tokens": 5}}}
        raw = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    FakeOpenRouter.bodies, FakeOpenRouter.headers_seen = [], []
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def parse(*argv: str) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    add_teacher_args(ap)
    return ap.parse_args(["--model", "vendor/teacher", *argv])


def ask(t: Teacher) -> str:
    return t.chat([{"role": "user", "content": "hello"}], temperature=0.5, max_tokens=50, retries=0)


def test_openrouter_reasoning_cost_and_budget(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    t = Teacher(parse("--provider", "openrouter", "--base-url", server, "--reasoning", "low", "--max-cost", "1.0"))
    assert ask(t) == "hi" and ask(t) == "hi" and ask(t) == "hi"  # 3 x $0.40 crosses the $1 budget
    with pytest.raises(BudgetExceeded):
        ask(t)
    assert len(FakeOpenRouter.bodies) == 3  # the 4th request was never sent
    body = FakeOpenRouter.bodies[0]
    assert body["reasoning"] == {"effort": "low", "exclude": True} and "chat_template_kwargs" not in body
    assert FakeOpenRouter.headers_seen[0]["authorization"] == "Bearer test-key"
    assert FakeOpenRouter.headers_seen[0]["x-title"] == "LittleKedi"
    assert t.usage == {"requests": 3, "prompt_tokens": 300, "completion_tokens": 60, "reasoning_tokens": 15,
                       "cost": pytest.approx(1.2)}
    assert "$1.200" in t.summary()


def test_openrouter_reasoning_off(server, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    ask(Teacher(parse("--provider", "openrouter", "--base-url", server, "--no-think")))
    assert FakeOpenRouter.bodies[0]["reasoning"] == {"effort": "none"}


def test_openrouter_needs_a_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setattr("cloud.load_dotenv", lambda *a, **k: None)  # don't pick up a real key from .env
    with pytest.raises(SystemExit, match="no OpenRouter key"):
        Teacher(parse("--provider", "openrouter"))


def test_compatible_endpoint_formats(server):
    ask(Teacher(parse("--base-url", server, "--no-think")))
    ask(Teacher(parse("--base-url", server, "--reasoning", "high")))
    ask(Teacher(parse("--base-url", server)))
    off, high, default = FakeOpenRouter.bodies
    assert off["chat_template_kwargs"] == {"enable_thinking": False} and "reasoning" not in off
    assert high["reasoning_effort"] == "high"
    assert not {"reasoning", "reasoning_effort", "chat_template_kwargs"} & set(default)
    t = Teacher(parse("--base-url", server, "--max-cost", "0.01"))  # no cost reports here: the budget is ignored
    for _ in range(3):
        ask(t)
    assert "$" not in t.summary()


def test_compatible_endpoint_needs_a_url():
    with pytest.raises(SystemExit, match="--base-url"):
        Teacher(parse())
