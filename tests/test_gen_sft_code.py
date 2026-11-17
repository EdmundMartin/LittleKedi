import json

from gen_sft_code import looks_like_python, parse_exercise, passes, safe_to_run

GOOD = {"task": "Write a Python function `count_vowels(s)` that returns the number of vowels in `s`.",
        "solution": "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
        "tests": ["assert count_vowels('') == 0", "assert count_vowels('Hello') == 2", "assert count_vowels('xyz') == 0"]}


def test_parse_exercise_takes_the_last_object_and_rejects_null():
    reply = 'Template: {"exercise": {"task": "...", "solution": "def ...", "tests": []}}\n' + json.dumps({"exercise": GOOD})
    assert parse_exercise(f"<think>hmm</think>```json\n{reply}\n```") == GOOD
    assert parse_exercise('{"exercise": null}') is None
    assert parse_exercise("not json") is None


def test_safe_to_run_blocks_imports_io_and_file_references():
    assert safe_to_run(GOOD) is None
    assert safe_to_run({**GOOD, "solution": "import os\n" + GOOD["solution"]}) == "import"
    assert safe_to_run({**GOOD, "solution": "from collections import Counter\n" + GOOD["solution"]}) is None
    assert safe_to_run({**GOOD, "solution": GOOD["solution"] + "\nprint(1)"}) == "forbidden call"
    assert safe_to_run({**GOOD, "task": "Rewrite the function from the file to count vowels in a string."}) == "refers to the file"


def test_passes_runs_the_asserts():
    assert passes(GOOD)
    assert not passes({**GOOD, "tests": ["assert count_vowels('Hello') == 3"]})
    assert not passes({**GOOD, "solution": "def count_vowels(s):\n    while True: pass"}, timeout=1)


def test_looks_like_python():
    py = "import math\n\n" + "\n".join(f"def f{i}(x):\n    return math.sqrt(x) + {i}\n" for i in range(12))
    js = "\n".join(f"function f{i}(x) {{\n  return x + {i};\n}}" for i in range(10))
    assert looks_like_python(py) and not looks_like_python(js)


def test_parse_tests_and_from_row_round_trip():
    from gen_sft_code import from_row, parse_tests, to_messages
    reply = '```python\nassert f("") == 1\nf(2)\n  assert f(0) == 1\n```'
    assert parse_tests(reply) == ['assert f("") == 1', "assert f(0) == 1"]
    assert parse_tests("no json") == []
    assert from_row({"messages": to_messages(GOOD), "tests": GOOD["tests"]}) == GOOD
