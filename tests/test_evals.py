import torch

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer
from evals import Scorer, _hellaswag_clean

TEXT = "The quick brown fox jumps over the lazy dog. Water freezes at zero degrees. " * 40


def _scorer(tmp_path, batch_size):
    tok = BPETokenizer.train_from_texts([TEXT], 300)
    cfg = ModelConfig.from_preset("tiny", vocab_size=tok.vocab_size, n_layers=2, max_seq_len=48)
    torch.manual_seed(0)
    model = LittleKediModel(cfg)
    path = tmp_path / "m.pt"
    torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": 7,
                "tokenizer": tok.to_str()}, path)
    return Scorer(str(path), "cpu", batch_size)


ITEMS = [
    {"prompt": "Water freezes at", "choices": ["zero degrees.", "fifty degrees.", "the dog."], "answer": 0},
    {"prompt": "The quick brown fox " * 6, "choices": ["jumps over the lazy dog.", "is blue."], "answer": 0},
    {"prompt": "The lazy", "choices": ["dog.", "fox jumps.", "water", "brown"], "answer": 0},
]


def test_batched_scores_match_one_at_a_time(tmp_path):
    reqs = [r for item in ITEMS for r in _scorer(tmp_path, 1).requests(item)]
    one = _scorer(tmp_path, 1).logprobs(reqs)
    batched = _scorer(tmp_path, 8).logprobs(reqs)
    assert torch.allclose(torch.tensor(one), torch.tensor(batched), atol=1e-4)


def test_long_context_is_cut_from_the_left_to_fit(tmp_path):
    sc = _scorer(tmp_path, 4)
    for ctx, cont in sc.requests(ITEMS[1]):
        assert len(ctx) + len(cont) - 1 <= sc.max_len
    r = sc.run(ITEMS)
    assert r["n"] == 3 and 0 <= r["acc"] <= 1 and 0 <= r["acc_norm"] <= 1


def test_hellaswag_cleanup():
    assert _hellaswag_clean(" Making tea [title] Boil water. [step] Pour it.  ") == "Making tea. Boil water. Pour it."
