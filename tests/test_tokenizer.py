from littlekedi.chat import encode_conversation, render_chat_text
from littlekedi.tokenizer import CHAT_TOKENS, BPETokenizer, encode_files, load_tokenizer


def train_tok(tmp_path, vocab=400):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text("the quick brown fox jumps over the lazy dog in 2024 and 1999\n" * 200, encoding="utf-8")
    b.write_text("héllo wörld — ünïcode 🙂 12345\n" * 50, encoding="utf-8")
    return BPETokenizer.train([str(a), str(b)], vocab_size=vocab), [str(a), str(b)]


def test_bpe_trains_roundtrips_and_serializes(tmp_path):
    tok, files = train_tok(tmp_path)
    assert 256 < tok.vocab_size <= 400
    text = "the lazy fox says héllo 🙂\n  indented 3.14159"
    assert tok.decode(tok.encode(text)) == text
    assert len(tok.encode("the quick brown fox")) < len("the quick brown fox")  # merges learned
    assert load_tokenizer(tok.to_str()).encode(text) == tok.encode(text)
    ids = encode_files(tok, files)
    assert (ids == tok.eot_id).sum() == 1


def test_digits_are_split_individually(tmp_path):
    tok, _ = train_tok(tmp_path)
    ids = tok.encode("2024")
    assert len(ids) == 4 and [tok.decode([i]) for i in ids] == ["2", "0", "2", "4"]


def test_chat_tokens_exist_from_the_start(tmp_path):
    tok, _ = train_tok(tmp_path)
    for t in CHAT_TOKENS:
        assert tok.token_to_id(t) is not None
    convo = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert len(tok.encode(render_chat_text(convo))) == len(tok.encode("hi")) + len(tok.encode("hello")) + 4
    ids, labels = encode_conversation(tok, convo, max_len=64)
    assert len(ids) == len(labels) and labels[-1] == tok.token_to_id("<|end|>")
