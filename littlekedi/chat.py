"""Chat template and SFT loss masking.

A conversation is rendered as

    [<|system|> ... <|end|>] <|user|> ... <|end|> <|assistant|> ... <|end|> <|user|> ...

Role markers are dedicated special tokens, present in every v2 tokenizer from the start, so
conversations in the pretraining mix and in SFT use the same tokens. During SFT only the
assistant's reply tokens and its closing <|end|> are trained on; everything the model is
*given* (system, user, role markers) is masked out of the loss. Learning to emit <|end|> is
what teaches the model to stop.
"""

from __future__ import annotations

from .tokenizer import ASSISTANT, CHAT_TOKENS, END, SYSTEM, USER, BPETokenizer

ROLE_TOKENS = {"system": SYSTEM, "user": USER, "assistant": ASSISTANT}
IGNORE = -100  # label value ignored by the loss


def add_chat_tokens(tok: BPETokenizer) -> int:
    """Make sure the chat tokens exist (v2 tokenizers already have them; v1 ones don't). Returns how many were added."""
    return tok.add_special_tokens(CHAT_TOKENS)


def render_chat_text(messages: list[dict]) -> str:
    """A conversation as text with the special tokens inline (for the pretraining mix)."""
    return "".join(f"{ROLE_TOKENS[m['role']]}{m['content']}{END}" for m in messages if m["role"] in ROLE_TOKENS)


def _ids(tok: BPETokenizer, token: str) -> int:
    i = tok.token_to_id(token)
    if i is None:
        raise ValueError(f"tokenizer has no {token}; call add_chat_tokens() first")
    return i


def encode_conversation(
    tok: BPETokenizer, messages: list[dict], max_len: int
) -> tuple[list[int], list[int]] | None:
    """Tokenize a conversation into (ids, labels) of equal length.

    labels[i] is ids[i] where the model should learn to produce that token and
    IGNORE elsewhere. If the conversation exceeds ``max_len`` tokens it is cut
    at the last assistant turn that still fits; returns None if not even the
    first exchange fits or there is nothing to train on.
    """
    end = _ids(tok, END)
    ids: list[int] = []
    labels: list[int] = []
    kept_ids: list[int] = []
    kept_labels: list[int] = []

    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role not in ROLE_TOKENS:
            continue
        body = tok.encode(content)
        ids += [_ids(tok, ROLE_TOKENS[role])] + body + [end]
        if role == "assistant":
            labels += [IGNORE] + body + [end]  # train on the reply and the stop token
        else:
            labels += [IGNORE] * (len(body) + 2)
        if len(ids) > max_len:
            break
        if role == "assistant":
            kept_ids, kept_labels = list(ids), list(labels)

    if not kept_ids or all(l == IGNORE for l in kept_labels):
        return None
    return kept_ids, kept_labels


def render_prompt(tok: BPETokenizer, messages: list[dict]) -> list[int]:
    """Token ids for a conversation so far, ending with the assistant marker to prompt a reply."""
    end = _ids(tok, END)
    ids: list[int] = []
    for msg in messages:
        ids += [_ids(tok, ROLE_TOKENS[msg["role"]])] + tok.encode(msg["content"]) + [end]
    return ids + [_ids(tok, ASSISTANT)]
