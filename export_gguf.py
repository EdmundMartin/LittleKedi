"""Export a LittleKedi checkpoint straight to GGUF for llama.cpp / Ollama / LM Studio.

The model is written in llama.cpp's native ``deepseek2`` layout (its DeepSeek-V2/V3 architecture),
which LittleKedi matches: MLA attention with the absorbed k_b/v_b split, sigmoid-gated DeepSeekMoE
with shared experts, expert groups and the balancing bias. Compared with v1's exporter:

  * routed experts are already stacked; they're transposed from our [E, in, out] to GGUF's
    PyTorch-style [E, out, in] (``ffn_{gate,up,down}_exps``);
  * tied embeddings are written as both ``token_embd`` and ``output`` (deepseek2 expects an output tensor);
  * there's no MTP module to leave out.

    python export_gguf.py --ckpt checkpoints/tiny.pt                       # f16 -> checkpoints/tiny-f16.gguf
    python export_gguf.py --ckpt checkpoints/tiny-sft.pt --dtype q8_0      # chat model, 8-bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gguf
import numpy as np
import torch
from tokenizers import pre_tokenizers

from littlekedi import ModelConfig
from littlekedi.tokenizer import ASSISTANT, CHAT_TOKENS, END, EOT, SYSTEM, USER, BPETokenizer

# Jinja chat template matching littlekedi/chat.py:render_prompt
CHAT_TEMPLATE = (
    "{% for m in messages %}"
    "{% if m['role'] == 'system' %}" + SYSTEM + "{{ m['content'] }}" + END +
    "{% elif m['role'] == 'user' %}" + USER + "{{ m['content'] }}" + END +
    "{% elif m['role'] == 'assistant' %}" + ASSISTANT + "{{ m['content'] }}" + END +
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}" + ASSISTANT + "{% endif %}"
)


def tensor_map(state: dict[str, torch.Tensor], cfg: ModelConfig) -> dict[str, torch.Tensor]:
    """Our parameter names -> llama.cpp deepseek2 tensor names (PyTorch [out, in] layout)."""
    H, nope, v = cfg.n_heads, cfg.qk_nope_head_dim, cfg.v_head_dim
    out = {
        "token_embd.weight": state["embed.weight"],
        "output_norm.weight": state["norm.weight"],
        "output.weight": state["embed.weight"] if cfg.tie_embeddings else state["head.weight"],
    }
    for i in range(cfg.n_layers):
        p, b = f"layers.{i}.", f"blk.{i}."
        out[b + "attn_norm.weight"] = state[p + "attn_norm.weight"]
        out[b + "ffn_norm.weight"] = state[p + "ffn_norm.weight"]
        if cfg.q_lora_rank > 0:
            out[b + "attn_q_a.weight"] = state[p + "attn.wq_a.weight"]
            out[b + "attn_q_a_norm.weight"] = state[p + "attn.q_norm.weight"]
            out[b + "attn_q_b.weight"] = state[p + "attn.wq_b.weight"]
        else:
            out[b + "attn_q.weight"] = state[p + "attn.wq.weight"]
        out[b + "attn_kv_a_mqa.weight"] = state[p + "attn.wkv_a.weight"]
        out[b + "attn_kv_a_norm.weight"] = state[p + "attn.kv_norm.weight"]
        # Split the KV up-projection per head for MLA weight absorption; k_b is stored transposed.
        kv_b = state[p + "attn.wkv_b.weight"].view(H, nope + v, cfg.kv_lora_rank)
        k_b, v_b = kv_b.split([nope, v], dim=1)
        out[b + "attn_k_b.weight"] = k_b.transpose(1, 2).contiguous()  # [H, kv_lora_rank, nope]
        out[b + "attn_v_b.weight"] = v_b.contiguous()                  # [H, v, kv_lora_rank]
        out[b + "attn_output.weight"] = state[p + "attn.wo.weight"]

        if i < cfg.n_dense_layers:
            out[b + "ffn_gate.weight"] = state[p + "ffn.w1.weight"]
            out[b + "ffn_up.weight"] = state[p + "ffn.w3.weight"]
            out[b + "ffn_down.weight"] = state[p + "ffn.w2.weight"]
            continue
        out[b + "ffn_gate_inp.weight"] = state[p + "ffn.gate.weight"]
        out[b + "exp_probs_b.bias"] = state[p + "ffn.gate.bias"]
        # Stacked experts are [E, in, out] (x @ w); GGUF wants each expert as [out, in].
        for ours, theirs in (("w_gate", "gate"), ("w_up", "up"), ("w_down", "down")):
            out[b + f"ffn_{theirs}_exps.weight"] = state[p + f"ffn.experts.{ours}"].transpose(1, 2).contiguous()
        if cfg.n_shared_experts > 0:
            for ours, theirs in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
                out[b + f"ffn_{theirs}_shexp.weight"] = state[p + f"ffn.shared.{ours}.weight"]
    return out


def add_hparams(w: gguf.GGUFWriter, cfg: ModelConfig) -> None:
    w.add_block_count(cfg.n_layers)
    w.add_context_length(cfg.max_seq_len)
    w.add_embedding_length(cfg.dim)
    w.add_feed_forward_length(cfg.dense_hidden_dim)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(1)  # MLA is run as MQA over the shared latent
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_layer_norm_rms_eps(cfg.norm_eps)
    w.add_vocab_size(cfg.vocab_size)
    w.add_leading_dense_block_count(cfg.n_dense_layers)
    # Always written, even as 0: llama.cpp only treats the key as optional for layer counts it
    # recognises as "lite" models (26/27 layers); a 0 value selects the uncompressed attn_q path.
    w.add_q_lora_rank(cfg.q_lora_rank)
    w.add_kv_lora_rank(cfg.kv_lora_rank)
    w.add_key_length(cfg.kv_lora_rank + cfg.qk_rope_head_dim)
    w.add_value_length(cfg.kv_lora_rank)
    w.add_key_length_mla(cfg.qk_head_dim)
    w.add_value_length_mla(cfg.v_head_dim)
    w.add_rope_dimension_count(cfg.qk_rope_head_dim)
    w.add_expert_count(cfg.n_routed_experts)
    w.add_expert_used_count(cfg.n_activated_experts)
    w.add_expert_shared_count(cfg.n_shared_experts)
    w.add_expert_feed_forward_length(cfg.moe_hidden_dim)
    w.add_expert_group_count(cfg.n_expert_groups)
    w.add_expert_group_used_count(cfg.n_limited_groups)
    w.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
    w.add_expert_weights_norm(True)
    w.add_expert_weights_scale(cfg.route_scale)


def gpt2_pretokenizer_mismatches(tok: BPETokenizer, texts: list[str]) -> list[str]:
    """Texts that llama.cpp's "gpt-2" pre-tokenizer would tokenize differently from ours.

    v2 tokenizers split digits before the GPT-2 regex. llama.cpp has no exact equivalent, but BPE never
    learned merges across those digit boundaries, so applying our merges after the plain GPT-2 regex
    should give the same tokens; this checks that on real text.
    """
    plain = BPETokenizer.from_str(tok.to_str())
    plain.tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    return [t for t in texts if plain.encode(t) != tok.encode(t)]


def add_tokenizer(w: gguf.GGUFWriter, tok: BPETokenizer, vocab_size: int, chat: bool) -> None:
    vocab = tok.tok.get_vocab(with_added_tokens=True)
    by_id = {i: t for t, i in vocab.items()}
    specials = {EOT, *CHAT_TOKENS}
    tokens, types = [], []
    for i in range(vocab_size):
        t = by_id.get(i)
        if t is None:
            tokens.append(f"[PAD{i}]")
            types.append(gguf.TokenType.UNUSED)
        else:
            tokens.append(t)
            types.append(gguf.TokenType.CONTROL if t in specials else gguf.TokenType.NORMAL)
    merges = json.loads(tok.to_str())["model"]["merges"]
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]

    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")  # ByteLevel pre-tokenizer with the GPT-2 split regex
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges(merges)
    eot = vocab[EOT]
    w.add_bos_token_id(eot)
    # Base models were trained on EOT-separated documents, so EOT doubles as BOS; SFT conversations
    # were trained without it, so chat models don't prepend it.
    w.add_add_bos_token(not chat)
    if chat:
        end = vocab[END]
        w.add_eos_token_id(end)
        w.add_eot_token_id(end)
        w.add_chat_template(CHAT_TEMPLATE)
    else:
        w.add_eos_token_id(eot)


def to_numpy(name: str, t: torch.Tensor, dtype: str) -> tuple[np.ndarray, gguf.GGMLQuantizationType | None]:
    a = t.detach().float().cpu().numpy()
    # Norms, biases and the router stay in f32, as llama.cpp's converter does.
    if a.ndim == 1 or name.endswith("ffn_gate_inp.weight") or dtype == "f32":
        return a.astype(np.float32), None
    if dtype == "q8_0" and a.shape[-1] % gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0][0] == 0:
        return gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0), gguf.GGMLQuantizationType.Q8_0
    return a.astype(np.float16), None  # f16, or q8_0 fallback for rows that don't fit 32-wide blocks


SAMPLE_TEXT = ["In 1969, 3 astronauts flew 384,400 km.", "x=12.5; y = x*2048 - 7", "Call 555-0123 at 9:30pm!",
               "The 2nd and 3rd of 10,000 items cost $4.99 each.", "<|user|>What's 7 + 6?<|end|>"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="checkpoints/tiny.pt")
    ap.add_argument("--out", default=None, help="default: <ckpt>-<dtype>.gguf")
    ap.add_argument("--dtype", choices=["f32", "f16", "q8_0"], default="f16")
    ap.add_argument("--name", default=None, help="model name stored in the file")
    # Recommended sampling, stored as llama.cpp's general.sampling.* defaults (command-line flags still override).
    # Small models loop without a repetition penalty; these match the model card and Ollama params.
    ap.add_argument("--repeat-penalty", type=float, default=1.15)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ModelConfig.from_dict(ckpt["config"])
    tok = BPETokenizer.from_str(ckpt["tokenizer"])
    if bad := gpt2_pretokenizer_mismatches(tok, SAMPLE_TEXT):
        print(f"WARNING: llama.cpp's gpt-2 pre-tokenizer would tokenize these differently: {bad}")
    chat = bool(ckpt.get("chat"))
    out = Path(args.out or Path(args.ckpt).with_name(f"{Path(args.ckpt).stem}-{args.dtype}.gguf"))
    name = args.name or Path(args.ckpt).stem

    w = gguf.GGUFWriter(str(out), gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.DEEPSEEK2])
    w.add_name(name)
    w.add_file_type({"f32": gguf.LlamaFileType.ALL_F32, "f16": gguf.LlamaFileType.MOSTLY_F16,
                     "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0}[args.dtype])
    add_hparams(w, cfg)
    w.add_sampling_penalty_repeat(args.repeat_penalty)
    w.add_sampling_temp(args.temp)
    w.add_sampling_top_p(args.top_p)
    add_tokenizer(w, tok, cfg.vocab_size, chat)
    for tname, t in tensor_map(ckpt["model"], cfg).items():
        data, qtype = to_numpy(tname, t, args.dtype)
        w.add_tensor(tname, data, raw_dtype=qtype)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f}MB): {name}, step {ckpt.get('step')}, "
          f"{'chat' if chat else 'base'} model, {args.dtype}, vocab {cfg.vocab_size}")


if __name__ == "__main__":
    main()
