"""The LittleKedi model card: one template for every size and sparsity variant.

The family, size and sparsity sections are shared text and read the same on every card (the
current model's row is highlighted). Everything model-specific is filled in from the checkpoint,
its training log (<ckpt>.log.jsonl), the dataset's meta.json and the eval logs (evals.jsonl,
mmlu_results.jsonl), so re-run prepare_release.py after evaluating. benchmark.py's factual probes are a
quick check only and deliberately stay off the card.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from littlekedi import ModelConfig
from littlekedi.config import PRESETS, SIZES, VARIANTS

FAMILY = "LittleKedi"
LOGO = "little_kedi.jpg"  # repo root; prepare_release.py copies it next to the card


def logo_html(src: str | None) -> str:
    return f'<p align="center"><img src="{src}" alt="{FAMILY} logo" width="640"></p>\n\n' if src else ""

# ---------------------------------------------------------------- shared across every card

VARIANT_NAMES = {"": "standard", "-sparse": "sparse", "-wide-sparse": "wide-sparse"}

FAMILY_INTRO = f"""{FAMILY} is a family of small DeepSeek-V3-style Mixture-of-Experts language models trained from
scratch. Every size comes in three variants that differ in **how sparse** they are: how many routed experts
they have, and what share of their parameters a token actually passes through.

| Variant | Routed experts |
|---|---|
| **standard** (`<size>`) | a few wide experts; the densest variant |
| **sparse** (`<size>-sparse`) | many narrower, fine-grained experts, more of them active per token |
| **wide-sparse** (`<size>-wide-sparse`) | more experts again; the largest total and smallest active share |

More experts means more total capacity to store knowledge, but each expert
sees fewer tokens (tokens × top-k ÷ experts per layer), so sparser variants need more training data before
their experts are well trained. The variants of a size share a tokenizer and dataset, so they can be
compared directly. Shapes change as the family scales; the table below is generated from the current
presets."""

FAMILY_FOOTNOTE = """"Active" counts everything a token passes through: embeddings, attention, the dense layer, shared
experts and its top-k routed experts. "Active non-embedding" leaves out the embedding table (tied with the
output head), which costs memory but almost no compute; it's the fair number for comparing compute
between models."""

DISCLAIMER_BASE = ("This is the **base model**, before any instruction tuning. It continues text; it doesn't follow "
                   "instructions. It's an educational and research artifact: it writes fluent, on-topic English but "
                   "**knows few facts and confidently makes things up**. Don't use it for anything that matters.")
DISCLAIMER_INSTRUCT = ("This is the **instruction-tuned (chat) model**. It answers in a chat format and stops when it's "
                       "done, but it's still a very small model: answers are fluent and on-topic but often **wrong or "
                       "vague**, and it **invents facts confidently**. It's an educational and research artifact; don't "
                       "use it for anything that matters.")
CHAT_FORMAT = """## Chat format

```
<|system|>You are a helpful assistant.<|end|><|user|>What causes the seasons?<|end|><|assistant|>
```

The model writes its reply and ends it with `<|end|>`. The system turn is optional. The GGUF files embed this
template, and `chat_template.jinja` / `tokenizer_config.json` carry it for other tools."""

# Datasets prepare_data.py can mix, as (HF repo, card name, what it is, license).
DATASETS = {
    "fineweb-edu": ("HuggingFaceFW/fineweb-edu", "FineWeb-Edu `sample/10BT`", "educational web pages", "ODC-BY 1.0"),
    "fineweb-edu-100bt": ("HuggingFaceFW/fineweb-edu", "FineWeb-Edu `sample/100BT`", "educational web pages", "ODC-BY 1.0"),
    "cosmopedia": ("HuggingFaceTB/cosmopedia", "Cosmopedia", "synthetic textbooks", "Apache 2.0"),
    "cosmo-stanford": ("HuggingFaceTB/cosmopedia", "Cosmopedia (Stanford)", "synthetic textbooks", "Apache 2.0"),
    "cosmo-openstax": ("HuggingFaceTB/cosmopedia", "Cosmopedia (OpenStax)", "synthetic textbooks", "Apache 2.0"),
    "cosmo-wikihow": ("HuggingFaceTB/cosmopedia", "Cosmopedia (WikiHow)", "synthetic how-to articles", "Apache 2.0"),
    "cosmo-khanacademy": ("HuggingFaceTB/cosmopedia", "Cosmopedia (Khan Academy)", "synthetic lessons", "Apache 2.0"),
    "finemath": ("HuggingFaceTB/finemath", "FineMath 4+", "maths web pages", "ODC-BY 1.0"),
    "smoltalk": ("HuggingFaceTB/smoltalk", "SmolTalk", "chat conversations",
                 "Apache 2.0 for its new subsets, with the datasets it includes under their own licenses"),
    "code": ("codeparrot/github-code-clean", "github-code-clean", "source code",
             "Apache 2.0, with each file under its own open-source license"),
}

# Published small-model numbers for the comparison table, as
# (card link, total params, non-embedding params, training tokens, {metric: value}).
# Pythia-70M: EleutherAI's evaluations as quoted on the Wisp-15M card (no HellaSwag given there).
# GPT-2 and Supra-50M: the Supra-50M-Base card's comparison table.
BASELINES = [
    ("[Pythia-70M](https://huggingface.co/EleutherAI/pythia-70m)", 70.4e6, 18.9e6, "300B (the Pile)",
     {"sciq": 0.552, "arc_easy": 0.350}),
    ("[GPT-2](https://huggingface.co/openai-community/gpt2)", 124.4e6, 85.1e6, "undisclosed (WebText)",
     {"sciq": 0.532, "arc_easy": 0.420, "hellaswag": 0.295}),
    ("[Supra-50M](https://huggingface.co/SupraLabs/Supra-50M-Base)", 51.8e6, 35.4e6, "20B (FineWeb-Edu)",
     {"sciq": 0.772, "arc_easy": 0.522, "hellaswag": 0.318}),
]
EVAL_TASKS = [("sciq", "SciQ"), ("arc_easy", "ARC-Easy"), ("hellaswag", "HellaSwag")]


# ---------------------------------------------------------------- helpers

def infer_preset(cfg: ModelConfig) -> str | None:
    """The preset a config was built from (checkpoints from before presets were recorded)."""
    d = cfg.to_dict()
    for name, preset in PRESETS.items():
        if all(d.get(k) == v for k, v in preset.items() if k not in ("vocab_size", "max_seq_len")):
            return name
    return None


def size_and_variant(preset: str) -> tuple[str, str]:
    for v in ("-wide-sparse", "-sparse"):
        if preset.endswith(v):
            return preset[: -len(v)], v
    return preset, ""


def model_name(preset: str, chat: bool = False) -> str:
    """LittleKedi-<preset>-base for pretrained models, -instruct for SFT (chat) models."""
    return f"{FAMILY}-{preset}-{'instruct' if chat else 'base'}"


def same_path(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and Path(a).resolve() == Path(b).resolve()


def read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def merged_results(path: str, ckpt: str, step) -> dict:
    """Logged results for one checkpoint at one step, merged across lines (later lines win)."""
    out: dict = {}
    for r in read_jsonl(path):
        if same_path(r.get("ckpt"), ckpt) and r.get("step") == step:
            out.update(r)
    return out


def read_run_log(path: str | Path, up_to_step: int | None = None) -> dict:
    """train.py's JSONL log, merged across resumes: first start's args, train/eval records by step.

    up_to_step drops records after a checkpoint's step (a milestone checkpoint shares the run's log).
    """
    starts, train, evals = [], {}, {}
    for r in read_jsonl(path):
        if r["event"] != "start" and up_to_step is not None and r.get("step", 0) > up_to_step:
            continue
        if r["event"] == "start":
            starts.append(r)
        elif r["event"] == "train":
            train[r["step"]] = r
        elif r["event"] == "eval":
            evals[r["step"]] = r
    return {"start": starts[0] if starts else None, "latest_start": starts[-1] if starts else None,
            "resumes": max(0, len(starts) - 1),
            "train": [train[s] for s in sorted(train)], "eval": [evals[s] for s in sorted(evals)]}


def sci(x: float, digits: int = 3) -> str:
    mantissa, exp = f"{x:.{digits - 1}e}".split("e")
    return f"{float(mantissa):g}e{int(exp)}"


def fmt_tokens(n: float) -> str:
    return f"{n / 1e6:,.2f}M" if n >= 1e6 else f"{n / 1e3:,.1f}K"


def pct(x) -> str:
    return f"{x:.1%}" if x is not None else "–"


def median(xs: list[float]) -> float | None:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def bio_med_acc(m: dict) -> tuple[float, int] | None:
    from mmlu_eval import BIO_MED
    counts = m.get("subject_counts") or {}
    subs = [s for s in BIO_MED if s in m["by_subject"] and s in counts]
    n = sum(counts[s] for s in subs)
    return (sum(m["by_subject"][s] * counts[s] for s in subs) / n, n) if n else None


# ---------------------------------------------------------------- sections

def family_table(current: str) -> str:
    rows = []
    for size in SIZES:
        for v in VARIANTS:
            name = size + v
            cfg = ModelConfig.from_preset(name)
            e = cfg.param_estimate()
            cells = [f"`{name}`", VARIANT_NAMES[v], f"{cfg.vocab_size // 1024}K",
                     f"{e['total'] / 1e6:.1f}M", f"{e['activated'] / 1e6:.1f}M",
                     f"{e['activated_non_embedding'] / 1e6:.1f}M", f"{e['fraction_activated']:.0%}",
                     f"{cfg.n_routed_experts} × {cfg.moe_hidden_dim}, top-{cfg.n_activated_experts}"]
            if name == current:
                cells = [f"**{c}**" for c in cells]
                cells[0] = f"**`{name}` ← this model**"
            rows.append("| " + " | ".join(cells) + " |")
    return ("| Model | Variant | Vocab | Total | Active | Active non-emb. | Active % | Routed experts |\n"
            "|---|---|---|---|---|---|---|---|\n" + "\n".join(rows))


def siblings_line(preset: str) -> str:
    size, variant = size_and_variant(preset)
    others = [f"`{size + v}`" for v in VARIANTS if v != variant]
    me = ModelConfig.from_preset(preset).param_estimate()
    return (f"This card is for **`{preset}`**, the **{VARIANT_NAMES[variant]}** {size} model: "
            f"{me['activated'] / 1e6:.1f}M of its {me['total'] / 1e6:.1f}M parameters "
            f"({me['fraction_activated']:.0%}) are active per token. Its siblings are {' and '.join(others)}.")


def architecture(cfg: ModelConfig, counts: dict, load_note: str | None) -> str:
    moe_layers = f"layers 1–{cfg.n_layers - 1}" if cfg.n_dense_layers == 1 else f"layers {cfg.n_dense_layers}–{cfg.n_layers - 1}"
    return f"""## Architecture

| Component | This model |
|---|---|
| Parameters | {counts['total'] / 1e6:.2f}M total, {counts['activated'] / 1e6:.2f}M active per token ({counts['fraction_activated']:.0%}), {counts['activated_non_embedding'] / 1e6:.2f}M active non-embedding |
| Layers / hidden size | {cfg.n_layers} / {cfg.dim} ({'layer 0 has' if cfg.n_dense_layers == 1 else f'the first {cfg.n_dense_layers} layers have'} a dense SwiGLU FFN of {cfg.dense_hidden_dim}; {moe_layers} are MoE) |
| Attention | **Multi-head Latent Attention (MLA)**: {cfg.n_heads} heads, KV latent {cfg.kv_lora_rank}{f', query latent {cfg.q_lora_rank}' if cfg.q_lora_rank else ''}, decoupled RoPE dim {cfg.qk_rope_head_dim}, head dims {cfg.qk_nope_head_dim}+{cfg.qk_rope_head_dim} (q/k) and {cfg.v_head_dim} (v). The KV cache stores only the {cfg.kv_lora_rank}-dim latent plus a {cfg.qk_rope_head_dim}-dim RoPE key per token |
| MoE | **DeepSeekMoE**: {cfg.n_routed_experts} routed experts (SwiGLU, {cfg.moe_hidden_dim} hidden), top-{cfg.n_activated_experts}, plus {cfg.n_shared_experts} shared expert{'s' if cfg.n_shared_experts > 1 else ''}; sigmoid gating{f'; group-limited routing ({cfg.n_expert_groups} groups, top-{cfg.n_limited_groups})' if cfg.n_expert_groups > 1 else ''} |
| Load balancing | **Auxiliary-loss-free** bias balancing (update speed {cfg.bias_update_speed}) plus a small sequence-wise loss (α = {cfg.seq_aux_loss_alpha}).{f' {load_note}' if load_note else ''} |
| Expert dispatch | Training: {'fixed capacity (capacity factor ' + str(cfg.capacity_factor) + '; overflow assignments are dropped)' if cfg.moe_backend == 'capacity' else 'grouped GEMM, dropless' if cfg.moe_backend == 'grouped_mm' else 'per-expert loop, dropless'}. Inference is always dropless |
| Embeddings | {'Input embedding and output head **tied**' if cfg.tie_embeddings else 'Separate input embedding and output head'} |
| Context | {cfg.max_seq_len:,} tokens |"""


def training(counts: dict, log: dict, meta: dict, tok_info: dict, tokens: int, step, hardware: str) -> str:
    a = (log["start"] or {}).get("args", {})
    latest = (log["latest_start"] or {}).get("args", a)
    tps = a.get("batch_size", 0) * (a.get("seq_len") or 0) * a.get("grad_accum", 1)
    mix = meta.get("mix_tokens") or {}
    rate = median([r["tok_s"] for r in log["train"] if r.get("tok_s")])
    hw = hardware + (f": about {tokens / rate / 3600:.1f} hours at a median ~{rate / 1e3:.1f}K tokens/s" if rate else "")
    if a.get("optimizer") == "muon":
        opt = (f"**Muon** (momentum 0.95, Nesterov, 5 Newton-Schulz steps; each expert's matrix orthogonalized separately) "
               f"for hidden matrices, AdamW (β = 0.9, 0.95) for embeddings, norms and the router; "
               f"weight decay {a.get('weight_decay')}, gradient clipping {a.get('grad_clip')}")
    else:
        opt = f"AdamW (β = 0.9, 0.95), weight decay {a.get('weight_decay')}, gradient clipping {a.get('grad_clip')}"
    if a.get("schedule") == "wsd":
        sched = (f"{sci(a.get('lr', 0))} peak; **WSD**: {a.get('warmup')} warm-up steps, constant, then linear decay to "
                 f"{a.get('min_lr', 0):g} over the last {a.get('decay_frac', 0.2):.0%} of steps")
    else:
        sched = f"{sci(a.get('lr', 0))} peak, {a.get('warmup')} warm-up steps, cosine decay to {sci(a.get('min_lr', 0))}"
    mix_md = ""
    if mix:
        total = sum(mix.values())
        mix_md = "\n| Source | What it is | Tokens | Share |\n|---|---|---|---|\n" + "\n".join(
            f"| {DATASETS.get(k, (k, k, '', ''))[1]} | {DATASETS.get(k, (k, k, '', ''))[2]} | {v / 1e6:,.0f}M | {v / total:.0%} |"
            for k, v in mix.items()) + "\n"
    full_pass = not meta.get("train_tokens") or tokens >= 0.99 * meta["train_tokens"]
    n_docs = meta.get("train_documents", meta.get("documents"))
    docs = f", {n_docs:,} documents" if n_docs and full_pass else ""
    tok_row = (f"Byte-level BPE, {tok_info['vocab']:,} tokens (~{meta.get('bytes_per_token', 0):.2f} bytes/token on this mix)"
               + ("; digits split individually" if tok_info["digits"] else "; digits not split (v1 tokenizer)")
               + ("; chat special tokens built in" if tok_info["chat"] else "; no chat tokens, so chat data is plain `User:` / `Assistant:` text"))
    return f"""## Training

| | |
|---|---|
| Data | {tokens / 1e6:,.0f}M tokens{docs} (~{tokens / counts['total']:.0f} tokens per parameter, ~{tokens / counts['activated']:.0f} per active parameter), {'one pass' if full_pass else f"the first {tokens / meta['train_tokens']:.0%} of a {meta['train_tokens'] / 1e6:,.0f}M-token dataset"}{', sources interleaved throughout (mix below)' if mix else ''} |
| Tokenizer | {tok_row} |
| Steps | {(step + 1) if isinstance(step, int) else '?':,} × {tps:,} tokens (batch {a.get('batch_size')} × {a.get('seq_len')}{f' × {a.get("grad_accum")} accumulation' if a.get('grad_accum', 1) > 1 else ''}) |
| Optimizer | {opt} |
| Learning rate | {sched}{f' (schedule set for {latest.get("steps"):,} steps)' if latest.get('steps') else ''} |
| Precision | {a.get('precision', '?')} autocast{', `torch.compile`' if a.get('compile') else ''} |
| Compute | ~{sci(6 * counts['activated_non_embedding'] * tokens, 2)} FLOPs (6 × active non-embedding parameters × tokens) |
| Hardware | {hw} |
{mix_md}"""


def evaluation(ckpt: str, step, log: dict, val_bpt: float | None, evals: dict, mmlu: dict) -> str:
    parts = ["## Evaluation"]
    ev = log["eval"]
    final = ev[-1] if ev else {}
    rows = []
    if "val_loss" in final:
        v = final["val_loss"]
        rows += [f"| Validation loss (nats/token, held-out mix) | **{v:.3f}** |", f"| Perplexity | {math.exp(v):.1f} |"]
        if val_bpt:
            rows.append(f"| Bits per byte | {v / val_bpt / math.log(2):.3f} |")
    for k, v in final.items():
        if k.startswith("val_") and k.endswith("_loss") and k != "val_loss":
            rows.append(f"| Validation loss: {k[4:-5]} | {v:.3f} |")
    if "val_dead_experts" in final:
        rows.append(f"| Experts never chosen on the validation sample | {final['val_dead_experts']:.0f} |")
    if rows:
        parts.append("| Metric | Value |\n|---|---|\n" + "\n".join(rows))
    if len(ev) > 1:
        idx = sorted({round(i * (len(ev) - 1) / 7) for i in range(8)})
        parts.append("Validation loss over training: " + " → ".join(
            f"{ev[i]['val_loss']:.2f} ({ev[i]['tokens'] / 1e6:,.0f}M tok)" for i in idx))

    if evals:
        rows = "\n".join(f"| {name} | {evals[t]['acc_norm']:.1%} ± {evals[t]['stderr']:.1%} | {evals[t]['acc']:.1%} | "
                         f"{evals[t]['n']:,} | 25% |" for t, name in EVAL_TASKS if t in evals)
        parts.append(f"""### Zero-shot benchmarks (`evals.py`)

| Task | acc_norm | acc | Items | Chance |
|---|---|---|---|---|
{rows}

Each answer choice is scored by the log-probability of its text after the prompt, using lm-evaluation-harness
prompts (SciQ includes its supporting passage). **acc_norm** divides by the answer's length in characters;
**acc** doesn't. Splits: SciQ test, ARC-Easy test, HellaSwag validation.""")

    if mmlu:
        fmts = [f for f in ("letter", "cloze") if f in mmlu]
        rows = "\n".join(f"| {f} | {mmlu[f]['accuracy']:.1%} | {mmlu[f]['scored']:,} (skipped {mmlu[f]['skipped']}) | "
                         f"{mmlu[f]['avg_shots']:.1f} |" for f in fmts)
        first = mmlu[fmts[0]]["by_category"]
        cat_rows = "\n".join(f"| {c} | {first[c]['subjects']} | {first[c]['questions']:,} | "
                             + " | ".join(pct(mmlu[f]["by_category"][c]["accuracy"]) for f in fmts) + " |" for c in first)
        bio = ""
        if "cloze" in mmlu and (b := bio_med_acc(mmlu["cloze"])):
            acc, n = b
            z = (acc - 0.25) / math.sqrt(0.25 * 0.75 / n)
            bio = (f"\n**Biology & medicine (cloze): {acc:.1%} on {n:,} questions**, {(acc - 0.25) * 100:+.1f} points "
                   f"against 25% chance (about {z:.1f} standard errors{', significant' if z >= 2 else ', not yet significant'}). "
                   "This pools the 10 life-science and health subjects; it isn't an official MMLU category.\n")
        parts.append(f"""### MMLU (`mmlu_eval.py`)

| Format | Accuracy | Questions scored | Average few-shot examples |
|---|---|---|---|
{rows}

| Category | Subjects | Questions | {" | ".join(f.capitalize() for f in fmts)} |
|---|---|---|{"---|" * len(fmts)}
{cat_rows}
{bio}
In the **letter** format the model compares " A"–" D", with same-subject examples added while they fit the
context. In the **cloze** format each answer's text is scored by log-probability per byte; at this size the
cloze results are the meaningful ones. Chance is 25%.""")

    if len(parts) == 1:
        parts.append("*(no evaluations logged for this checkpoint yet)*")
    return "\n\n".join(parts)


def comparison(columns: list[tuple]) -> str:
    """columns: (label, total, active non-emb, tokens label, evals, mmlu cloze, bio-med, val loss)."""
    def norm(e: dict, t: str):
        v = e.get(t)
        return v["acc_norm"] if isinstance(v, dict) else v

    rows = [("Parameters (total)", [f"{c[1] / 1e6:.1f}M" for c in columns]),
            ("Active non-embedding parameters", [f"{c[2] / 1e6:.1f}M" for c in columns]),
            ("Training tokens", [c[3] for c in columns])]
    rows += [(f"{name} (acc_norm)", [pct(norm(c[4], t)) for c in columns]) for t, name in EVAL_TASKS]
    rows += [("MMLU (cloze)", [pct(c[5]) for c in columns]),
             ("MMLU biology & medicine (cloze)", [pct(c[6]) for c in columns]),
             ("Validation loss (held-out mix)", [f"{c[7]:.3f}" if c[7] is not None else "–" for c in columns])]
    table = "\n".join(f"| {n} | " + " | ".join(v) + " |" for n, v in rows)
    return f"""### Compared with its siblings and other small models

| | {" | ".join(c[0] for c in columns)} |
|---|{"---|" * len(columns)}
{table}

{FAMILY} columns are measured with this repo's `evals.py` and `mmlu_eval.py` on the checkpoints named. Validation
losses are only comparable between models that share a tokenizer and validation set. The other columns are
**published figures**: Pythia-70M from EleutherAI's evaluations as quoted on the
[Wisp-15M card](https://huggingface.co/DedeProGames/Wisp-15M), GPT-2 and Supra-50M from the
[Supra-50M-Base card](https://huggingface.co/SupraLabs/Supra-50M-Base). Their prompts may differ from ours,
so treat gaps of a few points as noise."""


def usage(name: str, code_url: str | None, chat: bool, gguf: list[str], repo: str | None) -> str:
    code = f"[the training code]({code_url})" if code_url else "the training code"
    q8 = next((g for g in gguf if "q8_0" in g), gguf[0] if gguf else None)
    parts = ["## Usage"]
    if chat and q8:
        parts.append(f"""**llama.cpp** (GGUF, llama.cpp's built-in `deepseek2` architecture; the chat template is embedded):

```sh
llama-cli -m {q8} -cnv --temp 0.7 --top-p 0.9 --repeat-penalty 1.15
```

**Ollama** (uses the bundled `template` and `params` files): `ollama run hf.co/{repo or '<user>/' + name}:Q8_0`""")
    if chat:
        parts.append(f"""**PyTorch** (needs the `littlekedi` package from {code}):

```python
import json, torch
from safetensors.torch import load_file
from littlekedi import LittleKediModel, ModelConfig
from littlekedi.chat import END, render_prompt
from littlekedi.tokenizer import BPETokenizer

model = LittleKediModel(ModelConfig.from_dict(json.load(open("config.json")))).eval()
model.load_state_dict(load_file("model.safetensors"))
tok = BPETokenizer.from_file("tokenizer.json")
ids = torch.tensor([render_prompt(tok, [{{"role": "user", "content": "What causes the seasons?"}}])])
out = model.generate(ids, 200, temperature=0.7, top_p=0.9, repetition_penalty=1.15, stop_ids={{tok.token_to_id(END)}})
print(tok.decode(out[0, ids.shape[1]:].tolist()).replace(END, "").strip())
```

Or in a terminal: `python chat.py --ckpt <checkpoint>.pt`.""")
    else:
        parts.append(f"""**PyTorch** (needs the `littlekedi` package from {code}):

```python
import json, torch
from safetensors.torch import load_file
from littlekedi import LittleKediModel, ModelConfig
from littlekedi.tokenizer import BPETokenizer

model = LittleKediModel(ModelConfig.from_dict(json.load(open("config.json")))).eval()
model.load_state_dict(load_file("model.safetensors"))
tok = BPETokenizer.from_file("tokenizer.json")
ids = torch.tensor([[tok.eot_id] + tok.encode("Photosynthesis is")])
print(tok.decode(model.generate(ids, 60, temperature=0.7, top_p=0.9, repetition_penalty=1.15)[0].tolist()))
```

Or interactively: `python sample.py --ckpt <checkpoint>.pt`. Base models are released as PyTorch weights only;
GGUF builds come with the instruct models.""")
    return "\n\n".join(parts)


def sft_section(sft: dict, base_link: str) -> str:
    rows = [f"| Base model | {base_link} |"]
    if "conversations" in sft:
        rows.append(f"| Conversations | {sft['conversations']:,} ({sft['train']:,} train / {sft['val']:,} validation), "
                    f"{fmt_tokens(sft['assistant_tokens'])} assistant tokens trained on |")
    if sft.get("data_note"):
        rows.append(f"| Data | {sft['data_note']} |")
    elif sft.get("data"):
        rows.append("| Data | " + ", ".join(f"`{Path(d).name}`" if ":" not in d[2:] else f"`{d.split(':')[0]}`"
                                            for d in sft["data"]) + " |")
    rows.append("| Loss | On assistant replies and their closing `<|end|>` only; prompts and padding are masked |")
    if "steps" in sft:
        rows.append(f"| Schedule | {sft['steps']:,} steps, batch {sft['batch_size']}, {sft['epochs']:g} epochs, "
                    f"{'Muon + AdamW' if sft.get('optimizer') == 'muon' else 'AdamW'}, peak LR {sci(sft['lr'])}, cosine decay |")
    rows.append("| MoE routing | Load-balancing biases " + ("**kept updating** during SFT" if sft.get("update_moe_bias")
                else "**frozen** at their pretrained values") + ", with padding excluded from expert-load statistics; "
                "experts run dropless |")
    loss = ""
    if "val_before" in sft:
        loss = (f"\n\nValidation loss on {sft.get('val', 0):,} held-out conversations (assistant tokens only): "
                f"**{sft['val_before']:.3f} → {sft['val_after']:.3f}**.")
    return "## Fine-tuning (SFT)\n\n| | |\n|---|---|\n" + "\n".join(rows) + loss


def base_vs_instruct(base: dict, inst: dict) -> str:
    """base / inst: {"evals", "mmlu"} results for the base and instruct checkpoints."""
    def get(r, path):
        for k in path:
            r = r.get(k) if isinstance(r, dict) else None
        return r

    def bio(r):
        c = r.get("mmlu", {}).get("cloze")
        b = bio_med_acc(c) if c else None
        return b[0] if b else None


    metrics = [(f"{n} (acc_norm)", lambda r, t=t: get(r, ["evals", t, "acc_norm"]), pct) for t, n in EVAL_TASKS]
    metrics += [("MMLU (letter)", lambda r: get(r, ["mmlu", "letter", "accuracy"]), pct),
                ("MMLU (cloze)", lambda r: get(r, ["mmlu", "cloze", "accuracy"]), pct),
                ("MMLU biology & medicine (cloze)", bio, pct)]
    rows = [f"| {label} | {fmt(f(base))} | {fmt(f(inst))} |" for label, f, fmt in metrics
            if f(base) is not None or f(inst) is not None]
    if not rows:
        return "### Base vs instruct\n\n*(no evaluations logged for these checkpoints yet)*"
    return ("### Base vs instruct\n\nBoth are scored on the same raw-text prompts, without the chat template, so the "
            "numbers are directly comparable. SFT mainly teaches format, turn-taking and when to stop, so large "
            "changes in knowledge benchmarks aren't expected. Chance on the multiple-choice tasks is 25%.\n\n"
            "| Metric | Base | **Instruct** |\n|---|---|---|\n" + "\n".join(rows))


def in_progress_note(p: dict | None) -> str:
    """A short notice for a checkpoint from a run that hasn't finished."""
    if not p:
        return ""
    return (f"\n> **Training in progress.** This is a checkpoint at **{p['tokens'] / 1e9:.2f}B of {p['planned'] / 1e9:.1f}B "
            f"tokens** ({p['tokens'] / p['planned']:.0%}). The weights and results below will be replaced as training "
            "continues, and the final model will be better than this checkpoint.\n")


def build_card(*, preset: str, cfg: ModelConfig, counts: dict, step, tokens: int, log: dict, meta: dict,
               tok_info: dict, val_bpt: float | None, evals: dict, mmlu: dict, columns: list[tuple],
               files: list[tuple[str, str]], hardware: str, capabilities: str | None, code_url: str | None,
               ckpt: str, logo: str | None = LOGO, chat: bool = False, sft: dict | None = None,
               base_results: dict | None = None, inst_results: dict | None = None, base_repo: str | None = None,
               gguf: list[str] | None = None, repo: str | None = None, in_progress: dict | None = None) -> str:
    name = model_name(preset, chat)
    base_name = model_name(preset)
    base_link = f"[{base_name}](https://huggingface.co/{base_repo})" if base_repo else f"`{base_name}`"
    empty = {"evals": {}, "mmlu": {}}
    front_base = f"base_model: {base_repo}\nbase_model_relation: finetune\n" if chat and base_repo else ""
    kind_tags = "- chat\n- instruction-tuned" if chat else "- base"
    chat_format = f"\n{CHAT_FORMAT}\n" if chat else ""
    sft_md = sft_section(sft or {}, base_link) + "\n\n" if chat else ""
    # An instruct card's pretraining validation curve belongs to the base model; its SFT loss is in the SFT table.
    eval_md = evaluation(ckpt, step, {**log, "eval": []} if chat else log, val_bpt, evals, mmlu)
    bvi_md = "\n" + base_vs_instruct(base_results or empty, inst_results or empty) + "\n" if chat else ""
    intro = f"the **instruction-tuned (chat) version of {base_link}**, " if chat else ""
    chat_license = "\n- Chat data: see the SFT table above for its sources, each under its own license." if chat else ""
    mix = meta.get("mix_tokens") or {}
    repos = list(dict.fromkeys(DATASETS[k][0] for k in mix if k in DATASETS))
    licenses, seen = [], set()
    for k in mix:
        if k in DATASETS and DATASETS[k][0] not in seen:
            seen.add(DATASETS[k][0])
            licenses.append(f"- Pretraining data: **{DATASETS[k][1].split(' (')[0].split(' `')[0]}** ({DATASETS[k][3]}).")
    train = log["train"]
    late = train[len(train) // 2:]
    load_note = None
    if late and "mean_load_ratio" in late[0]:
        load_note = (f"Over the second half of training the median batch put {median([r['mean_load_ratio'] for r in late]):.2f}× "
                     f"the mean load on the busiest expert ({median([r['max_load_ratio'] for r in late]):.2f}× in the worst layer)"
                     + (f", and {median([r['drop_rate'] for r in late]):.2%} of expert assignments were dropped for capacity"
                        if "drop_rate" in late[0] else "") + ".")
    data_desc = (f"a {tokens / 1e6:,.0f}M-token mix ({', '.join(DATASETS.get(k, (k, k))[1].split(' `')[0] for k in mix)})"
                 if mix else f"{tokens / 1e6:,.0f}M tokens")
    return f"""---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
datasets:
{chr(10).join(f"- {r}" for r in repos) or "- HuggingFaceFW/fineweb-edu"}
{front_base}tags:
- mixture-of-experts
- moe
- deepseek
- multi-head-latent-attention
- mla
- from-scratch
- tiny
- littlekedi
{kind_tags}
---

{logo_html(logo)}# {name}
{in_progress_note(in_progress)}
`{name}` is {intro}a **small DeepSeek-V3-style Mixture-of-Experts language model trained from scratch**. It has
**{counts['total'] / 1e6:.1f}M parameters** in total, of which **{counts['activated'] / 1e6:.1f}M are active per token**
({counts['activated_non_embedding'] / 1e6:.2f}M excluding the embedding table). It was pretrained on {data_desc}.

*Part of the {FAMILY} family. Not affiliated with DeepSeek.*

{DISCLAIMER_INSTRUCT if chat else DISCLAIMER_BASE}
{chat_format}
## The {FAMILY} family

{FAMILY_INTRO}

{siblings_line(preset)}

{family_table(preset)}

{FAMILY_FOOTNOTE}

{architecture(cfg, counts, load_note)}

{training(counts, log, meta, tok_info, tokens, step, hardware)}
{sft_md}{eval_md}
{bvi_md}
{comparison(columns)}

## What it can and can't do

{capabilities or ("*To be written when training completes.*" if in_progress else "*TODO: describe what this model can and can't do (pass `--capabilities FILE`).*")}

{usage(name, code_url, chat, gguf or [], repo)}

## Files

| File | Contents |
|---|---|
{chr(10).join(f"| `{n}` | {d} |" for n, d in files)}

## Acknowledgements

- The architecture follows **DeepSeek-V3** (DeepSeek-AI, 2024): MLA, DeepSeekMoE and auxiliary-loss-free balancing.
- **Muon** (Keller Jordan et al., 2024) with Moonlight's update scaling (Liu et al., 2025).
{chr(10).join(licenses)}{chat_license}

*Card generated {time.strftime('%Y-%m-%d')} by `prepare_release.py` from `{Path(ckpt).name}` (step {step}).*
"""
