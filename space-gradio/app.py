"""LittleKedi's Reading Nook: chat with the LittleKedi instruct models.

LittleKedi is a family of very small DeepSeek-style mixture-of-experts language models. The models run on the CPU
with the `littlekedi` package (a copy of the training code's model package), loaded from their Hugging Face repos.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import torch
from huggingface_hub import hf_hub_download, repo_exists
from safetensors.torch import load_file

from littlekedi import LittleKediModel, ModelConfig
from littlekedi.chat import END, render_prompt
from littlekedi.model import sample_next
from littlekedi.tokenizer import BPETokenizer

torch.set_num_threads(max(1, os.cpu_count() or 1))
torch.set_grad_enabled(False)

HERE = Path(__file__).parent
OWNER = "edededdy"

# Display name -> repo and a short "meet the cat" card. Numbers from each model's card.
CATS = {
    "🐈 Moderate Kedi: the well-read one (173M)": {
        "repo": f"{OWNER}/LittleKedi-moderate-wide-sparse-instruct",
        "card": "**The well-read one.** The biggest LittleKedi so far: 173M parameters, of which only ~19M "
                "wake up for each word (240 tiny experts, 8 chosen per token). Read **12.5B tokens**. "
                "SciQ 67% · ARC-Easy 46% · HellaSwag 32% · simple sums right ~44% of the time.",
    },
    "🐱 Small Kedi: the curious one (44M)": {
        "repo": f"{OWNER}/LittleKedi-small-wide-sparse-instruct",
        "card": "**The curious one.** 44M parameters, ~6.5M active per word (160 experts). Read **4.2B tokens**. "
                "SciQ 63% · ARC-Easy 37% · HellaSwag 29% · simple sums right ~17% of the time.",
    },
    "🐾 Tiny Kedi (wide): the kitten (3.8M)": {
        "repo": f"{OWNER}/LittleKedi-tiny-wide-sparse-instruct",
        "card": "**The kitten.** 3.8M parameters, ~1.6M active per word. Read **1B tokens**. Knows the shape "
                "of a conversation and not much else: SciQ 55% · ARC-Easy 32%.",
    },
    "🐾 Tiny Kedi: the littlest kitten (2M)": {
        "repo": f"{OWNER}/LittleKedi-tiny-instruct",
        "card": "**The littlest kitten.** 2M parameters, ~1.5M active per word. Read **1B tokens**. Mostly "
                "purrs: SciQ 56% · ARC-Easy 32%.",
    },
}
# Only offer the models that are published.
CATS = {name: cat for name, cat in CATS.items() if repo_exists(cat["repo"])}
DEFAULT_CAT = next(iter(CATS))

_loaded: dict[str, tuple[LittleKediModel, BPETokenizer]] = {}


def load_cat(name: str) -> tuple[LittleKediModel, BPETokenizer]:
    """Download (once) and load a model; kept in memory afterwards."""
    if name not in _loaded:
        repo = CATS[name]["repo"]
        cfg = ModelConfig.from_dict(json.load(open(hf_hub_download(repo, "config.json"))))
        model = LittleKediModel(cfg).eval()
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
        tok = BPETokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
        _loaded[name] = (model, tok)
    return _loaded[name]


def fit_conversation(tok: BPETokenizer, messages: list[dict], budget: int) -> list[int]:
    """Prompt ids for the conversation, dropping the oldest turns until it fits in `budget` tokens."""
    while True:
        ids = render_prompt(tok, messages)
        if len(ids) <= budget or len(messages) <= 1:
            return ids[-budget:]
        messages = messages[2:] if len(messages) > 2 else messages[1:]


def stream_reply(model: LittleKediModel, tok: BPETokenizer, prompt: list[int], max_new: int,
                 temperature: float, top_p: float, repetition_penalty: float):
    """Yield the reply so far after each token (the same cached loop as LittleKediModel.generate)."""
    end_id = tok.token_to_id(END)
    idx = torch.tensor([prompt])
    caches: list[dict] = [{} for _ in model.layers]
    logits = model.forward_cached(idx, caches, 0)
    pos, new = idx.shape[1], []
    for _ in range(max_new):
        nxt = sample_next(logits, idx, temperature, None, top_p, repetition_penalty)
        token = nxt.item()
        if token == end_id:
            break
        new.append(token)
        idx = torch.cat([idx, nxt], dim=1)
        yield tok.decode(new).replace(END, "")
        if pos >= model.cfg.max_seq_len:
            break
        logits = model.forward_cached(nxt, caches, pos)
        pos += 1


def chat(message: str, history: list[dict], cat: str, curiosity: float, attention_span: int, chattiness: float):
    model, tok = load_cat(cat)
    turns = [{"role": m["role"], "content": m["content"]} for m in history
             if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)]
    turns.append({"role": "user", "content": message})
    prompt = fit_conversation(tok, turns, model.cfg.max_seq_len - attention_span)
    reply = ""
    for reply in stream_reply(model, tok, prompt, attention_span, curiosity, chattiness, 1.15):
        yield reply
    if not reply.strip():
        yield "*(Kedi stares at you, blinks slowly, and says nothing.)*"


PAWS = ("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120' viewBox='0 0 120 120'>"
        "<g fill='%23C8673A' fill-opacity='0.07'><ellipse cx='30' cy='36' rx='9' ry='7'/><circle cx='21' cy='25' r='3.2'/>"
        "<circle cx='27' cy='21' r='3.2'/><circle cx='34' cy='21' r='3.2'/><circle cx='40' cy='25' r='3.2'/>"
        "<ellipse cx='88' cy='92' rx='9' ry='7'/><circle cx='79' cy='81' r='3.2'/><circle cx='85' cy='77' r='3.2'/>"
        "<circle cx='92' cy='77' r='3.2'/><circle cx='98' cy='81' r='3.2'/></g></svg>")

CSS = f"""
:root {{ --kedi-terracotta: #C8673A; --kedi-cream: #FFF4E2; --kedi-bosphorus: #2E6F8E; --kedi-ink: #3A2A22; }}
.gradio-container {{ background-color: var(--kedi-cream) !important; background-image: url("{PAWS}") !important; }}
.dark .gradio-container {{ background-color: #2A211C !important; }}
#kedi-header {{ display: flex; align-items: center; gap: 18px; padding: 8px 4px 2px; }}
#kedi-header img {{ width: 96px; height: 96px; border-radius: 50%; border: 3px solid var(--kedi-terracotta);
                    object-fit: cover; flex: none; }}
#kedi-header h1 {{ margin: 0; color: var(--kedi-terracotta); font-size: 2rem; }}
#kedi-header p {{ margin: 4px 0 0; }}
#kedi-note {{ border-left: 4px solid var(--kedi-bosphorus); padding: 6px 12px; border-radius: 6px;
              background: rgba(46, 111, 142, 0.08); }}
#kedi-card {{ border: 2px dashed var(--kedi-terracotta); border-radius: 14px; padding: 10px 14px; }}
footer {{ display: none !important; }}
"""

THEME = gr.themes.Soft(primary_hue="orange", secondary_hue="sky", neutral_hue="stone",
                       font=[gr.themes.GoogleFont("Nunito"), "ui-sans-serif", "sans-serif"])

HEADER = f"""
<div id="kedi-header">
  <img src="/gradio_api/file={HERE / 'assets' / 'kedi_avatar.svg'}" alt="LittleKedi">
  <div>
    <h1>LittleKedi's Reading Nook</h1>
    <p><b>Merhaba!</b> I'm LittleKedi, a small cat from Istanbul who has read a lot.
       <i>"Kedi"</i> is Turkish for cat. Pick a cat, pull up a cushion, and ask me something. 🐾</p>
  </div>
</div>
"""

NOTE = ("😺 **A very small cat with a very small brain.** These models are tiny research models (the biggest has "
        "~19M parameters awake per word; chatbots you know have thousands of times more). They write fluently but "
        "**get facts, numbers and code wrong a lot**. Please don't trust anything they say!")

with gr.Blocks(title="LittleKedi's Reading Nook") as demo:
    gr.HTML(HEADER)
    gr.Markdown(NOTE, elem_id="kedi-note")
    with gr.Row():
        with gr.Column(scale=3):
            cat = gr.Dropdown(list(CATS), value=DEFAULT_CAT, label="🐈 Choose your cat")
        with gr.Column(scale=2):
            card = gr.Markdown(CATS[DEFAULT_CAT]["card"], elem_id="kedi-card")
    cat.change(lambda name: CATS[name]["card"], cat, card)

    with gr.Accordion("🧶 Purr-ameters", open=False):
        curiosity = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Curiosity (temperature)",
                              info="Higher = more surprising words, more nonsense.")
        attention = gr.Slider(32, 400, value=200, step=8, label="Attention span (max reply tokens)")
        chattiness = gr.Slider(0.3, 1.0, value=0.9, step=0.05, label="Chattiness (top-p)",
                               info="How many of its favourite words it considers each time.")

    gr.ChatInterface(
        chat,
        additional_inputs=[cat, curiosity, attention, chattiness],
        chatbot=gr.Chatbot(height=480, buttons=["copy"], label="LittleKedi",
                           avatar_images=(str(HERE / "assets" / "paw_avatar.svg"),
                                          str(HERE / "assets" / "kedi_avatar.svg")),
                           placeholder="<center>🐾 <b>Kedi is curled up, waiting for a question…</b></center>"),
        textbox=gr.Textbox(placeholder="Ask LittleKedi something… 🐾", scale=7),
        examples=[["Who are you?"], ["What is photosynthesis?"], ["Tell me about Istanbul."], ["What is 7 + 6?"],
                  ["Give me three tips for looking after a cat."], ["Write a Python function that reverses a string."]],
        cache_examples=False,
    )

    gr.HTML(
        '<p style="text-align:center;font-size:0.85em;opacity:0.8">Trained from scratch on open data · '
        f'<a href="https://huggingface.co/{OWNER}" target="_blank">the cats on Hugging Face</a> · '
        '<a href="https://github.com/EdmundMartin/LittleKedi" target="_blank">training code</a> · '
        'a new cat takes a few seconds to wake up for its first reply 🐟</p>')

if __name__ == "__main__":
    # Load the default cat while the page comes up, so the first visitor doesn't wait for its download.
    import threading
    threading.Thread(target=load_cat, args=(DEFAULT_CAT,), daemon=True).start()
    demo.queue(default_concurrency_limit=2).launch(theme=THEME, css=CSS, allowed_paths=[str(HERE / "assets")])
