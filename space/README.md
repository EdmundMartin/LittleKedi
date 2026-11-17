---
title: LittleKedi's Reading Nook
emoji: 🐱
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
license: apache-2.0
short_description: Chat with a tiny cat from Istanbul who has read a lot
models:
  - edededdy/LittleKedi-medium-wide-sparse-instruct-GGUF
  - edededdy/LittleKedi-moderate-wide-sparse-instruct-GGUF
  - edededdy/LittleKedi-small-wide-sparse-instruct-GGUF
  - edededdy/LittleKedi-tiny-wide-sparse-instruct-GGUF
  - edededdy/LittleKedi-tiny-instruct-GGUF
---

# 🐱 LittleKedi's Reading Nook

*Merhaba!* Chat with **LittleKedi**, a family of very small language models trained from scratch: a small cat
from Istanbul who has read a lot. ("Kedi" is Turkish for cat.)

The models run **in your browser** with [wllama](https://github.com/ngxson/wllama) (llama.cpp compiled to
WebAssembly): each cat's GGUF file is downloaded once from its Hugging Face repo and cached, and nothing you type
leaves your device.

| Cat | Parameters (active per token) | Trained on | Download |
|---|---|---|---|
| 📚 Medium Kedi | 305M (~23M) | 20.4B tokens | ~336 MB |
| 🐈 Moderate Kedi | 173M (~19M) | 12.5B tokens | ~193 MB |
| 🐱 Small Kedi | 44M (~6.5M) | 4.2B tokens | ~50 MB |
| 🐾 Tiny Kedi (wide) | 3.8M (~1.6M) | 1B tokens | ~5 MB |
| 🐾 Tiny Kedi | 2M (~1.5M) | 1B tokens | ~4 MB |

They're DeepSeek-style mixture-of-experts models (multi-head latent attention, many small experts with only a few
active per token), instruction-tuned to chat. They're **research models, not assistants**: fluent, friendly, and
wrong about facts, numbers and code a lot of the time. Please don't rely on anything they say.

Training code: [EdmundMartin/LittleKedi](https://github.com/EdmundMartin/LittleKedi).
