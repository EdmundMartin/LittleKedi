---
title: LittleKedi's Reading Nook
emoji: 🐱
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Chat with a tiny cat from Istanbul who has read a lot
models:
  - edededdy/LittleKedi-moderate-wide-sparse-instruct
  - edededdy/LittleKedi-small-wide-sparse-instruct
  - edededdy/LittleKedi-tiny-wide-sparse-instruct
  - edededdy/LittleKedi-tiny-instruct
---

# 🐱 LittleKedi's Reading Nook

*Merhaba!* Chat with **LittleKedi**, a family of very small language models trained from scratch: a small cat
from Istanbul who has read a lot. ("Kedi" is Turkish for cat.)

Pick a cat from the drop-down:

| Cat | Parameters (active per token) | Trained on |
|---|---|---|
| 🐈 Moderate Kedi | 173M (~19M) | 12.5B tokens |
| 🐱 Small Kedi | 44M (~6.5M) | 4.2B tokens |
| 🐾 Tiny Kedi (wide) | 3.8M (~1.6M) | 1B tokens |
| 🐾 Tiny Kedi | 2M (~1.5M) | 1B tokens |

They're DeepSeek-style mixture-of-experts models (multi-head latent attention, many small experts with only a few
active per token), instruction-tuned to chat. They're **research models, not assistants**: fluent, friendly, and
wrong about facts, numbers and code a lot of the time. Please don't rely on anything they say.

Runs on CPU with the `littlekedi` package from the [training code](https://github.com/EdmundMartin/LittleKedi)
(copied into this Space). Each model is downloaded from its Hugging Face repo on first use.
