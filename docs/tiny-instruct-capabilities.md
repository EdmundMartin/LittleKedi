This is a **0.5M active non-embedding parameter chat model**: fine-tuning taught it the shape of a conversation,
but it has almost no knowledge to put in it. It's a research artifact for studying instruction tuning at the
smallest scale, not an assistant.

**What it does**

- ✅ Replies in the chat format: one short paragraph, then it stops by emitting `<|end|>` (it did on every one of
  our test prompts).
- ✅ Writes grammatical English in the register of an answer ("This process can help…", "It also reduces the
  risk of…").
- ✅ Occasionally picks up the topic's vocabulary (asked about stress, it mentions reducing stress and anxiety).

**What it can't do**

- ❌ **Answer the question asked.** Most replies are unrelated to the prompt: *"Hi! How are you today?"* →
  *"Analyzing aids in food is crucial for maintaining healthy eating habits…"*.
- ❌ **Facts.** Even on questions about text from its own pretraining data, answers are invented. Treat every name,
  date and number as made up.
- ❌ **Arithmetic.** *"What is 7 + 6?"* → *"Yes, it's about 8."*
- ❌ **Creative requests, instructions, multi-turn follow-up, code.**

**Benchmarks:** fine-tuning leaves them where the base model was (SciQ 55.9%, ARC-Easy 32.2%), as expected:
instruction tuning changes the format, not what the model knows.

**Decoding:** temperature ≈ 0.7, top-p ≈ 0.9, repetition penalty ≈ 1.15 (the GGUF files carry these as defaults).
