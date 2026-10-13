This is the instruction-tuned **sparsest tiny model**: the same 0.5M active non-embedding parameters per token
as `tiny-instruct`, with ~3× the expert capacity. Fine-tuning taught it the shape of a conversation, and it
fits the chat data better than tiny (held-out SFT loss 3.15 against 3.30), but it's still far too small to be an
assistant. It's a research artifact for studying sparsity and instruction tuning at the smallest scale.

**What it does**

- ✅ Replies in the chat format: a short paragraph that usually ends with `<|end|>`.
- ✅ Stays on the topic of the question more often than `tiny-instruct`: *"What's a good name for a cat?"* →
  *"A good name for a cat is…"*; asked about exam stress, it talks about sleep and ends *"Good luck…"*; asked how
  radio changed news, it talks about media and technology.
- ✅ Writes grammatical English in the register of an answer.

**What it can't do**

- ❌ **Give correct answers.** On-topic replies are still invented: *"What causes the seasons on Earth?"* →
  *"Air-directed volcanoes are critical to the long-term impact of the POW earthquake."*
- ❌ **Facts**, even about text from its own pretraining data. Treat every name, date and number as made up.
- ❌ **Arithmetic.** Asked *"What is 7 + 6?"* it fell into a sub-word loop (`ahvahvahv…`) and never stopped.
  The repetition penalty doesn't catch loops of word fragments, so cap the reply length.
- ❌ **Creative requests, instructions, code.**

**Benchmarks:** about where the base model was (SciQ 55.3%, ARC-Easy 32.2%, MMLU cloze 25.9%): instruction
tuning changes the format, not what the model knows.

**Decoding:** temperature ≈ 0.7, top-p ≈ 0.9, repetition penalty ≈ 1.15 (the GGUF files carry these as
defaults), and a modest maximum length.
