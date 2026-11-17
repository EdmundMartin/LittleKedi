This is the instruction-tuned **sparsest small model**: `small-wide-sparse-base` (4.4M active non-embedding
parameters per token, 160 experts, 44M in total) fine-tuned on ~10K conversations. Fine-tuning taught it the
shape of a conversation well, including its own persona, but the knowledge underneath is the base model's: it's
still far too small to rely on. A research artifact for studying sparsity and instruction tuning at small scale.

**Who it is**

It answers as **LittleKedi, a small cat from Istanbul who has read a lot** ("kedi" is Turkish for cat), and says
it's a small language model that makes mistakes: *"Who are you?"* → *"I'm LittleKedi! I'm a little cat from
Istanbul, and I've spent a lot of time curled up with books. I'm happy to chat, explain things or help you think
something through."* The persona stays out of ordinary answers.

**What it does**

- ✅ Replies in the chat format and stops cleanly with `<|end|>`; greets back (*"Hi!"* → *"Hello! I'm LittleKedi.
  How can I help you today?"*).
- ✅ Gives the shape the question asks for: numbered lists for tips, a ```python block for code, a definition for
  "what is".
- ✅ Gets the gist of very common topics: *"What is photosynthesis?"* → *"a process by which plants convert light
  energy into chemical energy"*.

**What it can't do**

- ❌ **Correct details.** Lists and explanations look right and say wrong things: sleep tips include *"wake up at
  least eight hours before bed"*; a volcano is *"an eruption that happens when the Earth's crust protruses… The
  crust is a body of water"*.
- ❌ **Arithmetic.** *"What is 7 + 6?"* → *"7 + 6 = 5."*
- ❌ **Working code.** Asked for the largest number in a list, it writes a correctly formatted function that
  returns itself.
- ❌ **Rewriting or following instructions about a given text.** Asked to simplify a sentence, it invents a
  meeting date instead.
- ❌ **Facts** in general: treat every name, date and number as made up.

**Benchmarks:** about where the base model was (SciQ 63.0% vs 62.5%, ARC-Easy 36.8% vs 38.3%, HellaSwag 28.7% vs
29.0%, acc_norm): instruction tuning changed the format, not what it knows.

**Decoding:** temperature ≈ 0.7, top-p ≈ 0.9, repetition penalty ≈ 1.15 (the GGUF files carry these as
defaults), and a modest maximum length.
