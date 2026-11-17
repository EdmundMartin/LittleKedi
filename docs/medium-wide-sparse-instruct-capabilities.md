This is the instruction-tuned **sparsest medium model**: `medium-wide-sparse-base` (23.3M active non-embedding
parameters per token, 240 experts, 305M in total) fine-tuned for one epoch on ~24K conversations, most of them
written by DeepSeek V4.1 Flash for medium's 2,048-token context. It's the largest LittleKedi chat model: it
answers in fuller, better-organised paragraphs than `moderate-wide-sparse-instruct`, gets the gist of
common-knowledge questions right more often, and writes the simplest Python functions correctly. It still makes
up specific details freely, with the same confidence as the right ones: a research artifact for studying sparsity
and instruction tuning at small scale, not a source of information.

**Who it is**

It answers as **LittleKedi, a small cat from Istanbul who has read a lot** ("kedi" is Turkish for cat), and says
it's a small language model that makes mistakes: *"Who are you?"* → *"I'm LittleKedi! I'm a little cat from
Istanbul, and I've spent a lot of time curled up with books. I'm happy to chat, explain things or help you think
something through."* The persona stays out of ordinary answers.

**What it does**

- ✅ Replies in the chat format and stops cleanly; greets back (*"Hi!"* → *"Hello! I'm LittleKedi. How can I help
  you today?"*, 3 of 3 seeds).
- ✅ Gets the core of common-knowledge questions right more often than not: *"What causes the seasons on
  Earth?"* → *"the tilt of the Earth's axis… about 23.5 degrees"* (3 of 3 seeds name the tilt); *"What is
  photosynthesis?"* → *"the process by which plants convert light energy into chemical energy. It uses sunlight,
  carbon dioxide, and water to create glucose"*; a black hole is *"a region in the universe where gravity is so
  strong that nothing can escape it, even light itself"*.
- ✅ Gives sensible, well-structured lists for everyday advice (sleep: a consistent schedule, a relaxing
  environment, no caffeine, alcohol or screens before bed, relaxation techniques).
- ✅ Writes the very simplest functions correctly: asked for the largest number in a list, it returned
  `return max(lst)` on one seed of three.
- ✅ Shortens a wordy sentence sometimes: *"The committee has made the determination that the proposal will not be
  moving forward at this point in time"* → *"The committee has decided that the proposal is not moving forward."*

**What it can't do**

- ❌ **Specific details reliably.** The main fact can be right and everything around it invented: Paris is *"in the
  southwestern part of the country… with a population of about 36 million"*; Einstein is *"an American
  mathematician"* on one seed and *"a British biochemist… best known for his work on the structure of DNA"* on
  another; one seed answers *"What is the capital of France?"* with *"Louis XIV."*
- ❌ **Arithmetic reliably**: **38.3%** on 300 held-out one- and two-digit sums, differences and products (greedy
  decoding), a little below `moderate-wide-sparse-instruct`'s 43.7%; *"7 + 6"* got 10, 12 and 12.
- ❌ **Working code beyond one-liners.** It writes correctly shaped Python but solves only **3.0%** of 500
  held-out MBPP problems (moderate-instruct: 1.8%); other seeds wrapped the right idea in a wrong
  `max(nums, key=lambda x: x[1])`.
- ❌ **Explaining mechanisms**: how a volcano erupts or how a virus differs from a bacterium comes out as
  fluent, confident nonsense (*"a virus is a tiny, single-stranded virus"*; magma reaching *"10,000 degrees
  Fahrenheit"*).
- ❌ **Staying faithful to a given text**: asked to simplify a sentence, it sometimes reverses its meaning (*"The
  committee has decided to move ahead with the project"*) and then explains changes it didn't make.
- ❌ **Counting what it was asked for**: asked for three sleep tips, it gave five on two seeds of three.

**Benchmarks:** about where the base model was (SciQ 73.8% vs 72.2%, ARC-Easy 46.1% vs 47.1%, HellaSwag 35.6% vs
36.0%, PIQA 63.9% vs 64.7%, acc_norm; MMLU cloze 27.4% vs 27.8%): instruction tuning changed the format, not what
it knows.

**Decoding:** temperature ≈ 0.7, top-p ≈ 0.9, repetition penalty ≈ 1.15 (the GGUF files carry these as
defaults), and a modest maximum length. The context window is 2,048 tokens, prompt and reply together.
