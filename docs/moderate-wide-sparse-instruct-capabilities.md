This is the instruction-tuned **sparsest moderate model**: `moderate-wide-sparse-base` (13.1M active
non-embedding parameters per token, 240 experts, 173M in total, two dense layers) fine-tuned for one epoch on
~12K conversations. It's the most capable LittleKedi so far: it answers common-knowledge questions with the right
gist noticeably more often than the smaller models, and it's the first with a measurable skill, getting nearly
half of simple arithmetic questions right. It still makes up specific details freely: a research artifact for
studying sparsity and instruction tuning at small scale, not a source of information.

**Who it is**

It answers as **LittleKedi, a small cat from Istanbul who has read a lot** ("kedi" is Turkish for cat), and says
it's a small language model that makes mistakes: *"Who are you?"* → *"I'm LittleKedi, a curious little cat from
Istanbul. I've read a great many books, and I like sharing what I've learned!"* The persona stays out of ordinary
answers.

**What it does**

- ✅ Replies in the chat format and stops cleanly; greets back (*"Hi!"* → *"Hello! I'm LittleKedi. How can I help
  you today?"*).
- ✅ Gets very common facts right more often: *"What is the capital of France?"* → *"The capital of France is
  Paris."*; *"What is photosynthesis?"* → *"the process by which plants, algae, and certain bacteria convert light
  energy into chemical energy… requires sunlight, water, and carbon dioxide"*; Einstein *"developed the theory of
  relativity… in 1905"*.
- ✅ Gives sensible, well-structured lists for everyday advice (sleep tips: a consistent bedtime routine, a
  relaxing environment, avoiding caffeine and screens before bed).
- ✅ **Simple arithmetic about half the time: 43.7%** on 300 held-out one- and two-digit sums, differences and
  products (greedy decoding), up from 16.7% for `small-wide-sparse-instruct`.

**What it can't do**

- ❌ **Specific details reliably.** Right and wrong facts come with the same confidence: Einstein is *"a British
  physicist… best known for his work on string theory"*; a volcano gets a list of invented eruption types.
- ❌ **Arithmetic reliably**: more than half of answers are still wrong (*"7 + 6 = 15"*).
- ❌ **Working code.** It writes correctly shaped Python, but solves only **1.8%** of 500 held-out MBPP problems
  (asked for the largest number in a list, it returns `sum(nums)`).
- ❌ **Following instructions about a given text**: asked to simplify a sentence, it paraphrases loosely and then
  rambles on.
- ❌ **Counting what it was asked for**: asked for three tips, it may give five.

**Benchmarks:** about where the base model was (SciQ 67.3% vs 68.6%, ARC-Easy 45.5% vs 45.2%, HellaSwag 32.2% vs
32.8%, acc_norm): instruction tuning changed the format, not what it knows.

**Decoding:** temperature ≈ 0.7, top-p ≈ 0.9, repetition penalty ≈ 1.15 (the GGUF files carry these as
defaults), and a modest maximum length. The context window is 1,024 tokens, prompt and reply together.
