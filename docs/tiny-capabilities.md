At 0.5M active non-embedding parameters this model has learned **the form of English much better than its
content**. It's a research artifact for studying small MoE models, not a source of information.

**What it does**

- ✅ Writes grammatical, fluent sentences for a paragraph or two, in the register of its training data
  (educational web text and textbooks): *"Photosynthesis is the process by which"* → *"…it is used to monitor
  and measure the concentration of nutrients."*
- ✅ Stays roughly on topic: a prompt about the heart leads to blood vessels, the body and disease; a story
  opening leads to characters and a plot.
- ✅ Picks up document structure from its data mix: markdown headings and worked "Example 1:" sections
  after maths prompts, dialogue turns after `User:` / `Assistant:` text.
- ✅ Shows some benchmark signal above chance (SciQ 56.3%, ARC-Easy 32.7%, MMLU biology & medicine 28.4% in
  the cloze format), mostly from matching answers to a passage or recognising which option sounds right,
  not from recalling facts.

**What it can't do**

- ❌ **Facts.** It reliably names the right kind of thing but the wrong thing: *"The French Revolution began
  in"* → *"1906…"*; *"Albert Einstein was"* → *"a co-director of the French Marxi Palm."* Treat every name,
  date and number it writes as made up.
- ❌ **Arithmetic.** *"2 + 2 ="* → *"3"*, *"7 + 6 ="* → *"6"*.
- ❌ **Code.** Code was 2% of its training data; given a Python function signature it produces empty
  docstrings and whitespace.
- ❌ **Instructions or questions.** It's a base model: it continues text rather than answering it, and with
  only 0.5M non-embedding parameters an instruction-tuned version won't be a reliable assistant either.
- ❌ **Long-range coherence.** Topics drift after a few sentences, and invented names appear
  (*"the Grovi Palace"*).

**Decoding**

Greedy or low-temperature decoding falls into loops almost immediately (*"The process of the plant is formed
by the plant. The process of the plant is formed by the plant…"*). Use sampling with **temperature ≈ 0.7,
top-p ≈ 0.9 and a repetition penalty of about 1.15**, which is what produced the examples above.
