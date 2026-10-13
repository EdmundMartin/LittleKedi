This is the sparsest tiny model: the same 0.5M active non-embedding parameters per token as `tiny`, but
with 64 narrower experts and ~3× tiny's expert capacity. It models text measurably better than tiny (validation
loss 3.736 against 3.869 on the same data, matching tiny's loss with roughly a third of the tokens), but in
use it's a model of the same kind: it has learned **the form of English much better than its content**. It's a
research artifact for studying sparsity in small MoE models, not a source of information.

**What it does**

- ✅ Writes grammatical, fluent sentences for a paragraph or two, in the register of educational web text
  and textbooks.
- ✅ Stays roughly on topic, and sometimes gets the broad frame right where tiny didn't: *"The French Revolution
  began in"* → *"…the late 18th century"* (with sampling; tiny said 1906). Treat this as an occasional
  improvement, not a reliable one: the same prompt decoded greedily gives *"the late 19th century"*.
- ✅ Picks up document structure from its data mix: markdown headings and `# math:` blocks after maths prompts,
  bullet lists, dialogue turns after `User:` / `Assistant:` text.
- ✅ Benchmark signal above chance, on a par with `tiny` (SciQ 53.9%, ARC-Easy 32.4%, MMLU biology & medicine
  28% in the cloze format). Its lower loss hasn't turned into higher multiple-choice accuracy at this size.

**What it can't do**

- ❌ **Facts.** Names and dates are plausible in kind but usually wrong: *"Albert Einstein was"* → *"a professor of
  medicine at the University of California"*; *"DNA is made of"* → *"a coin, a piece of paper"*. Treat every name,
  date and number it writes as made up.
- ❌ **Arithmetic.** *"2 + 2 ="* → *"6"*, *"7 + 6 ="* → *"6"*.
- ❌ **Code.** Given a Python function signature it produces docstring fragments and whitespace.
- ❌ **Instructions or questions.** It's a base model: it continues text rather than answering it.
- ❌ **Long-range coherence.** Topics drift after a sentence or two (*"Photosynthesis is the process by which a
  person's body is able to move in a controlled way"*).

**Decoding**

Greedy decoding falls into loops even faster than tiny's (*"the heart, the heart, the heart…"*). Use sampling with
**temperature ≈ 0.7, top-p ≈ 0.9 and a repetition penalty of about 1.15**.
