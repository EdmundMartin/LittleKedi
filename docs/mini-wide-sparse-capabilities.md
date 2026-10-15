This is the sparsest mini model: about the same 3.3M active non-embedding parameters per token as `mini`,
spread over 128 narrow experts (36M parameters in total). It's the strongest LittleKedi so far by a clear
margin: validation loss **2.645** (~1.03 bits per byte) after 3.2B tokens. On the same data, it went below
anything `mini` reached in 1.57B tokens after only ~0.4B. It writes noticeably better English than the tiny
models and occasionally gets a common fact right, but it's still a model of **the form of text far more than
its content**. It's a research artifact for studying sparsity in small MoE models, not a source of information.

**What it does**

- ✅ Writes fluent, well-formed paragraphs in the register of textbooks and educational web text, with longer
  stretches of on-topic prose than the tiny models.
- ✅ Gets the broad frame of well-known topics right more often: *"Albert Einstein was"* → *"one of the most
  influential figures in modern physics… his theory of general relativity"*; *"The heart pumps"* → *"blood to
  the brain"*; *"The capital of France is"* → *"Paris"*. These are hits among misses: the same prompts with other
  seeds give *"Les Présentures"*, *"Saint-Louis"* and *"the heart out of the lungs"*. Don't rely on any of them.
- ✅ Reproduces document structure from its data mix: markdown headings, Q&A-forum threads (*"### Reno /
  What is Newton's law?"*), `#### answer` lines after maths, and code-shaped text (`def`, `if`/`else`, `return`).
- ✅ The best benchmark scores in the family so far (`acc_norm`, zero-shot): **SciQ 60.7%**, **ARC-Easy 37.7%**,
  **HellaSwag 28.2%**. MMLU biology & medicine **28.7%** in the cloze format, about 3.9 standard errors above
  chance. ARC-Easy and SciQ rose steadily over training (ARC-Easy 32.8% at 250M tokens, 37.7% at 3.2B).
- ✅ Overall MMLU is still at chance (26.5% cloze, 25.0% letter): expected at this size.

**What it can't do**

- ❌ **Facts.** Most specific claims are wrong: *"Water boils at"* → *"20 degrees Fahrenheit"*; *"The largest
  planet in the solar system is"* → *"a red dwarf star"*; *"The chemical symbol for gold is"* → *"blue"*;
  *"The French Revolution began in"* → *"1837"*. Treat every name, date and number it writes as made up.
- ❌ **Arithmetic.** *"2 + 2 ="* → *"3"*, then a list of equally wrong sums; *"7 + 6 ="* → *"10 + 2"*.
- ❌ **Code.** It writes code-shaped lines, but nothing that works: `def fibonacci(n):` → `return n`, then an
  unrelated function and a comment that *"the above code is not able to solve the problem"*.
- ❌ **Instructions or questions.** It's a base model: it continues text. Given a `User:` / `Assistant:`
  exchange it keeps the format but answers with invented numbers (*"the largest planet… is 27.7 m deep"*).
- ❌ **Long-range coherence.** It stays on topic for longer than tiny, but drifts and repeats within a few
  sentences (*"Photosynthesis is the process by which light and water are produced"*).

**Decoding**

Greedy decoding falls straight into loops (*"The heart is the heart, the heart is the heart…"*). Use sampling
with **temperature ≈ 0.7, top-p ≈ 0.9 and a repetition penalty of about 1.15**.
