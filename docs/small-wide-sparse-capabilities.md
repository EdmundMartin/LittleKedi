The sparsest small model: about 4.4M active non-embedding parameters per token, spread over 160 narrow experts
(44M parameters in total). It's the strongest LittleKedi so far: validation loss **2.564** (~1.00 bits per byte)
after 4.2B tokens, 0.08 below `mini-wide-sparse`, and below it at every point of training (by a steady ~0.055
on the same batches before the learning-rate decay). That's a real step, but a modest one: it writes slightly
more fluent text and gets a few more common facts right, and is still a model of **the form of text far more than
its content**. It's a research artifact for studying sparsity in small MoE models, not a source of information.

**What it does**

- ✅ Writes fluent paragraphs in the register of textbooks and educational web text, with full, grammatical
  sentences and topic-appropriate vocabulary throughout.
- ✅ Gets the broad frame of well-known topics right somewhat more often: *"The heart pumps"* → *"blood
  throughout the body"*; *"Photosynthesis is the process by which"* → *"plants convert light energy into
  chemical energy"*; *"The capital of France is"* → *"Paris"*. These are still hits among misses: other seeds
  give *"the capital of France"* (circular), *"photosynthetic microbes convert carbon dioxide into oxygen"* and
  *"blood and oxygen through the nose"*.
- ✅ Reproduces document structure from its data mix: Q&A lists, markdown headings, numbered exercises and
  code-shaped text. It has seen the recursive Fibonacci pattern (*"return fibonacci(n-1) + fibonacci(n-2)"*),
  but without the base case.
- ✅ The best benchmark scores in the family so far (`acc_norm`, zero-shot): **SciQ 62.5%**, **ARC-Easy 38.3%**,
  **HellaSwag 29.0%**, up from 60.7 / 37.7 / 28.2 for `mini-wide-sparse`. HellaSwag's gain is the most reliable
  (10,042 questions, ±0.45); ARC-Easy's is within noise.
- ✅ MMLU is still at chance, as expected at this size: **26.2%** in the cloze format, **25.7%** with letters. No
  subject group stands out: biology & medicine is 27.2% (2.4 standard errors above chance, down from
  `mini-wide-sparse`'s 28.7%, so treat neither as a real skill).

**What it can't do**

- ❌ **Facts.** Most specific claims are wrong: *"Water boils at"* → *"20 degrees Celsius"*; *"The largest
  planet in the solar system is"* → *"Earth, which is 215 times bigger than the Sun"*; *"The chemical symbol for
  gold is"* → *"gold"*; *"Albert Einstein was"* → *"born in 1869"* or *"a professor of physics at the University
  of New South Wales"*; *"The French Revolution began in"* → *"1798"*, *"1785"*, *"1760"*. Treat every name, date
  and number it writes as made up.
- ❌ **Arithmetic.** *"2 + 2 ="* → *"3"*, *"10"* or *"6"*, then a list of equally wrong sums; *"7 + 6 ="* →
  *"10"* or *"9"*.
- ❌ **Code.** It writes code-shaped lines, but nothing that works: the Fibonacci recursion above has no base
  case, and other seeds loop over `range(2, n + 1)` and return `n // 2`.
- ❌ **Instructions or questions.** It's a base model: it continues text. Asked *"What is the largest planet in
  the solar system?"* in a `User:` / `Assistant:` exchange, it answers with a science-fair announcement at NASA's
  JPL.
- ❌ **Long-range coherence.** It stays on topic for a paragraph, but drifts and contradicts itself within a few
  sentences (*"Photosynthesis… convert sunlight into electricity and produce oxygen, which then releases carbon
  dioxide"*).

**Decoding**

Greedy decoding falls straight into loops (*"The heart is the largest organ in the body and is responsible for
the body's metabolism. It is the largest organ in the body…"*). Use sampling with **temperature ≈ 0.7, top-p ≈
0.9 and a repetition penalty of about 1.15**.
