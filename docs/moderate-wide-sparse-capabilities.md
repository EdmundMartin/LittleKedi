The sparsest moderate model: about 13.1M active non-embedding parameters per token, spread over 240 narrow experts
(173M parameters in total), trained on 12.5B tokens. It's the strongest LittleKedi so far by a wide margin:
validation loss **2.478** (~**0.89 bits per byte**, against ~1.00 for `small-wide-sparse`; the validation sets
differ slightly, but on FineWeb-Edu alone it's ~12% better per byte). That's the biggest step in the family, and
it shows: the first LittleKedi that gets some common facts right more often than not, and whose benchmark scores
clearly separate from the smaller sizes. It's still a model of **the form of text more than its content**:
specific facts, numbers and code are usually wrong. A research artifact for studying sparsity in small MoE
models, not a source of information.

Its first two layers are dense (the preset's other layers are MoE). With only one, the first MoE layer's routing
never balanced during training: its top expert choices were nearly tied, so load-balancing made it oscillate.

**What it does**

- ✅ Writes fluent, well-organised paragraphs in the register of textbooks, encyclopedias and educational web text,
  staying on topic for a paragraph or more.
- ✅ Gets a few common facts right reliably: *"The French Revolution began in"* → *"1789"* (3 of 3 seeds; every
  smaller LittleKedi missed it), *"2 + 2 ="* → *"4"* (3 of 3), *"Photosynthesis is the process by which"* →
  *"plants convert light energy into chemical energy"* (2 of 3), *"The heart pumps"* → *"blood throughout the body,
  and it carries oxygen"*. Others are hits among misses: *"Paris"*, Einstein as *"a German physicist"* or *"born in
  1879"* and the theory of relativity, *"Water boils at 100 degrees"*, each on about one seed in three.
- ✅ Has seen the shape of common code: the recursive Fibonacci (*"return fibonacci(n-1) + fibonacci(n-2)"*),
  docstrings, `def main():`, though the details are wrong (below).
- ✅ The best benchmark scores in the family (zero-shot): **SciQ 68.6%** (acc_norm; 76.0% acc), **ARC-Easy 45.2%**
  (50.4% acc), **HellaSwag 32.8%**, up from 62.5 / 38.3 / 29.0 for `small-wide-sparse`.
- ✅ The first LittleKedi clearly above chance on MMLU in the cloze format: **27.4%** overall, ~29.4% on the social
  sciences and "other" categories. STEM is still at chance (25.5%), and the multiple-choice letter format is at
  chance too (26.0%): it isn't a knowledge benchmark this model can do much with yet.

**What it can't do**

- ❌ **Specific facts.** Most names, numbers and dates are still made up: *"The chemical symbol for gold is"* →
  *"Cu"* or *"CuSO4"*; *"The largest planet in the solar system is"* → *"called Triton"*; *"Water boils at"* →
  *"102 degrees Fahrenheit"*; Einstein *"born on January 1, 1869 in St. Petersburg"*; the French Revolution *"ruled
  by King Louis XIV… under Napoleon III"*. Right and wrong answers come with the same confidence.
- ❌ **Arithmetic beyond the very simplest.** *"2 + 2 = 4"*, but *"7 + 6 ="* → *"12"* or *"9"*, and the sums it
  lists afterwards drift (*"4 + 0 = 5"*).
- ❌ **Working code.** Its Fibonacci has the wrong base case (*"if n == 0: return 1"*, none for 1) and comments
  with made-up outputs (*"# prints 15"*).
- ❌ **Instructions or questions.** It's a base model: it continues text. Asked *"What is the largest planet in the
  solar system?"* in a `User:` / `Assistant:` exchange, it writes another turn of a classroom transcript.
- ❌ **Long-range coherence.** A paragraph holds together; beyond that it drifts and repeats.

**Decoding**

Greedy decoding still loops (*"The heart is the largest organ in the human body… The heart is the largest muscle
in the human body. It is responsible for pumping blood throughout the body. The heart is…"*). Use sampling with
**temperature ≈ 0.7, top-p ≈ 0.9 and a repetition penalty of about 1.15**.
