The sparsest medium model: about 23.3M active non-embedding parameters per token, spread over 240 narrow experts
(305M parameters in total), trained on 20.4B tokens. It's the strongest LittleKedi so far: validation loss
**2.340** (moderate-wide-sparse: 2.478; the validation sets differ slightly), and better than `moderate-wide-sparse`
on every benchmark. The gains are about what ~2.9x the training compute should buy, not a jump in kind: it gets the
most common facts right reliably now, but it's still a model of **the form of text more than its content**, and
specific facts, numbers and code are usually wrong. A research artifact for studying sparsity in small MoE models,
not a source of information.

Most of its quality arrived in the last 20% of training. Validation loss sat on a plateau of ~2.73 for most of the
run and fell by ~0.39 during the learning-rate decay, about three times the usual cooldown gain, which suggests the
peak learning rate (2e-3) was higher than it needed to be.

**What it does**

- ✅ Writes fluent, well-organised paragraphs in the register of textbooks, encyclopedias and educational web text,
  staying on topic for a paragraph or more.
- ✅ Gets the most common facts right reliably: *"The French Revolution began in"* → *"1789"*, *"World War II ended
  in"* → *"1945"* and *"The first person to walk on the Moon was"* → *"Neil Armstrong"* (each 3 of 3 seeds; the
  Moon landing's *"1969"* on 2 of 3), *"The capital of France is"* → *"Paris"* (2 of 3), *"Photosynthesis is the
  process by which"* → *"plants convert light energy into chemical energy"* (2 of 3), *"The heart pumps"* →
  *"blood through the body to supply oxygen and nutrients to all cells"*. Einstein is *"born in 1879"* on 2 of 3
  seeds and *"a German-born physicist"* on the third.
- ✅ Occasionally writes correct short code: one seed of *"def is_prime(n):"* produced the standard trial-division
  loop up to `int(n**0.5) + 1`. It knows the shape of common functions (docstrings, base cases, recursion) even
  when the logic is wrong (below).
- ✅ The best benchmark scores in the family (zero-shot): **SciQ 72.2%** (acc_norm; 78.8% acc), **ARC-Easy 47.1%**
  (53.2% acc), **HellaSwag 36.0%**, **PIQA 64.7%**, **ARC-Challenge 26.3%**, **ArithMark-3 37.8%**, up from
  68.6 / 45.2 / 32.8 / 62.0 / 25.4 / 36.2 for `moderate-wide-sparse`. On the Open SLM Leaderboard's Intelligence
  Index that's **~19.4**, against ~15.9 for moderate.
- ✅ Above chance on MMLU in the cloze format: **27.8%** overall, ~30% on the social sciences and "other"
  categories. STEM (25.9%) and the humanities (26.1%) are still at about chance, and the multiple-choice letter
  format is at chance too (24.5%).

**What it can't do**

- ❌ **Less common facts.** *"The chemical symbol for gold is"* → *"Cu"* or *"Pt"*; *"The largest planet in the
  solar system is"* → *"Ganymede"*, *"the Earth"* or a comparison with Mercury; *"Water boils at"* → *"100°C"* on
  one seed, *"100 degrees Fahrenheit"* and *"103.6 degrees Fahrenheit"* on the others. Around a right answer the
  details are made up: Einstein *"born in Budapest, Hungary"* or *"in St. Petersburg, Russia"*, France *"ruled by
  King Louis XVI… a monarchy under Napoleon III"*. Right and wrong answers come with the same confidence.
- ❌ **Arithmetic beyond the very simplest.** *"2 + 2 ="* → *"4"* on 2 of 3 seeds (*"5"* on the third), but
  *"7 + 6 ="* → *"14"*, *"15"* or a multiple-choice question whose marked answer is *"45"*.
- ❌ **Working code, most of the time.** Its Fibonacci is wrong on every seed (*"return 2 * fibonacci(n - 1) +
  fibonacci(n - 2)"*, or the triangle-number formula), and two of three `is_prime` completions are just
  *"return n % 2 == 0"*.
- ❌ **Instructions or questions.** It's a base model: it continues text. Asked *"What is the largest planet in the
  solar system?"* in a `User:` / `Assistant:` exchange, it repeats the question, writes a quiz page, or continues
  a dialogue that names Mercury or Pluto.
- ❌ **Long-range coherence.** A paragraph holds together; beyond that it drifts and repeats.

**Decoding**

Greedy decoding still loops, though later than the smaller models (*"The heart is a muscle that pumps blood
throughout the body. It is a muscle that is made up of two parts… The heart is a muscle that is made up"*). Use
sampling with **temperature ≈ 0.7, top-p ≈ 0.9 and a repetition penalty of about 1.15**.
