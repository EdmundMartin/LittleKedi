~24,000 conversations (2.25M assistant tokens in training), built for medium's 2,048-token context: every part
was filtered to conversations that fit in 2,048 tokens (the smaller models' sets stopped at 1,024), and the
generated answers are longer. The grounded Q&A and code exercises were written by **DeepSeek V4.1 Flash** (via
OpenRouter, reasoning off), a much stronger teacher than the Gemma 4 E4B used for the smaller models, for a few dollars
in total.

| Part | Conversations | Source |
|---|---|---|
| Grounded Q&A | 8,656 | Questions and answers written by DeepSeek V4.1 Flash from 3,000 FineWeb-Edu documents that were in medium's pretraining data; answers of 2–6 sentences that stand on their own without the document |
| **Wikipedia Q&A** | 5,859 | The same, from 2,000 of the earliest English Wikipedia articles in medium's pretraining data. Early articles were chosen as a proxy for well-known topics: Q&A about facts a model never learned mostly teaches it to make things up |
| General chat | 3,553 | `HuggingFaceTB/smol-smoltalk`, filtered to conversations that fit in 2,048 tokens |
| Easy code | 2,588 | Python exercises written by DeepSeek V4.1 Flash from Python files in the pretraining data; every solution passes its own asserts *and* an independent set written from the task alone |
| Instructions | 1,200 | `databricks/databricks-dolly-15k` (QA, classification, brainstorming, extraction) |
| Conversations | 800 | `OpenAssistant/oasst2`, English, the top-ranked reply at each turn |
| Arithmetic | 700 | Templated single- and two-digit sums, differences and products |
| Code problems | 473 | MBPP train/validation/prompt splits (solutions verified against their tests) |
| Persona | 213 | Short conversations establishing LittleKedi, a small cat from Istanbul who has read a lot |

Held out: 5% of each part (by source document for the generated Q&A and code), plus MBPP's 500 test problems,
300 arithmetic questions and 25 persona questions.

Training: one epoch, batch size 8, Muon with a peak learning rate of **5e-5** (cosine to 5e-6), dropless experts,
routing biases frozen, on one RTX 4090 (RunPod) in about 20 minutes. A first run at the smaller models' 2e-4 made
held-out loss *worse* (1.973 → 2.176): medium got an unusually large share of its quality from its pretraining
learning-rate decay, and re-warming to 10% of the peak undid it. At 5e-5 held-out loss fell to **1.859**.
