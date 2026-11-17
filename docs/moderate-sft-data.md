~12,000 conversations (1.07M assistant tokens): the mini/small SFT set plus Wikipedia Q&A grounded in
moderate's own pretraining data.

| Part | Conversations | Source |
|---|---|---|
| Grounded Q&A | 3,705 | Questions and answers written by Gemma 4 E4B from FineWeb-Edu, Cosmopedia (Stanford/OpenStax/WikiHow) and FineMath documents, cleaned of pairs that refer back to "the document" |
| **Wikipedia Q&A** | 1,910 | Written by Gemma 4 E4B from long (4,000+ token) English Wikipedia articles that were in moderate's pretraining data. Long articles were chosen as a proxy for well-known topics: Q&A about facts a model never learned mostly teaches it to make things up |
| General chat | 2,339 | `HuggingFaceTB/smol-smoltalk`, filtered to conversations that fit in 1,024 tokens |
| Instructions | 1,140 | `databricks/databricks-dolly-15k` (QA, classification, brainstorming, extraction) |
| Easy code | 787 | Python exercises written by Gemma from Python files in the pretraining data; every solution passes its own asserts *and* an independent set written from the task alone |
| Conversations | 760 | `OpenAssistant/oasst2`, English, the top-ranked reply at each turn |
| Arithmetic | 700 | Templated single- and two-digit sums, differences and products |
| Code problems | 473 | MBPP train/validation/prompt splits (solutions verified against their tests) |
| Persona | 213 | Short conversations establishing LittleKedi, a small cat from Istanbul who has read a lot |

Held out: 5% of each part (by source document for the generated Q&A and code), plus MBPP's 500 test problems,
300 arithmetic questions and 25 persona questions. Held-out skill scores: arithmetic 43.7%, MBPP pass@1 1.8%,
persona 64% (greedy decoding).
