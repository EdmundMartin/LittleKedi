~10,100 conversations (1.05M assistant tokens), shared by every mini and small model, since they were
pretrained on the same data with the same tokenizer:

| Part | Conversations | Source |
|---|---|---|
| Grounded Q&A | 3,705 | Questions and answers written by Gemma 4 E4B from documents these models saw in pretraining (FineWeb-Edu, Cosmopedia Stanford/OpenStax/WikiHow, FineMath), cleaned of pairs that refer back to "the document" |
| General chat | 2,339 | `HuggingFaceTB/smol-smoltalk`, filtered to conversations that fit in 1,024 tokens |
| Instructions | 1,140 | `databricks/databricks-dolly-15k` (QA, classification, brainstorming, extraction) |
| Easy code | 787 | Python exercises written by Gemma from Python files in the pretraining data; every solution passes its own asserts *and* an independent set written from the task alone |
| Conversations | 760 | `OpenAssistant/oasst2`, English, the top-ranked reply at each turn |
| Arithmetic | 700 | Templated single- and two-digit sums, differences and products |
| Code problems | 473 | MBPP train/validation/prompt splits (solutions verified against their tests) |
| Persona | 213 | Short conversations establishing LittleKedi, a small cat from Istanbul who has read a lot |

Held out: 5% of each part (by source document for the generated Q&A and code), plus MBPP's 500 test problems,
300 arithmetic questions and 25 persona questions. One epoch over the mix was about as good as two
(validation loss was lowest at the end of the first epoch).
