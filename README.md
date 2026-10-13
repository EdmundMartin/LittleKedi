<p align="center"><img src="little_kedi.jpg" alt="LittleKedi logo" width="640"></p>

# LittleKedi

Small DeepSeek-style MoE language models in plain PyTorch: v2 of `DeepseekStyleMOE`, built from the
lessons in its `docs/architecture-v2.md`.

| Component | Where | What it does |
|---|---|---|
| Multi-head Latent Attention | `littlekedi/mla.py` | Caches only a low-rank latent `c_KV` and one shared RoPE key per token; weight-absorbed cached decoding. |
| DeepSeekMoE | `littlekedi/moe.py` | Shared + fine-grained routed experts, sigmoid gating, group-limited routing, aux-loss-free bias balancing, a small sequence-wise balance loss. |
| Batched expert dispatch | `littlekedi/moe.py` | Stacked expert weights; capacity or grouped-GEMM dispatch with no host syncs, so the training forward compiles to one graph. |
| Muon + WSD | `littlekedi/optim.py` | Muon for hidden matrices (experts orthogonalized per expert), AdamW for embeddings, norms and the router; warmup-stable-decay schedule. |
| Tokenizer | `littlekedi/tokenizer.py` | Byte-level BPE trained on your data, digits split individually, chat tokens built in. |

## What changed from v1

- **No Multi-Token Prediction.** It cost ~4M parameters and compute per step at small-sparse for no measured benefit.
- **No FP8 simulation.** Nothing at these sizes used it.
- **Tied embeddings** by default (`tie_embeddings=False` to untie). Saves `vocab × dim` parameters (2.1M at small).
- **Stacked experts + batched dispatch** (`moe_backend`):
  - `capacity` (default): GShard-style fixed capacity, `[E, C, D]` buffers, 3 batched matmuls. Static shapes; top-1 choices are placed before top-2 and so on; overflow is dropped and logged as `drop_rate`. `capacity_factor` defaults to 2.0.
  - `grouped_mm`: `torch._grouped_mm` over expert-sorted tokens. Dropless and sync-free. Uses a hand-written backward because the built-in one fails on ROCm 10 / gfx1150.
  - `sorted`: v1's per-expert loop, kept as the exact reference for tests.
  - Evaluation and generation always run dropless.
- **A pure forward.** Metrics come back as tensors and `.item()` runs once per log step. MoE load comes back as `expert_load` and the bias update happens outside the forward.
- **Muon + WSD** by default (`--optimizer adamw`, `--schedule cosine` for v1 behaviour). With WSD, a run still in its constant phase can be resumed with a larger `--steps` to train longer.
- **Parameter reporting**: `total`, `activated`, `embedding`, `activated_non_embedding`, `fraction_activated`. Compare models on activated non-embedding.
- **Tokenizer**: digits split individually; `<|system|> <|user|> <|assistant|> <|end|>` exist from the start; vocab is 16K from `moderate` up.
- **Data**: validation documents are held out per source before any training document is drawn (`val.bin` plus `val_<source>.bin`), and every eval reports each set.
- **Training ops**: `<out>.log.jsonl` appends across resumes; milestone checkpoints `<out>-<N>M.pt` every `--milestone-every` tokens (default 250M); eval reports expert usage (`val_dead_experts`, `val_min_load_ratio`).

## Presets: three variants per size

Each size has `<size>`, `<size>-sparse` and `<size>-wide-sparse`, which differ in sparsity. In the current presets, sparse uses 5× the experts at half width with 2× top-k, and wide-sparse doubles the sparse expert count again; at the moment all three have about the same active parameters. The variants can be scaled independently: nothing in the code requires them to match.

| Preset | Vocab | Total | Activated | Activated non-emb. | Activated % | Experts | Tokens/expert at 1B |
|---|---|---|---|---|---|---|---|
| tiny | 8K | 2.0M | 1.5M | 0.5M | 78% | 8 × 64, top-2 | 250M |
| tiny-sparse | 8K | 2.6M | 1.6M | 0.5M | 60% | 32 × 32, top-4 | 125M |
| tiny-wide-sparse | 8K | 3.8M | 1.6M | 0.5M | 42% | 64 × 32, top-4 | 62M |
| mini | 8K | 9.4M | 4.8M | 3.2M | 51% | 12 × 128, top-3 | 250M |
| mini-sparse | 8K | 19.8M | 4.8M | 3.2M | 24% | 64 × 64, top-6 | 94M |
| mini-wide-sparse | 8K | 36.4M | 4.9M | 3.3M | 13% | 128 × 64, top-6 | 47M |
| small | 8K | 12.2M | 6.3M | 4.2M | 52% | 16 × 128, top-4 | 250M |
| small-sparse | 8K | 24.1M | 6.4M | 4.3M | 27% | 80 × 64, top-8 | 100M |
| small-wide-sparse | 8K | 43.9M | 6.5M | 4.4M | 15% | 160 × 64, top-8 | 50M |
| moderate | 16K | 49.8M | 18.8M | 12.5M | 38% | 24 × 192, top-4 | 167M |
| moderate-sparse | 16K | 105.8M | 19.1M | 12.8M | 18% | 120 × 96, top-8 | 67M |
| moderate-wide-sparse | 16K | 199.0M | 19.4M | 13.1M | 10% | 240 × 96, top-8 | 33M |
| medium | 16K | 85.9M | 30.8M | 22.4M | 36% | 24 × 256, top-4 | 167M |
| medium-sparse | 16K | 185.3M | 31.2M | 22.8M | 17% | 120 × 128, top-8 | 67M |
| medium-wide-sparse | 16K | 350.9M | 31.6M | 23.2M | 9% | 240 × 128, top-8 | 33M |
| base | 32K | 258.7M | 105.4M | 80.2M | 41% | 32 × 256, top-6 | 188M |
| base-sparse | 32K | 542.8M | 106.3M | 81.2M | 20% | 160 × 128, top-12 | 75M |
| base-wide-sparse | 32K | 1015.9M | 107.6M | 82.4M | 11% | 320 × 128, top-12 | 38M |

Tokens per expert is per MoE layer, assuming balanced routing. Below ~25–50M, experts are probably undertrained. `train.py` prints the figure for the planned run and warns below 25M. Wide-sparse needs a bigger data budget than its sparse twin.

### Speed (Radeon 890M iGPU, ROCm 10, bf16, 8 × 1,024 tokens/step, Muon)

| Preset | v1 per-expert loop (eager) | capacity (eager) | capacity (compiled) | grouped_mm (compiled) |
|---|---|---|---|---|
| small | 3.9K tok/s | 3.9K | 9.0K | — |
| small-sparse | 2.0K | 3.7K | 7.2K | 7.5K |
| small-wide-sparse | crashes (HIP fault) | 2.6K | 5.9K | — |

Eager numbers on the iGPU vary by about ±15% from run to run. Compiling takes 1–2 minutes the first time. A 60-step `train.py --compile` run of small-sparse at 16K tokens/step sustained ~7.9K tok/s.

## Setup

Python 3.12 (the ROCm Windows wheels target it). On Windows with an AMD GPU, install the ROCm build of PyTorch first. The example is for gfx1150 (Radeon 890M):

```powershell
py -V:3.12 -m venv .venv
.venv\Scripts\python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "torch[device-gfx1150]==2.13.0+rocm10.0.0"
.venv\Scripts\python -m pip install numpy "tokenizers>=0.23.2" pyarrow "huggingface-hub>=1.33.0" pytest
```

Elsewhere: `uv venv --python 3.12 .venv && uv sync`.

## Usage

```sh
python prepare_data.py --preset small-sparse --target-tokens 1e9 \
    --mix fineweb-edu=0.6 cosmopedia=0.2 code=0.1 smoltalk=0.1 --out-dir data/mix-small-1b
python train.py --data-dir data/mix-small-1b --preset small-sparse --batch-size 8 --grad-accum 8 \
    --compile --out checkpoints/small-sparse.pt                 # add --resume to continue
python sample.py --ckpt checkpoints/small-sparse.pt
python -m pytest
```

The three variants of a size share a vocab, so one dataset serves all three. That makes the
small vs small-sparse vs small-wide-sparse ablation a matter of changing `--preset`. Override any config field with
`--set key=value`, e.g. `--set moe_backend=grouped_mm capacity_factor=1.5`.

## Evaluation and model cards

```sh
python evals.py --ckpt checkpoints/tiny.pt --log evals.jsonl                                  # SciQ, ARC-Easy, HellaSwag
python mmlu_eval.py --ckpt checkpoints/tiny.pt --log mmlu_results.jsonl                       # MMLU letter + cloze
python benchmark.py --ckpt checkpoints/tiny.pt --log benchmarks.jsonl                         # factual probes + HTML report (not on the card)
python prepare_release.py --ckpt checkpoints/tiny.pt --capabilities docs/tiny-capabilities.md # release/LittleKedi-tiny-base/
```

`model_card.py` is one card template for every size and variant. The family section and its table of all
presets are shared, with the current model highlighted. Everything else is filled in from the checkpoint, its
`.log.jsonl`, the dataset's `meta.json` and the eval logs for that checkpoint's step: architecture, training
recipe, validation loss, benchmarks, MMLU, and a comparison with its sibling variants (when
`checkpoints/<size>-sparse.pt` and so on exist) and published small models. `--card-only` writes just the
card next to the checkpoint. Evaluate on the GPU only while nothing is training, or pass `--device cpu`.

## Chat fine-tuning (SFT) and GGUF

```sh
python prepare_sft_data.py --tokenizer data/mix-small-1b/tokenizer.json --max-len 1024 --limit 6000   # smol-smoltalk, filtered
python gen_sft_data.py --base-url http://localhost:11434/v1 --model <teacher> --data-dir data/mix-small-1b --docs 3000
python sft.py --init checkpoints/tiny.pt --data data/sft/smol_smoltalk_filtered.jsonl data/mix-small-1b/sft_qa.jsonl
python chat.py --ckpt checkpoints/tiny-sft.pt
python export_gguf.py --ckpt checkpoints/tiny-sft.pt --dtype q8_0
python prepare_release.py --ckpt checkpoints/tiny-sft.pt --repo <user>/LittleKedi-tiny-instruct --base-repo <user>/LittleKedi-tiny-base
```

- `sft.py` trains only on assistant replies and `<|end|>`. It uses Muon by default, keeps the routing biases frozen,
  and runs the experts dropless. v2 tokenizers already have the chat tokens; for a v1-tokenizer model they're
  added and the embedding grows. The output checkpoint is marked as a chat model and records its base
  checkpoint, SFT data and validation loss.
- `prepare_release.py` reads base vs instruct from the checkpoint. **Base** releases (`LittleKedi-<preset>-base`)
  are PyTorch weights only. **Instruct** releases (`LittleKedi-<preset>-instruct`) also get GGUF (f16 + q8_0),
  `chat_template.jinja`, `tokenizer_config.json` and Ollama `template` / `params`. Their card adds the chat format,
  an SFT table and a base-vs-instruct comparison (evaluate both checkpoints first).
- `export_gguf.py` writes llama.cpp's `deepseek2` layout: stacked experts transposed to `ffn_*_exps`, tied
  embeddings written as both `token_embd` and `output`. It checks that llama.cpp's `gpt-2` pre-tokenizer splits
  text the same way as our digit-splitting tokenizer. v1's exporter was checked against llama.cpp b11203; this one hasn't
  been run through llama.cpp yet.

## Training on RunPod

`cloud.py` launches training runs on RunPod from this repo:
- **code:** each pod clones GitHub at your current commit, so commit and push first;
- **data:** a private Hugging Face dataset;
- **outputs:** checkpoints and logs sync to a private Hugging Face repo (`edededdy/littlekedi-runs`) every 30 minutes;
- **clean-up:** the pod deletes itself when training ends.

If a pod dies or hits `--max-hours`, launch the same run again and it resumes from the last sync.

Credentials go in `.env` (gitignored; copy `.env.example`) or environment variables, never on the command line:
- `RUNPOD_API_KEY`: your RunPod API key.
- `LK_HF_TOKEN`: a fine-grained Hugging Face token that can read the dataset repo and write to the runs repo. It's stored
  in the pod's environment on RunPod, so don't use your main token.

```sh
python cloud.py push-data data/mix-8k-4b --repo edededdy/littlekedi-data-mix-8k-4b     # once per dataset
python cloud.py launch mini-wide-sparse --data-repo edededdy/littlekedi-data-mix-8k-4b --data-name mix-8k-4b     --train-args "--preset mini-wide-sparse --max-train-tokens 3.2e9 --batch-size 16 --grad-accum 4 --compile"
python cloud.py list                       # your pods, status and $/hr
python cloud.py logs mini-wide-sparse      # tail of the synced training log
python cloud.py pull mini-wide-sparse      # checkpoints, milestones and logs -> checkpoints/
python cloud.py terminate <pod-id>
```

`--dry-run` prints the pod request without launching. Other options: `--gpu` (RunPod GPU type ids, in priority order;
default RTX 4090), `--cloud SECURE`, `--spot` (interruptible: cheaper, and it resumes from the last sync if killed),
`--max-hours`, `--keep-pod`.

## v1 checkpoints

```sh
python convert_v1.py --ckpt ../DeepseekStyleMOE/checkpoints/small-sparse.pt --out checkpoints/v1-small-sparse.pt
```

The converter drops MTP, stacks the experts and keeps v1's untied head (`tie_embeddings=False`). The converted v1
small-sparse checkpoint gives logits identical to v1's code. The optimizer state isn't carried over, so
`train.py --resume` on a converted checkpoint starts a fresh optimizer. `export_qa_dataset.py` (publishing v1's
grounded Q&A set) wasn't ported.
