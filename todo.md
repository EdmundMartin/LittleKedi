# To do

## Build the 70B-token dataset for `base`

Not started: the first attempts were deleted so the build could be restarted with resume support. The code for this
(`prepare_data.py`, `littlekedi/data.py`, `train.py`, `cloud.py`) is uncommitted.

**Start** (about 40 hours at the last run's speed, ~300GB+ of disk with downloads):

```bash
mkdir -p data/mix-32k-70b
.venv/Scripts/python -u -W ignore prepare_data.py --preset base --target-tokens 70e9 --out-dir data/mix-32k-70b --mix fineweb-edu-100bt=57.94 cosmo-v2=11.99 finemath=14.865 code=10.375 wikipedia=3.70 smoltalk=1.14 --overflow-to code finemath > data/mix-32k-70b/build.log 2> data/mix-32k-70b/build.err
```

**Resume after a crash or reboot:** the same command with `--resume`, and `>>` so the logs aren't overwritten. Don't
change any arguments.

```bash
.venv/Scripts/python -u -W ignore prepare_data.py --preset base --target-tokens 70e9 --out-dir data/mix-32k-70b --mix fineweb-edu-100bt=57.94 cosmo-v2=11.99 finemath=14.865 code=10.375 wikipedia=3.70 smoltalk=1.14 --overflow-to code finemath --resume >> data/mix-32k-70b/build.log 2>> data/mix-32k-70b/build.err
```

- Position is saved every 250M tokens. Killed before the first save: delete `data/mix-32k-70b/` and start again.
- Late in the build, a resume spends a while re-reading sources to skip the documents already used.
- Output is `train-00000.bin` onwards, 40GB parts (Hugging Face stores files of at most 50GB).

**Mix** (no Reddit; 32K vocab):

| Source | Share | Tokens |
|---|---|---|
| FineWeb-Edu | 57.9% | 40.6B |
| FineMath | 14.9% | 10.4B |
| Cosmopedia v2 | 12.0% | 8.4B |
| Code | 10.4% | 7.3B |
| Wikipedia | 3.7% | 2.6B |
| Smoltalk | 1.1% | 0.8B |

Any shortfall goes to code and maths. FineMath may run out (10.4B is about all of its 4+ subset): the log would say
"finemath ran out" and the rest of its share goes to code.

## Then publish it

- [ ] Write the dataset card: every source and its licence (Wikipedia is CC-BY-SA, the GitHub code keeps its per-file
      licences).
- [ ] Upload, about 140GB, public:

```bash
python cloud.py push-data data/mix-32k-70b --repo edededdy/littlekedi-data-mix-32k-70b --public
```
