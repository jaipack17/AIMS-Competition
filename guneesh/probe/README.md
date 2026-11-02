# probe/: where does Qwen3-VL-8B represent item difficulty?

Goal: find a layer, token position or prompt whose activations rank items by IRT difficulty b **on unseen
benchmarks** better than the current last-layer, last-token embedding (held-out within-benchmark r: ridge 0.133,
MLP 0.158; see `EXPERIMENTS.md`). Ideas from Lee et al. 2025 (arXiv 2510.05969, linear difficulty probe and
final-layer "difficulty heads") and difference-of-means probes, plus a logit-lens readout of the model's own rating.

| File | What it does |
|---|---|
| `extract_activations.py` | Step 1 (GPU). Runs the model on a sample of items and saves hidden states (alternate layers by default) at 3 token positions, the logit-lens readout and a parity check. Full description in its docstring. |
| `analyze_probes.py` | Step 2 (CPU). Probe sweep over keys × layers (diff-of-means, ridge), logit-lens rating, length baseline, direction cosines; leave-one-fold-out by benchmark. |
| `acts_io.py` | Loads the saved files (`load_layer`, `load_all_layers`, `load_lens`, `consolidate`). |
| `run_extract.sh` | Starts the extraction with `nohup` so it survives a dropped SSH connection. |
| `requirements.txt` | Python packages for the GPU machine. |

## What one run saves

* **Prompts** (all start with `Task statement:\n<item>\n\n`):
  * `ap`: the Agent Psychometrics difficulty prompt, identical to the text of the existing embeddings.
  * `rate`: "How difficult is the above task for an AI model? Answer with a single digit from 0 (very easy) to 9
    (very hard).\nDifficulty: " (the next token is a digit, so the logit lens reads the model's own rating).
  * `solve`: "Solve the above task. Reply with only the final answer.\nAnswer:" (the state just before answering).
  * `rate_rev` (optional, `--prompts ap,rate,solve,rate_rev`): the rating with the scale reversed, a robustness check.
* **Positions:** `last` (last token, every prompt); `item_last` and `item_mean` (last token / mean of the item
  text, before the instruction). Because the model is causal, the item positions are identical under every prompt
  (checked: difference 0.0), so they are the "no prompt" state and are saved once, from `ap`.
* **Layers** (`--layers`): default `even` = 0, 2, …, 36 (19 of the 37 outputs). Also `all`, `every:4`, or a list
  such as `0,12,20,24,28,32,36`. Index 0 = token embeddings, 36 = last layer after the final norm (the vector of
  the existing embeddings; it is always saved, and the run checks it against `data/emb/qwen3vl8b/`).
  `acts_io.load_layer(..., layer=20)` takes the real layer number.
* **Logit lens** at `last`, on **every** layer whatever `--layers` says (it is tiny): P(digit 0–9), total digit
  mass, entropy; the last layer's top-5 tokens.
* **Sample:** `--mode pilot` = 6 benchmarks per fold (30) × 100 items; `--mode full` = all 115 × 300.
  Pilot items are a subset of the full-run items. Target = b of the final IRT fit; the fold of each benchmark is
  stored so the analysis can hold benchmarks out exactly as in the rest of the project.

Size (3 prompts = 5 saved vectors per item and layer, 8 KB each) and time on one A100/H100 (about 0.2 s per
forward pass; `--layers` does not change the time):

| `--layers` | layers | per item | pilot (≈ 3,000 items, ≈ 1 h) | full (≈ 34,500 items, ≈ 6–8 h) |
|---|---|---|---|---|
| `all` | 37 | 1.5 MB | 4.5 GB | 52 GB |
| **`even` (default)** | 19 | 0.78 MB | **2.3 GB** | **27 GB** |
| `every:4` | 10 | 0.41 MB | 1.2 GB | 14 GB |

Suggested: run the pilot with `even`, then run the full set only for the layers and prompts that win in the
pilot (for example `--layers 16,20,24,28,36 --prompts ap,rate`: about 7 GB).

## Running it on a GPU machine over SSH

```bash
# 0. on the server: get the code (branch Guneesh) and go to the project root
git clone -b Guneesh https://github.com/jaipack17/AIMS-Competition.git && cd AIMS-Competition/guneesh

# 1. environment (CUDA build of torch; pick the index URL that matches the server's CUDA)
python3 -m venv .venv && source .venv/bin/activate
pip install -r probe/requirements.txt

# 2. data: IRT difficulties + the existing embeddings (for the parity check) from Hugging Face (public).
#    One --include per pattern (with several patterns after one --include, the hf CLI ignores the filter).
#    data/irt/folds.csv is already in git.
hf download GunGG4/paiec-Qwen8Binstruct-guneesh-data --repo-type dataset --local-dir data \
    --include "irt/versionB_4parts_logitnormal/final/*" --include "emb/qwen3vl8b/*"
ls data/irt/versionB_4parts_logitnormal/final/items.parquet     # must exist
#    the item texts (data/measurement-db/<benchmark>/formatted_tables/items.parquet) are downloaded
#    automatically for the sampled benchmarks; --no_fetch turns that off.
#    aims-foundations/measurement-db is GATED: accept its terms on huggingface.co with your account, make a
#    Read token (huggingface.co/settings/tokens) and log in on the server once (never commit the token):
hf auth login
hf auth whoami

# 3. model cache in a folder you can write to with ~20 GB free (the first run downloads Qwen3-VL-8B-Instruct)
export HF_HOME=~/hf_cache            # a real path; check free space with  df -h ~
echo 'export HF_HOME=~/hf_cache' >> ~/.bashrc

# 4. check prompts and token positions without the GPU (tokenizer only)
python probe/extract_activations.py --dry_run

# 5. start the pilot in the background, then you can log out
bash probe/run_extract.sh pilot            # GPU=1 bash probe/run_extract.sh pilot  to pick another GPU
tail -f probe/logs/extract_pilot_*.log     # Ctrl-C stops watching, not the job
nvidia-smi                                 # GPU use
```

Each finished benchmark prints one line, for example
`benchX: 100 items (0 skipped) in 6s | parity 1 | digit mass ap=0.01 rate=0.99 solve=0.0 | 0.42 h, ETA 0.55 h`.
Check:

* **parity** ≤ about 2. It is the largest absolute difference over 4096 numbers, and the embeddings have a few values
  near 95 where bf16 steps are 0.5, so 1 (two rounding steps from a different GPU/kernel/batching) is normal. The
  first Qwen3-VL-8B pilot gives 1. A wrong token or text gives 5 or more: stop and investigate.
* **digit mass rate** close to 1, meaning the model really answers with a digit. If it is low, the rating
  readout is meaningless for that benchmark.

**Resuming:** rerun the same command. Finished benchmarks are skipped, and the item sample is read back from
`sample.parquet`. Changing the prompts on an existing output folder is refused; use `--out_dir`.
`--max_hours H` stops starting new benchmarks after H hours.

**Copying results back:** `rsync -avP user@server:AIMS-Competition/guneesh/probe/acts/pilot/ probe/acts/pilot/`.
`probe/acts/` and `probe/logs/` are large outputs and are not meant to be committed.

## Step 2: analyse (CPU, after the pilot)

```bash
python probe/analyze_probes.py probe/acts/pilot_l5        # on the server (no GPU needed) or locally after rsync
# options: --keys ap__last,rate__last  --layers 20,36  --methods dom,ridge,ridge_wc  --out <folder>
```

`analyze_probes.py` (docstring has the details) does leave-one-fold-out over the project's benchmark folds and,
for every saved key × layer, fits a difference-of-means probe (`dom`), a ridge (`ridge`, as `ridge_train.py`) and a
ridge on benchmark-centred data (`ridge_wc`). It scores each with the held-out within-benchmark r (bootstrap 95%
interval, per fold, partial r after removing token length, and paired difference vs the reference = ridge on
`ap__last` at the last layer trained on the same items). It also reports, with no training: the logit-lens rating
r per prompt and layer plus the digit mass, the length baseline, and whether per-benchmark difficulty directions
are shared (mean pairwise cosine vs b shuffled). Results go to `probe/results/<run>/`: `probe_sweep.csv`,
`per_benchmark.csv`, `lens.csv`, `direction_cosine.csv`, `layer_curves.png`.

How to read it: a layer/key/prompt is worth a full run only if `d_vs_ref` has `d_ci_lo > 0` (it beats the
existing vector on unseen benchmarks), `partial_r_len` stays close to `within_r` (it is not just length), and
`mean_cos` is clearly above `mean_cos_shuffled` (a shared difficulty axis exists). With 30 benchmarks the intervals
are wide; differences under ~0.03 are noise.

Tested on a fake run with a planted difficulty direction (15 benchmarks, hidden 512): the planted layer, position
and logit-lens rating come out on top (r 0.35–0.51), and keys and layers with nothing planted come out at ≈ 0.
Expected time on the real pilot (5 layers, 4 keys, hidden 4096): about 10–20 min on one CPU.

## Tested

CPU smoke test with `Qwen/Qwen3-0.6B` (same tokenizer family and model layout) on a small fake dataset with
real reference embeddings. Results: parity 0.0; item positions identical under another prompt (0.0); the
last-layer logit lens equals the model's real next-token probabilities (7e-9); the `rate` digit mass is 0.99;
resuming, the guard against mixing runs, and skipping of items with no tokens left after truncation all work.
On a GPU with Qwen3-VL-8B (pilot, 30 benchmarks): parity 1 (bf16 rounding, see above), `rate` digit mass 0.999–1.0
on every benchmark, about 6 s per 100 items. Also tested end to end with a tiny random Qwen3-VL in transformers 5.22.
