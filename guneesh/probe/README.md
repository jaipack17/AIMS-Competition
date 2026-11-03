# probe/: where does Qwen3-VL-8B represent item difficulty?

Goal: find a layer, token position or prompt whose activations rank items by IRT difficulty b **on unseen
benchmarks** better than the current last-layer, last-token embedding (held-out within-benchmark r: ridge 0.133,
MLP 0.158; see `EXPERIMENTS.md`). Ideas from Lee et al. 2025 (arXiv 2510.05969, linear difficulty probe and
final-layer "difficulty heads") and difference-of-means probes, plus a logit-lens readout of the model's own rating.

| File | What it does |
|---|---|
| `extract_activations.py` | Step 1 (GPU). Runs the model on a sample of items and saves hidden states (alternate layers by default) at 3 token positions, the logit-lens readout and a parity check. Full description in its docstring. |
| `analyze_probes.py` | Step 2 (CPU). Probe sweep over keys × layers (diff-of-means, ridge), logit-lens rating, length baseline, direction cosines; leave-one-fold-out by benchmark. |
| `compare_production.py` | Step 3 (CPU). Rebuilds the production ridge (all embedded items, leave-one-fold-out) and compares it per benchmark with chosen pilot probes on the same items. |
| `extract_feature.py` | Step 4 (GPU). The chosen feature (default: `rate` prompt, layer 20, last token) for **every** embedded item, as a drop-in replacement for `data/emb/qwen3vl8b/`. Uses `submission_src/common/probe_feature.py`, the same code the submission runs. |
| `run_probe_submission.sh` | Step 4 end to end in the background: extraction, ridge, stretch, submission zip, local ALC vs the ridge and MLP versions. |
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

## Step 3: compare with the production ridge (CPU)

```bash
python probe/compare_production.py probe/acts/pilot_l5      # after analyze_probes.py; ~8 GB RAM, ~10-30 min
```

The pilot probes are trained on ~2,400 items, so `d_vs_ref` in step 2 compares them with a ridge on the same small
set. This script rebuilds the real production ridge from `data/emb/qwen3vl8b` (every embedded item of the training
folds, standardised ridge, alpha by inner leave-one-fold-out) and gives the paired difference on the same benchmarks
and items. Its overall within-benchmark r on all benchmarks should be close to 0.133 (the number in EXPERIMENTS.md);
if not, the rebuild differs from the original and the comparison is not trustworthy. Tested: the ridge equals
sklearn's `Ridge` to 1e-13, and on fake data with a planted direction it reaches r 0.892 (oracle 0.898).

## Full-run results (115 benchmarks x up to 300 items, layers 18-30 even + 36)

* The pilot's best (`ap__item_last` L24 ridge_wc, r 0.234 on 30 benchmarks) **did not replicate**: 0.158, +0.008
  [-0.027, +0.040] vs the reference. It was the top of ~60 configurations on 30 benchmarks (winner's curse).
* Best: **`rate__last` at layers 20-22** (last token of the 0-9 rating prompt, middle of the network):
  ridge_wc L20 0.196 [0.161, 0.232]; vs the reference ridge (ap__last L36, same items) +0.045 [+0.011, +0.084];
  partial r after length 0.184.
* vs the rebuilt **production ridge** (all ~230k embedded items, `compare_production.py`; 0.142 on all items, 0.144 on
  the full-run items): rate__last L20 ridge_wc +0.056 [+0.015, +0.104] (better on 69/110), plain ridge +0.048
  [+0.004, +0.099], L22 ridge_wc +0.058 [+0.021, +0.103]; ap__item_last L24 +0.011 [-0.023, +0.043].
* The model's own rating (logit lens of `rate`, no training) gives 0.153; the `ap` prompt's lens 0.054 (digit mass 0).

## Step 4: submission with the probe feature (GPU, then CPU)

```bash
bash probe/run_probe_submission.sh            # LAYER=22 ... for another layer; GPU=1 ... to pick the GPU
tail -f probe/logs/probe_sub_L20_*.log
```

It runs, resuming whatever is already done: the Hugging Face download of the IRT fits' `items.parquet` and
`cells_binary.parquet`; `extract_feature.py` (the long step: every embedded item once, rate prompt, the forward pass
stops after layer 20); `ridge_train.py --emb_dir ... --runs versionB_4parts_logitnormal` and `stretch_factor.py` on
the new features; `build_submission.py ... --embed_spec <emb>/spec.json`, which bundles `embed_spec.json` and
`probe_feature.py` so `core.py` computes the same feature at test time; the kit checks (if `Starting Kit/` is there);
`simulate_alc.py` (150 draws, `+date`) and `analysis/paired_alc.py` against `regress_label_update_stretch2_date`
(0.2273) and `label_update_stretch2_date_mlp` (0.2234) on the same draws. Result:
`submissions/label_update_stretch2_date_probe_L20.zip`. Without `embed_spec.json`, `core.py` behaves exactly as before.

`extract_feature.py` prints, per benchmark, the relative difference vs the vectors saved by the full run
(`probe/acts/full_l8`): batched bf16 at layer 20 gives a median of about 1.5-2.5e-2 and a max of about 0.1-0.3.

**Massive activation at the last token.** About one item in a few hundred has a max of 1-66: on the real run, the
same tokens (equal token counts) give a vector dominated by dimension 2276 (400-6000, normal vectors have norm ~90
and |x| <= ~60) in one run and not in the other, in either direction. So whether the spike fires flips with bf16
rounding; it would flip at test time too. After standardisation a spike is dozens of SDs, and the stretch (k ~ 4)
multiplies the resulting b_hat error. `CLIP=1 bash probe/run_probe_submission.sh` (after the unclipped run, it
reuses the extracted features) builds `..._probe_L20_clip`: `probe/feature_clip.py` computes per-dimension bounds
(median +- 8 robust SDs, no labels) into `<emb>/clip.json`; `ridge_train.py` / `stretch_factor.py --clip` train on
clipped features and `build_submission.py --clip` bundles `feature_clip.json`, which `core.py` applies before the
ridge. Its `paired_alc` table also compares it with the unclipped version.

Tested on a tiny random Qwen3-VL (transformers 5.22): the early-stopped layer equals `hidden_states[L]` exactly;
batched (right padding) vs one at a time 7e-9; the feature equals extract_activations' `rate__last`; `core.py`
with `embed_spec.json` gives the same feature (7e-9) and b_hat as the training extraction; the shipped
`label_update_stretch2_date` still imports and predicts with the new `core.py`.

## Tested

CPU smoke test with `Qwen/Qwen3-0.6B` (same tokenizer family and model layout) on a small fake dataset with
real reference embeddings. Results: parity 0.0; item positions identical under another prompt (0.0); the
last-layer logit lens equals the model's real next-token probabilities (7e-9); the `rate` digit mass is 0.99;
resuming, the guard against mixing runs, and skipping of items with no tokens left after truncation all work.
On a GPU with Qwen3-VL-8B (pilot, 30 benchmarks): parity 1 (bf16 rounding, see above), `rate` digit mass 0.999–1.0
on every benchmark, about 6 s per 100 items. Also tested end to end with a tiny random Qwen3-VL in transformers 5.22.
