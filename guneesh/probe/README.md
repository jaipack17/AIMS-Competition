# probe/: where does Qwen3-VL-8B represent item difficulty?

Goal: find a layer, token position or prompt whose activations rank items by IRT difficulty b **on unseen
benchmarks** better than the current last-layer, last-token embedding (held-out within-benchmark r: ridge 0.133,
MLP 0.158; see `EXPERIMENTS.md`). Ideas from Lee et al. 2025 (arXiv 2510.05969, linear difficulty probe and
final-layer "difficulty heads") and difference-of-means probes, plus a logit-lens readout of the model's own rating.

| File | What it does |
|---|---|
| `extract_activations.py` | Step 1 (GPU). Runs the model on a sample of items and saves hidden states of all 37 layers at 3 token positions, the logit-lens readout and a parity check. Full description in its docstring. |
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
* **Layers:** all 37 outputs. Index 0 = token embeddings, 36 = last layer after the final norm (the vector of the
  existing embeddings; the run checks `ap` layer 36 against `data/emb/qwen3vl8b/` and logs the difference).
* **Logit lens** at `last`, every layer: P(digit 0–9), total digit mass, entropy; the last layer's top-5 tokens.
* **Sample:** `--mode pilot` = 6 benchmarks per fold (30) × 100 items; `--mode full` = all 115 × 300.
  Pilot items are a subset of the full-run items. Target = b of the final IRT fit; the fold of each benchmark is
  stored so the analysis can hold benchmarks out exactly as in the rest of the project.

Expected size and time on one A100/H100 (3 prompts, about 0.2 s per forward pass): pilot ≈ 3,000 items,
**≈ 4.5 GB, ≈ 1 h**; full ≈ 34,500 items, **≈ 52 GB, ≈ 6–8 h**.

## Running it on a GPU machine over SSH

```bash
# 0. on the server: get the code (branch Guneesh) and go to the project root
git clone -b Guneesh https://github.com/jaipack17/AIMS-Competition.git && cd AIMS-Competition/guneesh

# 1. environment (CUDA build of torch; pick the index URL that matches the server's CUDA)
python3 -m venv .venv && source .venv/bin/activate
pip install -r probe/requirements.txt

# 2. data: IRT difficulties + the existing embeddings (for the parity check) from Hugging Face (public)
hf download GunGG4/paiec-Qwen8Binstruct-guneesh-data --repo-type dataset --local-dir data \
    --include "irt/folds.csv" "irt/versionB_4parts_logitnormal/final/*" "emb/qwen3vl8b/*"
#    the item texts (data/measurement-db/<benchmark>/formatted_tables/items.parquet) are downloaded
#    automatically for the sampled benchmarks; --no_fetch turns that off

# 3. model cache somewhere with ~20 GB free (the first run downloads Qwen3-VL-8B-Instruct)
export HF_HOME=/path/with/space/hf_cache

# 4. check prompts and token positions without the GPU (tokenizer only)
python probe/extract_activations.py --dry_run

# 5. start the pilot in the background, then you can log out
bash probe/run_extract.sh pilot            # GPU=1 bash probe/run_extract.sh pilot  to pick another GPU
tail -f probe/logs/extract_pilot_*.log     # Ctrl-C stops watching, not the job
nvidia-smi                                 # GPU use
```

Each finished benchmark prints one line, for example
`benchX: 100 items (0 skipped) in 61s | parity 0 | digit mass ap=0.01 rate=0.99 solve=0.0 | 0.42 h, ETA 0.55 h`.
Check:

* **parity** ≈ 0 (≤ 1e-2). If not, the activations do not match the existing embeddings: stop and investigate.
* **digit mass rate** close to 1, meaning the model really answers with a digit. If it is low, the rating
  readout is meaningless for that benchmark.

**Resuming:** rerun the same command. Finished benchmarks are skipped, and the item sample is read back from
`sample.parquet`. Changing the prompts on an existing output folder is refused; use `--out_dir`.
`--max_hours H` stops starting new benchmarks after H hours.

**Copying results back:** `rsync -avP user@server:AIMS-Competition/guneesh/probe/acts/pilot/ probe/acts/pilot/`.
`probe/acts/` and `probe/logs/` are large outputs and are not meant to be committed.

## Tested

CPU smoke test with `Qwen/Qwen3-0.6B` (same tokenizer family and model layout) on a small fake dataset with
real reference embeddings. Results: parity 0.0; item positions identical under another prompt (0.0); the
last-layer logit lens equals the model's real next-token probabilities (7e-9); the `rate` digit mass is 0.99;
resuming, the guard against mixing runs, and skipping of items with no tokens left after truncation all work.
It has not yet been run on a GPU with Qwen3-VL-8B. The first pilot log lines (parity, digit mass) are that check.
