"""Score the simple baselines with the local harness on a seeded sample of binary benchmarks."""
import sys, random
sys.path.insert(0, ".")
from paiec import data, harness
from baselines import simple

ids = data.binary_benchmark_ids()
random.Random(0).shuffle(ids)
benches = ids[:30]
for name, fn in [("constant 0.5", simple.constant_half), ("empirical mean", simple.empirical_mean), ("smoothed mean", simple.smoothed_mean)]:
    r = harness.evaluate(fn, benches=benches, max_pairs=6, max_eval_rows=300)
    print(f"{name:16s} ALC {r['brier_alc']:.4f} | pairs {r['n_pairs']} over {r['n_benchmarks']} benchmarks | by budget {r['brier_by_budget']}")
