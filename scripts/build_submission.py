"""Fit stage 1 on ALL benchmarks (no text term) and write submission/artifacts.json."""
import sys, json
sys.path.insert(0, ".")
import numpy as np
from paiec import fit

T = fit.load_table("data/train_qwen.npz")
s1 = fit.fit_stage1(T, np.arange(len(T["benches"])), lam=dict(r=1000, name=0.03, prov=0.03, text=1e9), epochs=300)
cv = json.load(open("data/final_cv.json"))
art = dict(name_row={n: float(v) for n, v in zip(s1["names"], s1["name_row"])},
           prov_row={n: float(v) for n, v in zip(s1["provs"], s1["prov_row"])},
           yr=[float(x) for x in s1["yr"]], a=cv["a"], c=cv["c"], tau=0.5, tau_b=1.0)
json.dump(art, open("submission/artifacts.json", "w"))
print(len(art["name_row"]), "names", len(art["prov_row"]), "providers | yr", np.round(art["yr"], 3), "| a", round(art["a"], 3), "c", round(art["c"], 3), "| nll", round(s1["nll"], 4))
