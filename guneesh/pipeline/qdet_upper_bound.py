"""Step 3 of the QDET plan: how well can Benedetto's (AIED 2023) features rank the items of ONE benchmark, when the
model is trained on other items of that same benchmark?  This is the paper's own setting (train and test inside one
dataset), so it is an upper bound for our real task (unseen benchmarks); it also picks the text variant (q vs full).

Per benchmark with >= MIN_ITEMS items of known b (final IRT fit): a random half (seed = crc32 of the name) trains, the
other half tests.  Models, as in qdet-comparison (github.com/lucabenedetto/qdet-comparison, MIT):
  text2props models  (ling_paper, ling_paper+read; ling_code and read alone only on the first 9 benchmarks;
      on the question text "q" or the full text "full"):
      features -> rows scaled to unit L2 norm (sklearn normalize, as text2props' FeatureEngineeringModule) ->
      RandomForestRegressor, RandomizedSearchCV over n_estimators randint(20, 200), max_depth randint(2, 50)
  r2de models        (tfidf_qo, tfidf_qa, tfidf_full): TfidfVectorizer(stop_words='english') on r2de-preprocessed
      text -> RandomForestRegressor, RandomizedSearchCV over r2de's non-ARC grid (max_features, max_df, min_df, trees)
  hybrids            (ling_paper+read+tfidf_qa on q, ling_paper+read+tfidf_full on full): text2props' hybrid with its
      vectorizer settings FIXED (the code's min_df = max_df = 1000 keeps only words in exactly 1000 documents); we use
      max_features = 1000 with sklearn's default df limits
  emb_kridge         our baseline: kernel ridge on the standardised Qwen3-VL-8B embedding, penalty by inner CV.
Metrics on the test half: Pearson r, Spearman rho, RMSE.  RandomizedSearchCV scores by R^2 (sklearn's default, as in
the paper's code).
Output: data/qdet/upper_bound/<benchmark>.json (resumable), summary printed by analysis/qdet_report.py.
Run:  python pipeline/qdet_upper_bound.py [--n_iter 10] [--cv 5] [--jobs 7] [--only a,b]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from scipy.stats import randint, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import normalize

from avgdir_train import RUN, bench_files, load_bench
from lowrank_dir import informative
from mlp_b_train import keep_awake
from qdet_features import LING_PAPER, READ
from ridge_train import IRT_DIR

OUT = "data/qdet/upper_bound"
MIN_ITEMS = 40
T2P_GRID = {"n_estimators": randint(20, 200), "max_depth": randint(2, 50)}
R2DE_GRID = {"tfidf__max_features": randint(100, 2000), "regressor__max_depth": randint(2, 50),
             "regressor__n_estimators": randint(2, 200), "tfidf__max_df": [0.8, 0.85, 0.9, 0.95, 1.0],
             "tfidf__min_df": [1, 0.05, 0.1, 0.15, 0.2]}
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


def identity(s: str) -> str:  # texts are already r2de-preprocessed; sklearn then drops English stop words, as in r2de
    return s


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    # constant prediction (e.g. the question text is the same for every item): no ranking.  A relative tolerance,
    # because a forest predicting the training mean everywhere still has an std of ~1e-16 from rounding
    ok = np.ptp(pred) > 1e-9 * (1 + np.abs(pred).max())
    return {"r": float(np.corrcoef(pred, y)[0, 1]) if ok else np.nan,
            "rho": float(spearmanr(pred, y)[0]) if ok else np.nan,
            "rmse": float(np.sqrt(np.mean((pred - y) ** 2))), "constant": not ok}


def search(model, grid, X, y, args, seed):
    s = RandomizedSearchCV(model, grid, n_iter=args.n_iter, cv=KFold(args.cv, shuffle=True, random_state=seed),
                           n_jobs=args.jobs, random_state=seed, error_score=np.nan)
    s.fit(X, y)
    return s


def kernel_ridge(Z: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, cv: int, seed: int) -> np.ndarray:
    """Kernel ridge (linear kernel) on the embedding standardised with the training half; penalty lam * mean
    eigenvalue (scale-free, as lowrank_dir 'scratch'), chosen by inner K-fold CV on the training half."""
    m, s = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zs = (Z - m) / s

    def fit_pred(a, b, lam):
        K, KE = Zs[a] @ Zs[a].T, Zs[b] @ Zs[a].T
        ya = y[a] - y[a].mean()
        return KE @ np.linalg.solve(K + lam * np.trace(K) / len(a) * np.eye(len(a)), ya) + y[a].mean()

    score = {lam: [] for lam in LAMS}
    for a, b in KFold(cv, shuffle=True, random_state=seed).split(tr):
        for lam in LAMS:
            score[lam].append(np.mean((fit_pred(tr[a], tr[b], lam) - y[tr[b]]) ** 2))
    best = min(LAMS, key=lambda lam: np.mean(score[lam]))
    return fit_pred(tr, te, best)


def prepare(name: str, F: pd.DataFrame, emb_path: str, b_final: pd.Series):
    """Items of one benchmark with an embedding and a final-fit b, and its fixed 50/50 split (seed = crc32 of the
    name): (F, y, Z, tr, te, seed), or None if it has fewer than MIN_ITEMS such items or a constant b."""
    X_emb, emb_items = load_bench(emb_path)
    emb = pd.DataFrame(X_emb, index=emb_items)
    F = F[F["item"].isin(emb.index)]
    y = b_final.reindex(F["item"]).to_numpy()
    ok = ~np.isnan(y)
    F, y = F[ok].reset_index(drop=True), y[ok]
    Z = emb.loc[F["item"]].to_numpy()
    if len(y) < MIN_ITEMS or np.std(y) == 0:
        return None
    seed = zlib.crc32(name.encode())
    perm = np.random.default_rng(seed).permutation(len(y))
    return F, y, Z, np.sort(perm[: len(y) // 2]), np.sort(perm[len(y) // 2:]), seed


def run_benchmark(name: str, F: pd.DataFrame, emb_path: str, b_final: pd.Series, args) -> dict:
    data = prepare(name, F, emb_path, b_final)
    if data is None:
        return {"benchmark": name, "skipped": "too few items or constant b"}
    F, y, Z, tr, te, seed = data
    res = {"benchmark": name, "n_train": len(tr), "n_test": len(te), "models": {}}

    def record(key, pred, t0, extra=None):
        res["models"][key] = {**metrics(pred, y[te]), "sec": round(time.time() - t0, 1), **(extra or {})}

    # ling_code and read alone were dropped after the first 9 benchmarks (clearly the weakest; their files keep them)
    groups = {"ling_paper": LING_PAPER, "ling_paper+read": LING_PAPER + READ}
    for v in ("q", "full"):
        for g, cols in groups.items():
            t0 = time.time()
            X = normalize(F[[f"{v}__{c}" for c in cols]].to_numpy(np.float64))
            s = search(RandomForestRegressor(random_state=seed), T2P_GRID, X[tr], y[tr], args, seed)
            record(f"{g}[{v}]", s.predict(X[te]), t0, {"cv_r2": float(s.best_score_)})
    for col in ("tfidf_qo", "tfidf_qa", "tfidf_full"):
        t0 = time.time()
        pipe = Pipeline([("tfidf", TfidfVectorizer(stop_words="english", preprocessor=identity)),
                         ("regressor", RandomForestRegressor(random_state=seed))])
        texts = F[col].to_numpy()
        try:
            s = search(pipe, R2DE_GRID, texts[tr], y[tr], args, seed)
            record(col, s.predict(texts[te]), t0, {"cv_r2": float(s.best_score_)})
        except ValueError as e:  # every sampled setting failed (e.g. no words left in a tiny benchmark)
            res["models"][col] = {"failed": str(e)[:200]}
    for v, col in (("q", "tfidf_qa"), ("full", "tfidf_full")):
        t0 = time.time()
        try:
            vec = TfidfVectorizer(stop_words="english", preprocessor=identity, max_features=1000)
            T_tr, T_te = vec.fit_transform(F[col].to_numpy()[tr]), vec.transform(F[col].to_numpy()[te])
            D = F[[f"{v}__{c}" for c in LING_PAPER + READ]].to_numpy(np.float64)
            X_tr = normalize(hstack([csr_matrix(D[tr]), T_tr]).tocsr())
            X_te = normalize(hstack([csr_matrix(D[te]), T_te]).tocsr())
            s = search(RandomForestRegressor(random_state=seed), T2P_GRID, X_tr, y[tr], args, seed)
            record(f"ling_paper+read+{col}[{v}]", s.predict(X_te), t0, {"cv_r2": float(s.best_score_)})
        except ValueError as e:
            res["models"][f"ling_paper+read+{col}[{v}]"] = {"failed": str(e)[:200]}
    if informative(Z):
        t0 = time.time()
        record("emb_kridge", kernel_ridge(Z, y, tr, te, args.cv, seed), t0)
    return res


def main() -> None:
    keep_awake()
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=10)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--only", default="", help="comma-separated benchmarks (timing runs)")
    ap.add_argument("--features", default="data/qdet")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    feats = pd.read_parquet(os.path.join(args.features, "features.parquet"))
    texts = pd.read_parquet(os.path.join(args.features, "tfidf_text.parquet"))
    assert (feats["item"].values == texts["item"].values).all()
    feats = pd.concat([feats, texts[["tfidf_qo", "tfidf_qa", "tfidf_full"]]], axis=1)
    b_final = pd.read_parquet(os.path.join(IRT_DIR, RUN, "final", "items.parquet"), columns=["item", "b"]).set_index("item").b
    files = bench_files()
    names = args.only.split(",") if args.only else sorted(set(feats.benchmark) & set(files))
    os.makedirs(args.out, exist_ok=True)
    for i, name in enumerate(names):
        path = os.path.join(args.out, f"{name}.json")
        if os.path.exists(path):
            continue
        res = run_benchmark(name, feats[feats.benchmark == name], files[name], b_final, args)
        with open(path + ".tmp", "w") as f:
            json.dump(res, f, indent=1)
        os.replace(path + ".tmp", path)
        m = res.get("models", {})
        best = max(((k, v["r"]) for k, v in m.items() if not np.isnan(v.get("r", np.nan))), key=lambda kv: kv[1],
                   default=("-", np.nan))
        print(f"[{i + 1}/{len(names)}] {name}: {res.get('skipped', '')}emb r {m.get('emb_kridge', {}).get('r', np.nan):.3f}, "
              f"best {best[0]} r {best[1]:.3f}  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
