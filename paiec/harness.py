"""Local simulation of the competition's streaming protocol and Brier ALC scoring.

Per subject-benchmark pair with >= MIN_ITEMS distinct items: a persistent random 50/50 split into an
acquisition pool and an evaluation pool; candidates stream in a fixed random order; labels acquired so
far are nested prefixes, evaluated at caps 0,1,3,7,15,31. Simplification: `labeled` holds only this
pair's own labels (the platform also passes evidence from other sampled pairs).
"""
from __future__ import annotations

import zlib

import numpy as np

from . import data

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
MIN_ITEMS = 80


def make_pairs(bench: str, max_pairs: int, max_eval_rows: int, seed: int = 0):
    """Yield dicts with subject, acquisition candidates, and evaluation rows for eligible pairs."""
    rng = np.random.default_rng(zlib.crc32(f"{bench}:{seed}".encode()))
    r = data.load_responses(bench)
    subjects, items = data.load_subjects(bench), data.load_items(bench)
    n_items = r.groupby("subject_id").item_id.nunique()
    eligible = n_items[n_items >= MIN_ITEMS].index.to_numpy(dtype=object)
    rng.shuffle(eligible)
    pairs = []
    for sid in eligible[:max_pairs]:
        g = r[r.subject_id == sid]
        ids = g.item_id.unique().to_numpy(dtype=object); rng.shuffle(ids)
        half = (len(ids) + 1) // 2  # odd count: extra item to acquisition
        acq_ids, eval_ids = ids[:half], ids[half:]
        first = g.drop_duplicates("item_id").set_index("item_id").response
        ev = g[g.item_id.isin(eval_ids)]
        if len(ev) > max_eval_rows:
            keep = rng.choice(eval_ids, size=min(len(eval_ids), max(1, max_eval_rows // max(1, len(ev) // len(eval_ids)))), replace=False)
            ev = ev[ev.item_id.isin(keep)]
        pairs.append({
            "bench": bench, "sid": sid, "subject": subjects.get(sid, {k: "" for k in data.SUBJECT_KEYS}),
            "items": items,
            "acq": [(i, int(first[i])) for i in acq_ids],
            "eval": list(zip(ev.item_id.tolist(), ev.response.astype(int).tolist())),
        })
    return pairs


def _item(pair, item_id):
    d = pair["items"].get(item_id)
    return dict(d) if d else {"item_content": "", "item_features": "", "interactors": "", "benchmark_id": pair["bench"]}


def run_pair(pair, predict, acquisition=None):
    """Return the Brier score at each budget for one pair."""
    subj, cap = pair["subject"], BUDGETS[-1]
    labeled, n_acq = [], len(pair["acq"])
    for k, (iid, y) in enumerate(pair["acq"]):
        if len(labeled) >= cap:
            break
        if acquisition is None:
            take = True  # default random policy: stream order is already random
        else:
            inp = [subj, _item(pair, iid)]
            ctx = {"subject_id": pair["sid"], "benchmark_id": pair["bench"], "labels_remaining": cap - len(labeled),
                   "labels_acquired": len(labeled), "max_labels": cap, "items_remaining": n_acq - k}
            pred = predict(inp, [list(x) for x in labeled])
            take = bool(acquisition(inp, prediction=pred, labeled=[list(x) for x in labeled], context=ctx))
        if take:
            labeled.append(([subj, _item(pair, iid)], y))
    brier = []
    for b in BUDGETS:
        ev = [list(x) for x in labeled[:b]]
        cache, se = {}, []
        for iid, y in pair["eval"]:
            if iid not in cache:
                p = float(predict([subj, _item(pair, iid)], ev))
                assert np.isfinite(p) and 0.0 <= p <= 1.0, f"invalid probability {p}"
                cache[iid] = p
            se.append((cache[iid] - y) ** 2)
        brier.append(float(np.mean(se)))
    return brier


def evaluate(predict, acquisition=None, benches=None, max_pairs=10, max_eval_rows=300, seed=0, verbose=False):
    """Mean over pairs of Brier per budget and Brier ALC (each pair weighted equally)."""
    rows = []
    for b in benches:
        for pair in make_pairs(b, max_pairs, max_eval_rows, seed):
            rows.append((b, run_pair(pair, predict, acquisition)))
            if verbose:
                print(b, pair["sid"], np.round(rows[-1][1], 3), flush=True)
    arr = np.array([r[1] for r in rows])
    return {"n_pairs": len(rows), "n_benchmarks": len({r[0] for r in rows}),
            "brier_by_budget": dict(zip(BUDGETS, arr.mean(0).round(4))),
            "brier_alc": float((arr @ WEIGHTS).mean().round(4)), "per_pair": rows}


def run_group(pairs, predict):
    """Competition-faithful evidence sharing: all pairs of one benchmark draw share their acquired labels.

    Default random acquisition only (labels = first b candidates of each pair's random stream, nested across
    budgets). At budget b, every predict() call receives the labels of ALL pairs in the group, as the platform
    does ("all evidence purchased so far, across sampled pairs"). Returns one Brier list per pair.
    """
    cap = BUDGETS[-1]
    acq = [[(([p["subject"], _item(p, iid)]), y) for iid, y in p["acq"][:cap]] for p in pairs]
    out = [[] for _ in pairs]
    for b in BUDGETS:
        shared = [list(x) for lab in acq for x in lab[:b]]
        for j, p in enumerate(pairs):
            cache, se = {}, []
            for iid, y in p["eval"]:
                if iid not in cache:
                    pr = float(predict([p["subject"], _item(p, iid)], shared))
                    assert np.isfinite(pr) and 0.0 <= pr <= 1.0, f"invalid probability {pr}"
                    cache[iid] = pr
                se.append((cache[iid] - y) ** 2)
            out[j].append(float(np.mean(se)))
    return out
