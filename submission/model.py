"""PAIEC initial submission: IRT-style predictor, NumPy only, no model downloads.

logit = a*theta(subject) + c + m_bench + d_subject
  theta   = name effect + provider effect + year terms   (fitted offline on measurement-db, see artifacts.json)
  m, d    = benchmark offset and per-subject residual, MAP (Gaussian priors tau_b, tau) from the revealed labels
Unseen names/providers contribute 0. Any failure falls back to the global base rate.
"""
from __future__ import annotations

import json
import os

import numpy as np

_A = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts.json")))
_NAME, _PROV = _A["name_row"], _A["prov_row"]
_YR, _a, _c, _tau, _tau_b = _A["yr"], _A["a"], _A["c"], _A["tau"], _A["tau_b"]
_theta_cache: dict = {}
_fit_cache: dict = {}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _skey(subj):
    return tuple(sorted(subj.items()))


def _theta(subj: dict) -> float:
    key = (subj.get("normalized_name", ""), subj.get("provider", ""), subj.get("release_date", ""))
    if key in _theta_cache:
        return _theta_cache[key]
    name, prov, rd = key
    t = _NAME.get(name, 0.0) + _PROV.get(prov, 0.0)
    try:
        yr = float(rd[:4]) - 2024.0; unk = 0.0
    except ValueError:
        yr, unk = 0.0, 1.0
    t += _YR[0] * yr + _YR[1] * yr ** 2 / 10 + _YR[2] * unk
    _theta_cache[key] = float(t)
    return float(t)


def _fit_group(bench, labeled):
    """Joint Newton MAP over a benchmark offset m and per-subject residuals d_s, cached on the label set."""
    rel = [(s, it, y) for (s, it), y in (labeled or []) if it.get("benchmark_id") == bench]
    if not rel:
        return 0.0, {}
    key = (bench, tuple((hash(_skey(s)), hash(it.get("item_content", "")), y) for s, it, y in rel))
    if key in _fit_cache:
        return _fit_cache[key]
    keys = sorted({_skey(s) for s, _, _ in rel}); ix = {k: i for i, k in enumerate(keys)}
    si = np.array([ix[_skey(s)] for s, _, _ in rel]); y = np.array([y for *_, y in rel], float)
    base = np.array([_a * _theta(s) + _c for s, _, _ in rel])
    S = len(keys); par = np.zeros(S + 1); prior = np.r_[1 / _tau_b ** 2, np.full(S, 1 / _tau ** 2)]
    for _ in range(15):
        p_ = _sig(base + par[0] + par[1:][si]); r = y - p_; w = p_ * (1 - p_)
        g = np.r_[r.sum(), np.bincount(si, weights=r, minlength=S)] - prior * par
        H = np.zeros((S + 1, S + 1)); ws = np.bincount(si, weights=w, minlength=S)
        H[0, 0] = w.sum(); H[0, 1:] = H[1:, 0] = ws; H[np.arange(1, S + 1), np.arange(1, S + 1)] = ws
        step = np.linalg.solve(H + np.diag(prior), g); par += step
        if np.abs(step).max() < 1e-6:
            break
    res = (float(par[0]), {k: float(par[1 + i]) for k, i in ix.items()})
    if len(_fit_cache) > 4096:
        _fit_cache.pop(next(iter(_fit_cache)))
    _fit_cache[key] = res
    return res


def predict(input: list, labeled: list | None = None) -> float:
    try:
        subj, item = input
        m, d = _fit_group(item.get("benchmark_id", ""), labeled)
        p = float(_sig(_a * _theta(subj) + _c + m + d.get(_skey(subj), 0.0)))
        return float(min(max(p, 1e-3), 1 - 1e-3)) if np.isfinite(p) else float(_sig(_c))
    except Exception:
        return float(min(max(_sig(_c), 1e-3), 1 - 1e-3))
