"""Competition-interface predictor built from fitted artifacts (NumPy only at inference).

z = theta(subject) - beta_hat(item) + delta ;  p = sigmoid(a * z + c)
delta: 1D MAP shift from labels of the same subject and benchmark, Gaussian prior N(0, tau^2).
"""
from __future__ import annotations
import numpy as np


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class IRTPredictor:
    def __init__(self, s1, s2, embed, a=1.0, c=0.0, tau=1.0, tau_b=2.0, g=None):
        self.name_ix = {x: i for i, x in enumerate(s1["names"])}; self.prov_ix = {x: i for i, x in enumerate(s1["provs"])}
        self.s1, self.s2, self.embed, self.a, self.c, self.tau, self.tau_b = s1, s2, embed, a, c, tau, tau_b
        self.g = a if g is None else g   # scale on item difficulty (own slope; defaults to the ability slope)
        self._fit_cache = {}
        self._theta_cache, self._beta_cache = {}, {}
        self._has_head = bool(np.any(s2["coef"]))

    def theta(self, subj: dict) -> float:
        key = (subj.get("normalized_name", ""), subj.get("provider", ""), subj.get("release_date", ""))
        if key in self._theta_cache: return self._theta_cache[key]
        name, prov, rd = key
        t = 0.0
        if name in self.name_ix: t += self.s1["name_row"][self.name_ix[name]]
        if prov in self.prov_ix: t += self.s1["prov_row"][self.prov_ix[prov]]
        try:
            yr = float(rd[:4]) - 2024.0; unk = 0.0
        except ValueError:
            yr, unk = 0.0, 1.0
        y = self.s1["yr"]; t += y[0] * yr + y[1] * yr ** 2 / 10 + y[2] * unk
        if name or prov:
            t += float(self.embed(f"{name} | {prov}") @ self.s1["w"])
        self._theta_cache[key] = float(t); return float(t)

    def beta(self, item: dict) -> float:
        if not self._has_head: return float(self.s2["b0"])   # constant difficulty: skip item encoding
        text = (item.get("item_content") or "")[:2000]
        if text in self._beta_cache: return self._beta_cache[text]
        b = float(self.embed(text, item=True) @ self.s2["coef"] + self.s2["b0"]) if text else float(self.s2["b0"])
        self._beta_cache[text] = b; return b

    def _skey(self, subj):
        return tuple(sorted(subj.items()))

    def _fit_group(self, bench, labeled):
        """Joint MAP over a benchmark offset m (shared by all subjects, prior N(0, tau_b^2)) and per-subject
        residuals d_s (prior N(0, tau^2)), in the calibrated logit space. Cached on the label set."""
        rel = [(s, it, y) for (s, it), y in (labeled or []) if it.get("benchmark_id") == bench]
        key = (bench, tuple((hash(self._skey(s)), hash(it.get("item_content", "")), y) for s, it, y in rel))
        if key in self._fit_cache: return self._fit_cache[key]
        if not rel:
            res = (0.0, {})
        else:
            keys = sorted({self._skey(s) for s, _, _ in rel}); ix = {k: i for i, k in enumerate(keys)}
            si = np.array([ix[self._skey(s)] for s, _, _ in rel]); y = np.array([y for *_, y in rel], float)
            base = np.array([self.a * self.theta(s) - self.g * self.beta(it) + self.c for s, it, _ in rel])
            S = len(keys); par = np.zeros(S + 1); prior = np.r_[1 / self.tau_b ** 2, np.full(S, 1 / self.tau ** 2)]
            for _ in range(15):
                p_ = _sig(base + par[0] + par[1:][si]); r = y - p_; w = p_ * (1 - p_)
                g = np.r_[r.sum(), np.bincount(si, weights=r, minlength=S)] - prior * par
                H = np.zeros((S + 1, S + 1)); ws = np.bincount(si, weights=w, minlength=S)
                H[0, 0] = w.sum(); H[0, 1:] = H[1:, 0] = ws; H[np.arange(1, S + 1), np.arange(1, S + 1)] = ws
                H += np.diag(prior)
                step = np.linalg.solve(H, g); par += step
                if np.abs(step).max() < 1e-6: break
            res = (float(par[0]), {k: float(par[1 + i]) for k, i in ix.items()})
        self._fit_cache[key] = res
        if len(self._fit_cache) > 4096: self._fit_cache.pop(next(iter(self._fit_cache)))
        return res

    def __call__(self, input, labeled=None):
        subj, item = input
        logit = self.a * self.theta(subj) - self.g * self.beta(item) + self.c
        m, d = self._fit_group(item.get("benchmark_id"), labeled)
        return float(np.clip(_sig(logit + m + d.get(self._skey(subj), 0.0)), 1e-3, 1 - 1e-3))
