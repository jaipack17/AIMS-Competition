"""PAIEC submission "label update": AP two-stage IRT plus shifts estimated from the revealed labels.

    P(correct) = sigmoid( (theta(subject) + zeta) - (b_hat(item) + delta + epsilon) )

theta, b_hat   as in submission "AP pure" (core.py).
delta          shift of this benchmark's difficulty, shared by all its subjects: corrects the ridge's guess of
               the benchmark's overall level. Prior N(0, sigma_delta^2); sigma_delta = RMS over the 115 held-out
               benchmarks of the 5 folds of (mean IRT b - mean b_hat).
zeta           this subject's ability shift on this benchmark (LEGO-MB theta_{i,m} = theta_i + zeta_{i,m}, part of
               our IRT model). Prior N(0, sigma_zeta^2 + extra variance of theta for an unseen name).
epsilon        (if item_shifts) this item's own difficulty shift, shared by all subjects: a label of another
               subject on the same item tells how far b_hat is off for it. Prior N(0, sigma_item^2); sigma_item =
               RMS of the within-benchmark error of held-out b_hat. An item without labels keeps epsilon = 0.
All shifts are the joint MAP estimate given all labels of this benchmark in `labeled` (every subject, since
`labeled` holds evidence across all sampled pairs): Newton's method on the logistic log-likelihood plus the
Gaussian priors (strictly concave, so the maximum is unique).

If average_over_uncertainty: the prediction is the average of sigmoid over the remaining uncertainty of the
logit (Laplace posterior of the fitted shifts, the prior variance of the shifts without labels, and sigma_item^2
for the item when it has no item shift), with the probit approximation
    E[sigmoid(z)] ~ sigmoid(mean / sqrt(1 + pi * var / 8)),     z ~ N(mean, var).
This pulls a prediction toward 0.5 while little is known, and less so as labels arrive.
With no labels and both switches off the prediction equals "AP pure". A subject is matched by its 8 fields, an
item by its 4 fields, a benchmark by benchmark_id.

The embedding model loads at import in every process except the kit's local smoke tools: the acquisition hook
(labeling.py) needs predictions in the parent process, and predict() calls have a per-call time limit.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # the kit's loaders only put the submission dir on sys.path during import
    sys.path.insert(0, str(HERE))
import core  # noqa: E402

with open(HERE / "shift_priors.json", encoding="utf-8") as _f:
    _PRIORS = json.load(_f)
SIGMA_DELTA, SIGMA_ZETA = float(_PRIORS["sigma_delta"]), float(_PRIORS["sigma_zeta"])
SIGMA_ITEM = float(_PRIORS.get("sigma_item", 0.0))
ITEM_SHIFTS = bool(_PRIORS.get("item_shifts", False))
AVERAGE = bool(_PRIORS.get("average_over_uncertainty", False))
_SHIFT_MEMO: dict = {}

if not core.SMOKE_TEST:
    core.load_model()


def _key(d: dict) -> str:
    return json.dumps(d, sort_keys=True)


def fit_shifts(base: np.ndarray, y: np.ndarray, A: np.ndarray, prec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MAP of w for y_k ~ Bernoulli(sigmoid(base_k + A_k @ w)) with prior w ~ N(0, diag(1 / prec)).

    Returns w and the Laplace covariance of w (inverse Hessian of the negative log-posterior at w).
    """
    w = np.zeros(A.shape[1])

    def objective(v):
        eta = base + A @ v
        return float(np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(prec * v * v))

    def hessian(v):
        p = 1 / (1 + np.exp(-(base + A @ v)))
        return p, (A * (p * (1 - p))[:, None]).T @ A + np.diag(prec)

    f = objective(w)
    for _ in range(100):
        p, H = hessian(w)
        grad = A.T @ (p - y) + prec * w
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:                                       # backtracking keeps every step an improvement
            f_new = objective(w - t * step)
            if f_new <= f or t < 1e-8:
                break
            t /= 2
        w, f_old, f = w - t * step, f, f_new
        if abs(f_old - f) < 1e-12 and np.max(np.abs(t * step)) < 1e-10:
            break
    return w, np.linalg.inv(hessian(w)[1])


def _subject_prior_var(subject: dict) -> float:
    return SIGMA_ZETA ** 2 + core.subject_theta_var(subject)


def _benchmark_fit(benchmark_id: str, labeled: list) -> dict:
    """Shifts of one benchmark: w = [delta, zeta of each labeled subject, epsilon of each labeled item]."""
    rows = [(pair[0], pair[1], int(label)) for pair, label in labeled
            if str(pair[1].get("benchmark_id") or "") == benchmark_id]
    memo_key = (benchmark_id, ITEM_SHIFTS, json.dumps(rows, sort_keys=True))
    if memo_key in _SHIFT_MEMO:                           # every call of a checkpoint gets the same labels
        return _SHIFT_MEMO[memo_key]
    subject_of = {_key(s): s for s, _, _ in rows}
    subjects = {k: 1 + i for i, k in enumerate(sorted(subject_of))}                 # column of each zeta
    item_keys = sorted({_key(it) for _, it, _ in rows}) if ITEM_SHIFTS else []
    items = {k: 1 + len(subjects) + i for i, k in enumerate(item_keys)}             # column of each epsilon
    prec = np.empty(1 + len(subjects) + len(items))
    prec[0] = 1 / SIGMA_DELTA ** 2
    for k, i in subjects.items():
        prec[i] = 1 / _subject_prior_var(subject_of[k])
    if items:
        prec[1 + len(subjects):] = 1 / SIGMA_ITEM ** 2
    if rows:
        base = np.array([core.subject_theta(s) - core.item_b_hat(it) for s, it, _ in rows])
        y = np.array([lab for _, _, lab in rows], dtype=float)
        A = np.zeros((len(rows), len(prec)))
        A[:, 0] = -1.0                                    # delta
        for r, (s, it, _) in enumerate(rows):
            A[r, subjects[_key(s)]] = 1.0                 # zeta
            if items:
                A[r, items[_key(it)]] = -1.0              # epsilon
        w, cov = fit_shifts(base, y, A, prec)
    else:
        w, cov = np.zeros(1), np.array([[SIGMA_DELTA ** 2]])
    _SHIFT_MEMO[memo_key] = fit = {"w": w, "cov": cov, "subjects": subjects, "items": items}
    return fit


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    fit = _benchmark_fit(str(item.get("benchmark_id") or ""), labeled or [])
    s_i, j_i = fit["subjects"].get(_key(subject)), fit["items"].get(_key(item))
    a = np.zeros(len(fit["w"]))                           # logit = theta - b_hat + a @ w
    a[0] = -1.0
    if s_i is not None:
        a[s_i] = 1.0
    if j_i is not None:
        a[j_i] = -1.0
    mean = core.subject_theta(subject) - core.item_b_hat(item) + float(a @ fit["w"])
    if not AVERAGE:
        return core.sigmoid(mean)
    var = float(a @ fit["cov"] @ a)
    var += _subject_prior_var(subject) if s_i is None else 0.0
    var += SIGMA_ITEM ** 2 if j_i is None else 0.0
    return core.sigmoid(mean / math.sqrt(1.0 + math.pi * var / 8.0))
