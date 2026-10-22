"""PAIEC submission "label update": AP two-stage IRT plus two shifts estimated from the revealed labels.

    P(correct) = sigmoid( (theta(subject) + zeta) - (b_hat(item) + delta) )

theta, b_hat   as in submission "AP pure" (core.py).
delta          shift of this benchmark's difficulty, shared by all its subjects: corrects the ridge's guess of
               the benchmark's overall level. Prior N(0, sigma_delta^2); sigma_delta = RMS over the 115 held-out
               benchmarks of the 5 folds of (mean IRT b - mean b_hat).
zeta           this subject's ability shift on this benchmark (LEGO-MB theta_{i,m} = theta_i + zeta_{i,m}, part of
               our IRT model). Prior N(0, sigma_zeta^2) with sigma_zeta from the IRT fit.
Both are the joint MAP estimate given all labels of this benchmark in `labeled` (every subject, since `labeled`
holds evidence across all sampled pairs): Newton's method on the logistic log-likelihood plus the Gaussian
priors (strictly concave, so the maximum is unique). With no labels delta = zeta = 0 and the prediction equals
"AP pure". A subject is matched by its 8 fields, a benchmark by benchmark_id.

The embedding model loads at import in every process except the kit's local smoke tools: the acquisition hook
(labeling.py) needs predictions in the parent process, and predict() calls have a per-call time limit.
"""
from __future__ import annotations

import json
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
_SHIFT_MEMO: dict = {}

if not core.SMOKE_TEST:
    core.load_model()


def _subject_key(subject: dict) -> str:
    return json.dumps(subject, sort_keys=True)


def fit_shifts(base: np.ndarray, y: np.ndarray, who: np.ndarray, n_subjects: int,
               sigma_delta: float = SIGMA_DELTA, sigma_zeta: float = SIGMA_ZETA) -> tuple[float, np.ndarray]:
    """MAP of (delta, zeta_0..zeta_{S-1}) for y_k ~ Bernoulli(sigmoid(base_k + zeta_{who_k} - delta)).

    base = theta - b_hat of each label, who = subject index of each label. Returns delta and the zeta vector.
    """
    w = np.zeros(n_subjects + 1)                         # w[0] = delta, w[1:] = zeta
    prec = np.r_[1 / sigma_delta ** 2, np.full(n_subjects, 1 / sigma_zeta ** 2)]
    A = np.zeros((len(y), n_subjects + 1))               # eta = base + A @ w
    A[:, 0] = -1.0
    A[np.arange(len(y)), who + 1] = 1.0

    def objective(v):
        eta = base + A @ v
        return float(np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(prec * v * v))

    f = objective(w)
    for _ in range(100):
        eta = base + A @ w
        p = 1 / (1 + np.exp(-eta))
        grad = A.T @ (p - y) + prec * w
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(prec)
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
    return float(w[0]), w[1:]


def _benchmark_shifts(benchmark_id: str, labeled: list) -> tuple[float, dict]:
    rows = [(pair[0], pair[1], int(label)) for pair, label in labeled
            if str(pair[1].get("benchmark_id") or "") == benchmark_id]
    if not rows:
        return 0.0, {}
    memo_key = (benchmark_id, json.dumps(rows, sort_keys=True))
    if memo_key not in _SHIFT_MEMO:                       # every call of a checkpoint gets the same labels
        subjects = sorted({_subject_key(s) for s, _, _ in rows})
        index = {k: i for i, k in enumerate(subjects)}
        base = np.array([core.subject_theta(s) - core.item_b_hat(it) for s, it, _ in rows])
        y = np.array([lab for _, _, lab in rows], dtype=float)
        who = np.array([index[_subject_key(s)] for s, _, _ in rows])
        delta, zeta = fit_shifts(base, y, who, len(subjects))
        _SHIFT_MEMO[memo_key] = (delta, dict(zip(subjects, zeta.tolist())))
    return _SHIFT_MEMO[memo_key]


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    delta, zetas = _benchmark_shifts(str(item.get("benchmark_id") or ""), labeled or [])
    zeta = zetas.get(_subject_key(subject), 0.0)
    return core.sigmoid(core.subject_theta(subject) + zeta - core.item_b_hat(item) - delta)
