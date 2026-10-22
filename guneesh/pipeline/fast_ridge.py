"""Exact ridge regression for many alphas and cross-validation folds at once.

Reproduces scikit-learn's  StandardScaler -> RidgeCV(alphas, cv=<splits>)  (the pipeline of
agent-psychometrics FeatureBasedPredictor) without refitting a model for every alpha and every inner fold.

Why it gives the same answer:
  sklearn Ridge (fit_intercept=True, dense X, solver "auto" -> cholesky) centres the training rows and solves
      (Xc' Xc + alpha I) w = Xc' yc ,   intercept = mean(y) - mean(X) . w
  Every term above is a sum over rows, so it can be built from per-fold sums (n, sum z, sum zz', sum zy,
  sum y, sum y^2). The training sums of inner fold k are "all folds minus fold k". One eigendecomposition
  C = V diag(e) V' of the centred matrix then gives the solution for every alpha:
      w(alpha) = V diag(1 / (e + alpha)) V' c
  The held-out error of fold k also follows from fold k's sums, so no second pass over the data is needed:
      SSE = sum y^2 - 2 w'(sum zy) - 2 b sum y + w'(sum zz')w + 2 b w'(sum z) + n b^2
  RidgeCV(cv=...) runs GridSearchCV, which keeps the alpha with the best mean fold score and, on ties,
  the first one in the list; the default score is R^2 (Ridge.score), "mse" selects by mean squared error.
  "r2_within" (our addition, not an sklearn score) is R^2 after removing each benchmark's mean from both the
  targets and the predictions of the held-out rows, i.e. it only judges how items are ordered inside each
  benchmark. It needs each row's group (benchmark); the within-group sums follow from the fold sums minus
  n_g * mean_g * mean_g' per group, so it costs one extra (n_groups x d) accumulation.
  The proof against sklearn is analysis/ridge_vs_sklearn.py.

All sums are accumulated in float64, in row chunks, so the scaled matrix is never held in memory at once.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CHUNK = 8192


@dataclass
class Scaler:
    """Same as sklearn StandardScaler: population std (ddof=0); features with zero variance get scale 1."""
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, rows: np.ndarray) -> "Scaler":
        d = X.shape[1]
        s, ss, n = np.zeros(d), np.zeros(d), 0
        for i in range(0, len(rows), CHUNK):
            x = X[rows[i:i + CHUNK]].astype(np.float64)
            s += x.sum(0)
            n += len(x)
        mean = s / n
        for i in range(0, len(rows), CHUNK):  # two passes: numerically safer than E[x^2] - E[x]^2
            x = X[rows[i:i + CHUNK]].astype(np.float64) - mean
            ss += (x * x).sum(0)
        scale = np.sqrt(ss / n)
        scale[scale == 0] = 1.0
        return cls(mean, scale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x.astype(np.float64) - self.mean) / self.scale


@dataclass
class Sums:
    """Sufficient statistics of a set of rows, for one or more targets (columns of Y)."""
    n: float
    z: np.ndarray    # (d,)      sum of z
    zz: np.ndarray   # (d, d)    sum of z z'
    zy: np.ndarray   # (d, t)    sum of z y
    y: np.ndarray    # (t,)      sum of y
    yy: np.ndarray   # (t,)      sum of y^2
    within: tuple | None = None   # (Wzz, Wzy, Wyy): the same sums centred within each group (held-out use only)

    def __add__(self, o: "Sums") -> "Sums":
        return Sums(self.n + o.n, self.z + o.z, self.zz + o.zz, self.zy + o.zy, self.y + o.y, self.yy + o.yy)

    def __sub__(self, o: "Sums") -> "Sums":
        return Sums(self.n - o.n, self.z - o.z, self.zz - o.zz, self.zy - o.zy, self.y - o.y, self.yy - o.yy)


def fold_sums(X: np.ndarray, Y: np.ndarray, rows: np.ndarray, fold: np.ndarray, scaler: Scaler,
              groups: np.ndarray | None = None) -> list[Sums]:
    """Sums of the scaled rows X[rows] for each inner fold id 0..K-1 given in `fold` (aligned with `rows`).
    With `groups` (aligned with `rows`), also the within-group centred sums needed by score="r2_within"."""
    d, t = X.shape[1], Y.shape[1]
    out = []
    for k in range(int(fold.max()) + 1):
        r = rows[fold == k]
        s = Sums(0.0, np.zeros(d), np.zeros((d, d)), np.zeros((d, t)), np.zeros(t), np.zeros(t))
        for i in range(0, len(r), CHUNK):
            z = scaler.transform(X[r[i:i + CHUNK]])
            y = Y[r[i:i + CHUNK]].astype(np.float64)
            s.n += len(z)
            s.z += z.sum(0)
            s.zz += z.T @ z
            s.zy += z.T @ y
            s.y += y.sum(0)
            s.yy += (y * y).sum(0)
        if groups is not None:
            _, g = np.unique(groups[fold == k], return_inverse=True)
            gn, gz, gy = np.zeros(g.max() + 1), np.zeros((g.max() + 1, d)), np.zeros((g.max() + 1, t))
            for i in range(0, len(r), CHUNK):
                gi = g[i:i + CHUNK]
                np.add.at(gn, gi, 1.0)
                np.add.at(gz, gi, scaler.transform(X[r[i:i + CHUNK]]))
                np.add.at(gy, gi, Y[r[i:i + CHUNK]].astype(np.float64))
            mz, my = gz / gn[:, None], gy / gn[:, None]
            s.within = (s.zz - (mz * gn[:, None]).T @ mz, s.zy - (mz * gn[:, None]).T @ my,
                        s.yy - (gn[:, None] * my * my).sum(0))
        out.append(s)
    return out


def solve(s: Sums, alphas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ridge with intercept on the rows summarised by `s`, for every alpha and target.
    Returns coef (L, t, d) and intercept (L, t)."""
    mz, my = s.z / s.n, s.y / s.n
    C = s.zz - s.n * np.outer(mz, mz)          # centred X'X
    c = s.zy - s.n * np.outer(mz, my)          # centred X'y, (d, t)
    e, V = np.linalg.eigh(C)
    e = np.clip(e, 0.0, None)                  # C is positive semi-definite; remove -1e-12 rounding
    Vc = V.T @ c                               # (d, t)
    coef = np.einsum("dk,lk,kt->ltd", V, 1.0 / (e[None, :] + alphas[:, None]), Vc)
    intercept = my[None, :] - coef @ mz        # (L, t)
    return coef, intercept


def heldout_scores(s: Sums, coef: np.ndarray, intercept: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """MSE and R^2 of every (alpha, target) model on the rows summarised by `s`. Shapes (L, t)."""
    wzy = np.einsum("ltd,dt->lt", coef, s.zy)
    wzzw = np.einsum("ltd,de,lte->lt", coef, s.zz, coef)
    wz = coef @ s.z
    sse = s.yy - 2 * wzy - 2 * intercept * s.y + wzzw + 2 * intercept * wz + s.n * intercept ** 2
    sst = s.yy - s.y ** 2 / s.n
    return sse / s.n, 1.0 - sse / sst


def heldout_r2_within(s: Sums, coef: np.ndarray) -> np.ndarray:
    """R^2 with each group's mean removed from targets and predictions (the intercept cancels). Shape (L, t)."""
    Wzz, Wzy, Wyy = s.within
    sse = Wyy - 2 * np.einsum("ltd,dt->lt", coef, Wzy) + np.einsum("ltd,de,lte->lt", coef, Wzz, coef)
    return 1.0 - sse / Wyy


@dataclass
class RidgeCVResult:
    alphas: np.ndarray
    cv_score: np.ndarray    # (L, t) mean over inner folds of the selection score (R^2, -MSE or within R^2)
    cv_mse: np.ndarray      # (L, t) mean inner-fold MSE (reported whatever the selection score is)
    best: np.ndarray        # (t,) index of the chosen alpha per target
    coef: np.ndarray        # (t, d) refit on all training rows with the chosen alpha
    intercept: np.ndarray   # (t,)


def ridge_cv(folds: list[Sums], alphas, score: str = "r2") -> RidgeCVResult:
    """sklearn RidgeCV(alphas, cv=splits) for every target column, from per-inner-fold sums."""
    if score not in ("r2", "mse", "r2_within"):
        raise ValueError("score must be 'r2', 'mse' or 'r2_within'")
    if score == "r2_within" and any(s.within is None for s in folds):
        raise ValueError("r2_within needs fold_sums(..., groups=...)")
    alphas = np.asarray(alphas, dtype=np.float64)
    total = folds[0]
    for s in folds[1:]:
        total = total + s
    sel, mse = [], []
    for s in folds:
        coef, b = solve(total - s, alphas)
        m, r2 = heldout_scores(s, coef, b)
        sel.append(r2 if score == "r2" else -m if score == "mse" else heldout_r2_within(s, coef))
        mse.append(m)
    cv_score, cv_mse = np.mean(sel, 0), np.mean(mse, 0)
    best = np.argmax(cv_score, axis=0)         # first maximum, like GridSearchCV's rank-1 pick
    coef, b = solve(total, alphas)
    t = np.arange(coef.shape[1])
    return RidgeCVResult(alphas, cv_score, cv_mse, best, coef[best, t], b[best, t])
