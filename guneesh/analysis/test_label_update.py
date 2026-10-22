"""Checks of a built label_update submission (run after build_submission.py, from the project root):
    python analysis/test_label_update.py [submission folder name, default label_update]

1. fit_shifts agrees with a general-purpose optimiser (scipy BFGS on the same MAP objective) on simulated
   benchmarks with delta, zeta per subject and epsilon per item, and its covariance is the inverse Hessian.
2. Recovery: with many labels the shifts approach the truth.
3. predict() with no labels and both switches off equals the "AP pure" submission; labels of other benchmarks
   change nothing; correct labels raise the prediction.
4. Item shifts: a wrong answer of another subject on the same item lowers the prediction for that item only.
5. Averaging: pulls the prediction toward 0.5 and never across it, less so as labels arrive.
6. labeling.py: buys every candidate when items_remaining <= labels_remaining, never buys with no budget left,
   prefers p near 0.5, and returns a numeric priority when called without context.
The model is imported in local smoke-test mode (no GPU), so b_hat is the ridge intercept for every item.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
from scipy.optimize import minimize

os.environ["PREDICTIVE_EVAL_LOCAL_SMOKE_TEST"] = "1"


def load(variant: str, module: str):
    d = os.path.abspath(os.path.join("submissions", variant))
    sys.path.insert(0, d)
    sys.modules.pop("core", None)
    spec = importlib.util.spec_from_file_location(f"{variant}_{module}", os.path.join(d, f"{module}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.path.remove(d)
    return mod


def main() -> None:
    variant = sys.argv[1] if len(sys.argv) > 1 else "label_update"
    lu = load(variant, "model")
    lab = load(variant, "labeling")
    print(f"{variant}: sigma_delta={lu.SIGMA_DELTA:.3f} sigma_zeta={lu.SIGMA_ZETA:.3f} sigma_item={lu.SIGMA_ITEM:.3f} "
          f"item_shifts={lu.ITEM_SHIFTS} average={lu.AVERAGE}")
    rng = np.random.default_rng(0)
    sig_item = lu.SIGMA_ITEM or 1.0

    # 1 + 2: simulated benchmark, S subjects, J items, every subject labelled on every item n_rep times
    S, J = 4, 30
    for n_rep in (1, 200):
        errs, gaps, cov_gaps = [], [], []
        for rep in range(5):
            truth = np.r_[rng.normal(0, lu.SIGMA_DELTA), rng.normal(0, lu.SIGMA_ZETA, S), rng.normal(0, sig_item, J)]
            s_of, j_of = np.repeat(np.arange(S), J * n_rep), np.tile(np.repeat(np.arange(J), n_rep), S)
            A = np.zeros((len(s_of), 1 + S + J))
            A[:, 0] = -1.0
            A[np.arange(len(s_of)), 1 + s_of] = 1.0
            A[np.arange(len(s_of)), 1 + S + j_of] = -1.0
            base = rng.normal(0, 1.0, len(s_of))
            y = (rng.random(len(s_of)) < 1 / (1 + np.exp(-(base + A @ truth)))).astype(float)
            prec = np.r_[1 / lu.SIGMA_DELTA ** 2, np.full(S, 1 / lu.SIGMA_ZETA ** 2), np.full(J, 1 / sig_item ** 2)]
            w, cov = lu.fit_shifts(base, y, A, prec)

            def obj(v):
                eta = base + A @ v
                return np.sum(np.logaddexp(0, eta) - y * eta) + 0.5 * np.sum(prec * v * v)
            def grad(v):
                return A.T @ (1 / (1 + np.exp(-(base + A @ v))) - y) + prec * v
            ref = minimize(obj, np.zeros(len(prec)), jac=grad, method="BFGS", options={"gtol": 1e-10}).x
            gaps.append(np.max(np.abs(w - ref)))
            p = 1 / (1 + np.exp(-(base + A @ w)))
            cov_gaps.append(np.max(np.abs(cov @ ((A * (p * (1 - p))[:, None]).T @ A + np.diag(prec)) - np.eye(len(w)))))
            # identifiable contrasts: zeta_s - zeta_t and eps_j - eps_k (delta trades off against the means)
            errs.append(np.mean(np.abs(np.diff(w[1 + S:]) - np.diff(truth[1 + S:]))))
        print(f"labels per (subject, item) = {n_rep:>3}: max diff vs scipy {max(gaps):.1e}, "
              f"cov*H - I {max(cov_gaps):.1e}, mean |error| of item-shift differences {np.mean(errs):.3f}")
        assert max(gaps) < 1e-4 and max(cov_gaps) < 1e-8

    # 3: no labels + switches off -> AP pure; other benchmarks change nothing; correct labels raise p
    ap = load("ap_pure", "model")
    subj = {"normalized_name": "GPT-4o", "provider": "openai", "release_date": "", "access_date": "", "harness": "",
            "reasoning_effort": "", "harness_version": "", "subject_features_extra": ""}
    other_subj = {**subj, "normalized_name": "Claude 3 Opus", "provider": "anthropic"}
    item = {"item_content": "What is 1 + 1?", "item_features": "", "interactors": "", "benchmark_id": "b1"}
    item2 = {**item, "item_content": "What is 2 + 2?"}
    item3 = {**item, "item_content": "What is 3 + 3?"}
    other = {**item, "benchmark_id": "b2"}
    switches = (lu.ITEM_SHIFTS, lu.AVERAGE)
    lu.ITEM_SHIFTS, lu.AVERAGE = False, False
    p0, p_ap = lu.predict([subj, item], []), ap.predict([subj, item], [])
    p_other = lu.predict([subj, item], [[[subj, other], 1], [[subj, other], 1]])
    p_with = lu.predict([subj, item], [[[subj, item2], 1], [[subj, item3], 1]])
    print(f"switches off: no labels {p0:.6f} vs AP pure {p_ap:.6f}; labels on another benchmark {p_other:.6f}; "
          f"two correct labels {p_with:.6f}")
    assert abs(p0 - p_ap) < 1e-12 and abs(p_other - p0) < 1e-12 and p_with > p0

    # 4: item shifts: other_subj answered `item` wrong -> p(subj, item) drops below p(subj, item2)
    wrong = [[[other_subj, item], 0], [[other_subj, item3], 1]]
    lu.ITEM_SHIFTS = False
    off = lu.predict([subj, item], wrong), lu.predict([subj, item2], wrong)
    lu.ITEM_SHIFTS = True
    on = lu.predict([subj, item], wrong), lu.predict([subj, item2], wrong)
    print(f"another subject got this item wrong: item shifts off {off[0]:.4f} (same as unlabelled item {off[1]:.4f}); "
          f"on {on[0]:.4f} (unlabelled item {on[1]:.4f})")
    assert abs(off[0] - off[1]) < 1e-12 and on[0] < on[1] - 0.01

    # 5: averaging pulls toward 0.5, less with more labels
    many = [[[subj, {**item, "item_content": f"q{i}"}], int(i % 4 != 0)] for i in range(60)]
    for labels, name in (([], "no labels"), (many, "60 labels")):
        lu.AVERAGE = False
        raw = lu.predict([subj, item], labels)
        lu.AVERAGE = True
        avg = lu.predict([subj, item], labels)
        print(f"averaging, {name}: {raw:.4f} -> {avg:.4f}")
        assert (raw - 0.5) * (avg - 0.5) >= 0 and abs(avg - 0.5) <= abs(raw - 0.5)
        if name == "no labels":
            shrink0 = (avg - 0.5) / (raw - 0.5)
        else:
            assert (avg - 0.5) / (raw - 0.5) > shrink0, "averaging should fade as labels arrive"
    lu.ITEM_SHIFTS, lu.AVERAGE = switches

    # 6: acquisition hook
    ctx = lambda lr, ir: {"labels_remaining": lr, "items_remaining": ir}
    assert lab.acquisition_function(None, 0.99, [], ctx(5, 5)) is True
    assert lab.acquisition_function(None, 0.5, [], ctx(0, 9)) is False
    assert lab.acquisition_function(None, 0.5, [], ctx(3, 100)) is True
    assert lab.acquisition_function(None, 0.9, [], ctx(3, 100)) is False
    assert isinstance(lab.acquisition_function(None, 0.3), float)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
