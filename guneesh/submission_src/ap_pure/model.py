"""PAIEC submission "AP pure": Agent Psychometrics two-stage IRT, without using revealed labels.

    P(correct) = sigmoid(theta(subject) - b_hat(item))                       (AP paper, Eq. 3)
    (with mirt.json bundled, the logit is the MIRT one: see core.base_logit)

theta and b_hat come from core.py. `labeled` is not used, so the prediction is the same at every label budget.
Prediction workers (PAIEC_PREDICTION_WORKER_ID set) load the embedding model when they import this file, so no
predict() call pays for loading; the parent process (validation, default random acquisition) never needs it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = str(Path(__file__).resolve().parent)
if HERE not in sys.path:  # the kit's loaders only put the submission dir on sys.path during import
    sys.path.insert(0, HERE)
import core  # noqa: E402

if os.environ.get("PAIEC_PREDICTION_WORKER_ID") is not None and not core.SMOKE_TEST:
    core.load_model()


def predict(input: list, labeled: list | None = None) -> float:
    subject, item = input
    return core.sigmoid(core.base_logit(subject, item))
