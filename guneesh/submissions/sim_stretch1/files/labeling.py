"""Streaming acquisition by Fisher information.

In our 1PL model a label on an item with predicted probability p carries Fisher information p(1 - p) about the
subject's ability shift zeta (and about the benchmark shift delta): it is largest, 1/4, when p = 0.5 and small
when the outcome is nearly certain. `prediction` is the current prediction, already using every label bought
so far, so the choice adapts as evidence arrives.

Candidates arrive one at a time and must be taken or skipped at once, so the rule is a threshold:
    r    = labels_remaining / items_remaining       (share of the remaining candidates we can still buy)
    buy  if  p(1 - p) >= (1 - r^2) / 4             (equivalently |p - 0.5| <= r / 2)
If predictions were spread evenly over [0, 1], exactly a share r of candidates would pass, so the expected
purchases match the remaining budget while the most informative items are preferred. When the remaining
candidates are no more than the remaining labels (r >= 1), every candidate is bought, so the budget is used.
"""
from __future__ import annotations


def acquisition_function(input: list, prediction: float | None = None, labeled: list | None = None,
                         context: dict | None = None) -> bool | float:
    if context is None:  # older evaluators call without context and rank by a numeric priority
        return 0.25 if prediction is None else float(prediction * (1.0 - prediction))
    if prediction is None:
        raise ValueError("Fisher-information acquisition needs the current prediction")
    if context["labels_remaining"] <= 0:
        return False
    r = context["labels_remaining"] / max(context["items_remaining"], 1)
    if r >= 1:
        return True
    return bool(prediction * (1.0 - prediction) >= (1.0 - r * r) / 4.0)
