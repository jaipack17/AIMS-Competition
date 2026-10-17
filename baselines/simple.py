"""Reference predictors that need no item text. All match the competition interface."""


def _same(a, b):
    return a == b


def constant_half(input, labeled=None):
    return 0.5


def empirical_mean(input, labeled=None):
    """Unsmoothed pair mean, as in the organizers' baseline (0.5 with no labels)."""
    subj, item = input
    ys = [y for (s, it), y in (labeled or []) if s == subj and it.get("benchmark_id") == item.get("benchmark_id")]
    return sum(ys) / len(ys) if ys else 0.5


def smoothed_mean(input, labeled=None, strength=2.0, prior=0.5):
    """Beta-prior smoothed pair mean: avoids the 0/1 predictions that hurt Brier at small budgets."""
    subj, item = input
    ys = [y for (s, it), y in (labeled or []) if s == subj and it.get("benchmark_id") == item.get("benchmark_id")]
    return (sum(ys) + strength * prior) / (len(ys) + strength)


def pooled_mean(input, labeled=None, strength=2.0, prior=0.5):
    """Beta-smoothed mean of ALL labels on this benchmark, across subjects (uses the platform's shared evidence)."""
    _, item = input
    ys = [y for (s, it), y in (labeled or []) if it.get("benchmark_id") == item.get("benchmark_id")]
    return (sum(ys) + strength * prior) / (len(ys) + strength)
