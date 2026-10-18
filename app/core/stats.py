"""Small, dependency-free statistics helpers shared by the metrics API and the eval harness."""
import math
import random


def mean(values) -> float | None:
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 4) if values else None


def percentile(values, p: float) -> float | None:
    """Linear-interpolated percentile (same as numpy's default), p in [0, 100]."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return round(values[int(k)], 4)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 4)


def bootstrap_ci(values, iterations: int = 1000, alpha: float = 0.05, seed: int = 7) -> tuple | None:
    """95% bootstrap confidence interval of the mean (deterministic seed for reproducible reports)."""
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(iterations))
    return (round(means[int(iterations * alpha / 2)], 4), round(means[int(iterations * (1 - alpha / 2)) - 1], 4))
