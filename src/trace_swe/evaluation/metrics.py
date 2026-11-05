"""Small, dependency-free metrics for evaluation reports."""

import math
import random


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval for a binomial success rate."""
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """Unbiased pass@k estimate for n samples with c successes."""
    if n <= 0 or k <= 0 or k > n or c < 0 or c > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def cost_per_solved(total_cost: float | None, solved: int) -> float | None:
    if total_cost is None or solved <= 0:
        return None
    return total_cost / solved


def _bootstrap_means(values: list[float], resamples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    n = len(values)
    return sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))


def bootstrap_mean_ci(
    values: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float] | None:
    """Percentile bootstrap confidence interval for the mean."""
    if len(values) < 2:
        return None
    means = _bootstrap_means(values, resamples, seed)
    alpha = (1 - confidence) / 2
    return means[max(0, int(alpha * resamples))], means[min(resamples - 1, int((1 - alpha) * resamples))]


def paired_bootstrap_diff(
    current: list[float], baseline: list[float], confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> dict | None:
    """Paired bootstrap for the mean difference current - baseline."""
    if len(current) != len(baseline) or not current:
        return None
    differences = [a - b for a, b in zip(current, baseline)]
    interval = bootstrap_mean_ci(differences, confidence=confidence, resamples=resamples, seed=seed)
    delta = sum(differences) / len(differences)
    if interval is None:
        return {"delta": delta, "ci_low": None, "ci_high": None, "p_value": None}
    means = _bootstrap_means(differences, resamples, seed + 1)
    low_tail = (sum(1 for mean in means if mean <= 0) + 1) / (resamples + 1)
    high_tail = (sum(1 for mean in means if mean >= 0) + 1) / (resamples + 1)
    return {
        "delta": delta,
        "ci_low": interval[0],
        "ci_high": interval[1],
        "p_value": min(1.0, 2 * min(low_tail, high_tail)),
    }


def mcnemar_exact(current: list[bool], baseline: list[bool]) -> float | None:
    """Exact two-sided McNemar p-value for paired binary outcomes."""
    if len(current) != len(baseline):
        return None
    baseline_only = sum(1 for cur, base in zip(current, baseline) if base and not cur)
    current_only = sum(1 for cur, base in zip(current, baseline) if cur and not base)
    discordant = baseline_only + current_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(baseline_only, current_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))
