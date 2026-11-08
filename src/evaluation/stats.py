"""Uncertainty estimates for offline evaluation (bootstrap over users).

* bootstrap_ci     - percentile confidence interval of a mean metric
* paired_bootstrap - CI and two-sided p-value for the mean difference between two models
                     evaluated on the *same* users (resampling users keeps the pairing)
"""
from __future__ import annotations

import numpy as np


def _resample_index(n: int, n_resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_resamples, n))


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 42):
    values = np.asarray(values, dtype=float)
    means = values[_resample_index(len(values), n_resamples, seed)].mean(axis=1)
    a = (1 - confidence) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000, confidence: float = 0.95,
                     seed: int = 42) -> dict:
    """Mean(a - b) with CI and a two-sided bootstrap p-value for H0: mean difference = 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    boot = d[_resample_index(len(d), n_resamples, seed)].mean(axis=1)
    alpha = (1 - confidence) / 2
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"diff": float(d.mean()), "diff_ci_low": float(np.quantile(boot, alpha)),
            "diff_ci_high": float(np.quantile(boot, 1 - alpha)), "p_value": float(min(1.0, max(p, 1 / n_resamples)))}
