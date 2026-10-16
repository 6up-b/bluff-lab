"""Grid posterior over strategic bluff propensity q (SPEC §5.1). Shared by agents and hypotheses."""
from __future__ import annotations

from typing import Iterable

import numpy as np

GRID = np.linspace(0.0, 1.0, 101)


def uniform() -> np.ndarray:
    return np.full(GRID.shape, 1.0 / GRID.size)


def update(post: np.ndarray, p_c: float, was_bluff: bool, weight: float = 1.0) -> np.ndarray:
    """One revealed claim with recorded card-math prob p_c. `weight` tempers the likelihood (γ discounting)."""
    p_c = float(min(max(p_c, 0.0), 1.0))
    lik = p_c + (1.0 - p_c) * GRID if was_bluff else (1.0 - p_c) * (1.0 - GRID)
    lik = np.maximum(lik, 1e-12) ** weight
    out = post * lik
    s = out.sum()
    return out / s if s > 0 else uniform()


def posterior(claims: Iterable[tuple[float, bool] | tuple[float, bool, float]]) -> np.ndarray:
    post = uniform()
    for c in claims:
        post = update(post, c[0], c[1], c[2] if len(c) > 2 else 1.0)
    return post


def mean(post: np.ndarray) -> float:
    return float((post * GRID).sum())


def interval(post: np.ndarray, mass: float = 0.95) -> tuple[float, float, float]:
    """[mean, lo, hi] equal-tailed credible interval."""
    cdf = np.cumsum(post)
    lo = float(GRID[np.searchsorted(cdf, (1 - mass) / 2)])
    hi = float(GRID[min(np.searchsorted(cdf, 1 - (1 - mass) / 2), GRID.size - 1)])
    return (round(mean(post), 4), lo, hi)


def p_diff_greater(post_a: np.ndarray, post_b: np.ndarray, delta: float,
                   rng: np.random.Generator | None = None, n: int = 2000) -> float:
    """P(q_a - q_b > delta) from n samples of each grid posterior."""
    rng = rng or np.random.default_rng(0)
    a = rng.choice(GRID, size=n, p=post_a)
    b = rng.choice(GRID, size=n, p=post_b)
    return float(np.mean(a - b > delta))


def to_list(post: np.ndarray) -> list[float]:
    return [round(float(x), 8) for x in post]


def from_list(xs: list[float] | None) -> np.ndarray:
    if not xs:
        return uniform()
    a = np.asarray(xs, dtype=float)
    return a / a.sum()
