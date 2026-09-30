"""Dynamic Time Warping distance between two feature sequences (the hand-gesture matcher).

Convention: a DISTANCE - lower is better, 0 for identical sequences.

- Local cost: Euclidean distance between two frame vectors.
- Global path constraint: Sakoe-Chiba band of `band` (fraction of the longer sequence) around the (scaled) diagonal,
  so a small section of one gesture can never absorb most of the other.
- Normalization: accumulated cost divided by the number of steps on the optimal warping path, i.e. the mean per-step
  frame distance along the best alignment. This keeps distances comparable across sequence lengths.

Because the local cost is Euclidean, the distance is unchanged by any orthonormal transform applied to every frame
of both sequences (|Q a - Q b| = |a - b|) - which is what lets `sequence_transform.py` store keyed, transformed
sequences without changing a single DTW value.
"""

from __future__ import annotations

import math

import numpy as np

DEFAULT_BAND = 0.2


def dtw_distance(a: np.ndarray, b: np.ndarray, band: float = DEFAULT_BAND) -> float:
    """Path-length-normalized DTW distance between `a` (N, D) and `b` (M, D)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError(f"DTW needs two (length, dim) sequences with the same dim, got {a.shape} and {b.shape}")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("DTW needs non-empty sequences")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("DTW sequences must be finite")
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).tolist()  # (n, m) local frame distances
    window = max(int(np.ceil(band * max(n, m))), abs(n - m))
    # Row-by-row accumulation on plain Python floats (an order of magnitude faster than indexing numpy scalars; the
    # rotation search runs several DTWs per enrolled sample). Tie-breaking: diagonal, then vertical, then horizontal.
    acc_prev, steps_prev = [math.inf] * (m + 1), [0] * (m + 1)
    acc_prev[0] = 0.0
    for i in range(1, n + 1):
        acc_row, steps_row = [math.inf] * (m + 1), [0] * (m + 1)
        centre = int(round(i * m / n))
        lo, hi = max(1, centre - window), min(m, centre + window)
        cost_row = cost[i - 1]
        for j in range(lo, hi + 1):
            diagonal, vertical, horizontal = acc_prev[j - 1], acc_prev[j], acc_row[j - 1]
            if diagonal <= vertical and diagonal <= horizontal:
                best, steps = diagonal, steps_prev[j - 1]
            elif vertical <= horizontal:
                best, steps = vertical, steps_prev[j]
            else:
                best, steps = horizontal, steps_row[j - 1]
            acc_row[j] = cost_row[j - 1] + best
            steps_row[j] = steps + 1
        acc_prev, steps_prev = acc_row, steps_row
    if not math.isfinite(acc_prev[m]):
        raise ValueError("no warping path inside the band")
    return float(acc_prev[m] / steps_prev[m])


def aggregate(distances: list[float], method: str = "median") -> float:
    """Combine the distances to the enrolled attempts. `median` (default): robust to one poor enrollment attempt, and
    - unlike `min` - not decided by the single closest attempt (which would make a lucky impostor match easier)."""
    values = np.asarray(distances, dtype=np.float64)
    if values.size == 0:
        raise ValueError("no distances to aggregate")
    if method == "median":
        return float(np.median(values))
    if method == "mean":
        return float(values.mean())
    if method == "min":
        return float(values.min())
    raise ValueError(f"unknown aggregation {method!r}")


def dtw_alignment(a: np.ndarray, b: np.ndarray, band: float = DEFAULT_BAND) -> tuple[float, list[tuple[int, int]]]:
    """`dtw_distance` plus its optimal warping path [(i, j), ...] (same recurrence, band and tie-breaking). Used only
    for development diagnostics (which features dominate a distance); matching uses `dtw_distance`."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = len(a), len(b)
    cost = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)).tolist()
    window = max(int(np.ceil(band * max(n, m))), abs(n - m))
    acc = [[math.inf] * (m + 1) for _ in range(n + 1)]
    steps = [[0] * (m + 1) for _ in range(n + 1)]
    move = [[0] * (m + 1) for _ in range(n + 1)]
    acc[0][0] = 0.0
    for i in range(1, n + 1):
        centre = int(round(i * m / n))
        for j in range(max(1, centre - window), min(m, centre + window) + 1):
            diagonal, vertical, horizontal = acc[i - 1][j - 1], acc[i - 1][j], acc[i][j - 1]
            if diagonal <= vertical and diagonal <= horizontal:
                best, k, s = diagonal, 0, steps[i - 1][j - 1]
            elif vertical <= horizontal:
                best, k, s = vertical, 1, steps[i - 1][j]
            else:
                best, k, s = horizontal, 2, steps[i][j - 1]
            acc[i][j], steps[i][j], move[i][j] = cost[i - 1][j - 1] + best, s + 1, k
    if not math.isfinite(acc[n][m]):
        raise ValueError("no warping path inside the band")
    path, i, j = [], n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = ((i - 1, j - 1), (i - 1, j), (i, j - 1))[move[i][j]]
    return float(acc[n][m] / steps[n][m]), path[::-1]
