import numpy as np
from numpy.typing import NDArray

from ._methods import *


def get_bands(
    F_lo: NDArray[np.float32 | np.float64],
    F_hi: NDArray[np.float32 | np.float64],
    alpha: float = 0.05,
    *,
    eps: float = 1e-8,
    method: str = "uniform",
    min_val: float = 0.0,
    max_val: float = 1.0,
) -> dict[str, NDArray[np.float32 | np.float64]]:

    if F_lo.shape != F_hi.shape:
        raise ValueError(
            f"F_lo and F_hi shapes must be the same, got {F_lo.shape} != {F_hi.shape}"
        )

    if F_lo.ndim != 2:
        raise ValueError(f"F_lo and F_hi must be matrices, got shape {F_lo.shape}")

    if method not in ("uniform", "student"):
        raise ValueError(f"Method should be in ('uniform', 'student'), got {method}")

    if not 0 < alpha <= 1:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")

    if method == "uniform":
        return uniform(F_lo, F_hi, alpha)

    if eps <= 0:
        raise ValueError(
            f"eps must be strictly positive for the student method, got {eps}"
        )

    return student(F_lo, F_hi, alpha, eps, min_val, max_val)
