import itertools
from collections import defaultdict
from typing import Any, DefaultDict

import torch

from .types import *


def flat_from_tril(L: torch.Tensor) -> torch.Tensor:
    """Flatten the lower triangular part (including the diagonal) of a square matrix L
    into a 1D tensor, in row-wise order.

    Args:
        L (torch.Tensor): Square lower-triangular matrix of shape (n, n).

    Raises:
        ValueError: If the input is not square.
        RuntimeError: If the flattening fails.

    Returns:
        torch.Tensor: Flattened 1D tensor containing the lower triangular entries.
    """

    try:
        if L.ndim != 2 or L.shape[0] != L.shape[1]:
            raise ValueError("Input must be a square matrix")

        n = L.shape[0]
        i, j = torch.tril_indices(n, n)

        return L[i, j]

    except Exception as e:
        raise RuntimeError(f"Failed to flatten matrix: {e}") from e


def build_vec_rep(
    trajectories: list[Trajectory],
    c: torch.Tensor,
    surv: dict[tuple[int, int], tuple[BaseHazardFn, LinkFn]],
) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
    """Build vectorizable bucket representation.

    Args:
        trajectories (list[Trajectory]): The trajectories.
        c (torch.Tensor): Censoring times.
        surv (dict[tuple[int, int], tuple[BaseHazardFn, LinkFn]]) : The model survival dict.

    Raises:
        ValueError: If some keys are not in surv.
        RuntimeError: If the building fails.

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: The vectorizable buckets representation.
    """

    try:
        # Get survival transitions defined in the model
        trans = set(surv.keys())

        # Build alternative state mapping
        alt_map: DefaultDict[int, list[int]] = defaultdict(list)
        for from_state, to_state in trans:
            alt_map[from_state].append(to_state)

        # Initialize buckets
        buckets: DefaultDict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], [], []]
        )

        # Process each individual trajectory
        for i, trajectory in enumerate(trajectories):
            # Add censoring
            ext_trajectory = trajectory + [(float(c[i]), None)]

            for (t0, s0), (t1, s1) in itertools.pairwise(ext_trajectory):
                if t0 >= t1:
                    continue

                if s1 is not None and (s0, s1) not in trans:
                    raise ValueError(
                        f"Transition {(s0, s1)} must be in model_design.surv keys"
                    )

                for alt_state in alt_map.get(s0, []):
                    key = (s0, alt_state)
                    buckets[key][0].append(i)
                    buckets[key][1].append(t0)
                    buckets[key][2].append(t1)
                    buckets[key][3].append(alt_state == s1)

        processed_buckets: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {
            key: (
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
                torch.tensor(vals[3], dtype=torch.bool),
            )
            for key, vals in buckets.items()
            if vals[0]
        }

        return processed_buckets

    except Exception as e:
        raise RuntimeError(f"Error building survival buckets: {e}") from e
