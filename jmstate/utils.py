import itertools
from math import isqrt
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Callable, DefaultDict, TypeAlias

import torch


# Type Aliases
RegressionFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
LinkFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
IndividualEffectsFn: TypeAlias = Callable[
    [torch.Tensor | None, torch.Tensor], torch.Tensor
]
BaseHazardFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Trajectory: TypeAlias = list[tuple[float, Any]]
ClockMethod: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class ModelDesign:
    """Class containing all multistate joint model design.

    Raises:
        TypeError: If f is not callable.
        TypeError: If h is not callable.
        TypeError: If any of the base hazard functions is not callable.
        TypeError: If any of the link functions is not callable.
        ValueError: If the keys of alpha_dims and surv do not match.
    """

    f: IndividualEffectsFn
    h: RegressionFn
    surv: dict[
        tuple[int, int],
        tuple[
            BaseHazardFn,
            LinkFn,
        ],
    ]

    def __post_init__(self):
        """Runs the checks themselves.

        Raises:
            TypeError: If f is not callable.
            TypeError: If h is not callable.
            TypeError: If any of the base hazard functions is not callable.
            TypeError: If any of the link functions is not callable.
            ValueError: If the keys of alpha_dims and surv do not match.
        """
        if not callable(self.f):
            raise TypeError("f must be callable")
        if not callable(self.h):
            raise TypeError("h must be callable")

        for key, (base_fn, link_fn) in self.surv.items():
            if not callable(base_fn):
                raise TypeError(f"Base hazard function for key {key} must be callable")
            if not callable(link_fn):
                raise TypeError(f"Link function for key {key} must be callable")


@dataclass
class ModelData:
    """Dataclass containing learnable multistate joint model data.

    Raises:
        ValueError: If any tensor contains inf values.
        ValueError: If c is not 1D.
        ValueError: If x is not 2D.
        ValueError: If y is not 3D.
        ValueError: If the number of individuals is inconsistent.
        ValueError: If the shape of t is not broadcastable with y.
        ValueError: If t contains torch.nan where y is not.
        ValueError: If the trajectories are not sorted by time.

    Returns:
        _type_: The instance.
    """

    x: torch.Tensor | None
    t: torch.Tensor
    y: torch.Tensor
    trajectories: list[Trajectory]
    c: torch.Tensor
    valid_t_: torch.Tensor = field(init=False, repr=False)
    valid_y_: torch.Tensor = field(init=False, repr=False)
    valid_mask_: torch.Tensor = field(init=False, repr=False)
    n_valid_: torch.Tensor = field(init=False, repr=False)
    buckets_: dict[tuple[int, int], tuple[torch.Tensor, ...]] = field(
        init=False, repr=False
    )

    def __post_init__(self):
        """Runs the post init conversions and checks."""

        # Convert to float32
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.t = torch.as_tensor(self.t, dtype=torch.float32)
        self.y = torch.as_tensor(self.y, dtype=torch.float32)
        self.c = torch.as_tensor(self.c, dtype=torch.float32)

        self._check()

    def _check(self):
        """Validate tensor dimensions and consistency.

        Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D.
            ValueError: If x is not 2D or None.
            ValueError: If y is not 3D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the shape of t is not broadcastable with y.
            ValueError: If t contains torch.nan where y is not.
            ValueError: If the trajectories are not sorted by time.
        """

        # Check for inf tensors
        for name, tensor in [
            ("c", self.c),
            ("x", self.x),
            ("y", self.y),
            ("t", self.t),
        ]:
            if tensor is not None and tensor.isinf().any():
                raise ValueError(f"{name} cannot contain inf values")

        # Check dimensions
        if self.c.ndim != 1:
            raise ValueError(f"c must be 1D, got {self.c.ndim}D")
        if self.x is not None and self.x.ndim != 2:
            raise ValueError(f"x must be None or 2D, got {self.x.ndim}D")
        if self.y.ndim != 3:
            raise ValueError(f"y must be 3D, got {self.y.ndim}D")

        # Check consistent size
        n = self.size
        if not (
            (self.x is None or self.x.shape[0] == n)
            and self.y.shape[0] == self.c.numel() == n
        ):
            raise ValueError("Inconsistent number of individuals")

        # Check time compatibility
        if self.t.shape not in ((self.y.shape[1],), self.y.shape[:2]):
            raise ValueError(f"Invalid t shape: {self.t.shape}")

        # Check for NaNs in t where y is valid
        if (
            self.t.shape == self.y.shape[:2]
            and (~self.y.isnan().all(dim=2) & self.t.isnan()).any()
        ):
            raise ValueError("t cannot be NaN where y is valid")

        # Check trajectory sorting
        if any(
            any(t0 > t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
            for trajectory in self.trajectories
        ):
            raise ValueError("Trajectories must be sorted by time")

        # Check if c is at least equal to the greatest part of each trajectory
        if any(
            trajectory[-1][0] > c for trajectory, c in zip(self.trajectories, self.c)
        ):
            raise ValueError("Last trajectory time must not be greater than c")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)


@dataclass
class SampleData:
    """Dataclass for data used in sampling.

    Raises:
        ValueError: If any tensor contains inf values.
        ValueError: If c is not 1D or None.
        ValueError: If x is not 2D.
        ValueError: If psi is not 2D.
        ValueError: If the number of individuals is inconsistent.
        ValueError: If the trajectories are not sorted by time.
        ValueError: If the last trajectory time is greater than c

    Returns:
        _type_: The instance.
    """

    x: torch.Tensor | None
    trajectories: list[Trajectory]
    psi: torch.Tensor
    c: torch.Tensor | None = None

    def __post_init__(self):
        """Runs the post init conversions and checks."""

        # Convert to float32
        self.x = (
            torch.as_tensor(self.x, dtype=torch.float32) if self.x is not None else None
        )
        self.c = (
            torch.as_tensor(self.c, dtype=torch.float32) if self.c is not None else None
        )
        self.psi = torch.as_tensor(self.psi, dtype=torch.float32)

        self._check()

    def _check(self):
        """Validate tensor dimensions and consistency.

        Raises:
            ValueError: If any tensor contains inf values.
            ValueError: If c is not 1D or None.
            ValueError: If x is not 2D.
            ValueError: If psi is not 2D.
            ValueError: If the number of individuals is inconsistent.
            ValueError: If the trajectories are not sorted by time.
            ValueError: If the last trajectory time is greater than c
        """

        # Check for inf tensors
        for name, tensor in [("c", self.c), ("x", self.x), ("psi", self.psi)]:
            if tensor is not None and tensor.isinf().any():
                raise ValueError(f"{name} cannot contain inf values")

        # Check dimensions
        if self.c is not None and self.c.ndim != 1:
            raise ValueError(f"c must be 1D, got {self.c.ndim}D")
        if self.x is not None and self.x.ndim != 2:
            raise ValueError(f"x must be None or 2D, got {self.x.ndim}D")
        if self.psi.ndim != 2:
            raise ValueError(f"psi must be 2D, got {self.psi.ndim}D")

        # Check consistent size
        n = self.size
        if not (
            (self.x is None or self.x.shape[0] == n)
            and self.psi.shape[0] == n
            and (self.c is None or self.c.numel() == n)
        ):
            raise ValueError("Inconsistent number of individuals")

        # Check trajectory sorting
        if any(
            any(t0 > t1 for t0, t1 in itertools.pairwise(t for t, _ in trajectory))
            for trajectory in self.trajectories
        ):
            raise ValueError("Trajectories must be sorted by time")

        # Check if c is at least equal to the greatest part of each trajectory
        if self.c is not None:
            if any(
                trajectory[-1][0] > c
                for trajectory, c in zip(self.trajectories, self.c)
            ):
                raise ValueError("Last trajectory time must not be greater than c")

    @property
    def size(self) -> int:
        """Gets the number of individuals.

        Returns:
            int: The number of individuals.
        """
        return len(self.trajectories)


@dataclass
class ModelParams:
    """Dataclass containing model parameters.

    Raises:
        ValueError: If any of the main tensors contains inf.
        ValueError: If any of the main tensors is not 1D.
        ValueError: If any of the alpha tensors contains inf.
        ValueError: If any of the alpha tensors is not 1D.
        ValueError: If any of the beta tensors contains inf.
        ValueError: If any of the beta tensors is not 1D.
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the number of elements is not a triangular number and the method is "full".
        ValueError: If the number of elements is not one and the method is "ball".
        ValueError: If the name matrix is not "Q" nor "R".
        ValueError: If the name matrix is not "Q" nor "R".

    Returns:
        _type_: The instance.
    """

    gamma: torch.Tensor | None
    Q_repr: tuple[torch.Tensor, str]
    R_repr: tuple[torch.Tensor, str]
    alphas: dict[tuple[int, int], torch.Tensor]
    betas: dict[tuple[int, int], torch.Tensor] | None
    Q_dim_: int = field(init=False, repr=False)
    R_dim_: int = field(init=False, repr=False)

    def __post_init__(self):
        """Convert and init to float32 the parameters."""

        # Convert components to float32
        Q_flat, Q_method = self.Q_repr
        R_flat, R_method = self.R_repr

        Q_flat = torch.as_tensor(Q_flat, dtype=torch.float32)
        R_flat = torch.as_tensor(R_flat, dtype=torch.float32)

        # Update representation tuples
        self.Q_repr = (Q_flat, Q_method)
        self.R_repr = (R_flat, R_method)

        # Convert the rest to float32
        self.gamma = (
            torch.as_tensor(self.gamma, dtype=torch.float32)
            if self.gamma is not None
            else None
        )

        for key, alpha in self.alphas.items():
            self.alphas[key] = torch.as_tensor(alpha, dtype=torch.float32)

        if self.betas is not None:
            for key, beta in self.betas.items():
                self.betas[key] = torch.as_tensor(beta, dtype=torch.float32)

        self._check()
        self._set_dims("Q")
        self._set_dims("R")

    def _check(self):
        """Validate all tensors are 1D and don't contain inf.

        Raises:
            ValueError: If any of the main tensors contains inf.
            ValueError: If any of the main tensors is not 1D.
            ValueError: If any of the alpha tensors contains inf.
            ValueError: If any of the alpha tensors is not 1D.
            ValueError: If any of the beta tensors contains inf.
            ValueError: If any of the beta tensors is not 1D.
        """

        # Check main tensors
        for name, tensor in [
            ("gamma", self.gamma),
            ("Q_flat_", self.Q_repr[0]),
            ("R_flat_", self.R_repr[0]),
        ]:
            if tensor is None:
                continue
            if tensor.isinf().any():
                raise ValueError(f"{name} contains inf")
            if tensor.ndim != 1:
                raise ValueError(f"{name} must be 1D")

        # Check dictionary tensors
        for key, alpha in self.alphas.items():
            if alpha.isinf().any():
                raise ValueError(f"alpha {key} contains inf")
            if alpha.ndim != 1:
                raise ValueError(f"alpha {key} must be 1D")

        if self.betas is not None:
            for key, beta in self.betas.items():
                if beta.isinf().any():
                    raise ValueError(f"beta {key} contains inf")
                if beta.ndim != 1:
                    raise ValueError(f"beta {key} must be 1D")

    def _set_dims(self, matrix: str) -> None:
        """Sets dimensions for matrix.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the name matrix is not "Q" nor "R".
            ValueError: If the number of elements is not a triangular number and the method is "full".
            ValueError: If the number of elements is not one and the method is "ball".
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        flat, method = getattr(self, matrix + "_repr")

        match method:
            case "full":
                n = (isqrt(1 + 8 * flat.numel()) - 1) // 2
                if (n * (n + 1)) // 2 != flat.numel():
                    raise ValueError(
                        f"{flat.numel()} is not a triangular number for matrix {matrix}"
                    )
                setattr(self, matrix + "_dim_", n)
            case "diag":
                n = flat.numel()
                setattr(self, matrix + "_dim_", n)
            case "ball":
                if 1 != flat.numel():
                    f"Inocrrect number of elements for flat, got {flat.numel()} but expected {1}"
                setattr(self, matrix + "_dim_", 1)
            case _:
                raise ValueError(f"Got method {method} unknown for matrix {matrix}")

    @property
    def as_list(self) -> list[torch.Tensor]:
        """Get a list of all the parameters for optimization.

        Returns:
            list[torch.Tensor]: The list of the parameters.
        """

        iterables: Iterable[torch.Tensor | list[torch.Tensor]] = (
            [self.gamma] if self.gamma is not None else [],
            [self.Q_repr[0], self.R_repr[0]],
            list(self.alphas.values()),
            list(self.betas.values()) if self.betas is not None else [],
        )

        return list(itertools.chain.from_iterable(iterables))

    @property
    def numel(self) -> int:
        """Return the number of parameters.

        Returns:
            int: The number of the parameters.
        """

        return sum(p.numel() for p in self.as_list)

    def get_cholesky(self, matrix: str) -> torch.Tensor:
        """Get Cholesky factor.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            torch.Tensor: The precision matrix.
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        # Get flat then log cholesky
        flat, method = getattr(self, matrix + "_repr")
        n = getattr(self, matrix + "_dim_")

        L = log_cholesky_from_flat(flat, n, method)
        L.diagonal().exp_()

        return L

    def get_cholesky_and_log_eigvals(
        self, matrix: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get Cholesky factor as well as log eigenvalues.

        Args:
            matrix (str): Either "Q" or "R".

        Raises:
            ValueError: If the matrix is not in ("Q", "R")

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The tuple of precision matrix and log eigenvalues of precision.
        """

        if not matrix in ("Q", "R"):
            raise ValueError(f"matrix should be either Q or R, got {matrix}")

        # Get flat then log cholesky
        flat, method = getattr(self, matrix + "_repr")
        n = getattr(self, matrix + "_dim_")

        L = log_cholesky_from_flat(flat, n, method)
        eigvals = 2 * L.diagonal()
        L.diagonal().exp_()

        return L, eigvals

    def require_grad(self, req: bool):
        """Enable gradient computation on all parameters.

        Args:
            req (bool): Wether to require or not.
        """

        # Enable or diasable gradients
        for tensor in self.as_list:
            tensor.requires_grad_(req)


def _tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """Generate the lower triangular matrix associated with flat tensor.

    Args:
        flat (torch.Tensor): Flat tehsnro
        n (int): Dimension of the matrix.

    Raises:
        ValueError: Error if the the dimensions do not allow matrix computation.
        RuntimeError: Error if the computation fails.

    Returns:
        torch.Tensor: The lower triangular matrix.
    """

    if flat.numel() != (n * (n + 1)) // 2:
        raise ValueError("Incompatible dimensions for lower triangular matrix")

    L = torch.zeros(n, n, dtype=flat.dtype).index_put_(
        tuple(torch.tril_indices(n, n)), flat
    )

    return L


def _flat_from_tril(L: torch.Tensor) -> torch.Tensor:
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


def log_cholesky_from_flat(
    flat: torch.Tensor, n: int, method: str = "full"
) -> torch.Tensor:
    """Computes log cholesky from flat tensor according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        n (int): The dimension of the matrix.
        method (str, optional): The method, either for full, diagonal or isotropic covariance matrix. Defaults to "full".

    Raises:
        ValueError: If the array is not flat.
        ValueError: If the number of parameters is inconsistent with n.
        ValueError: If the number of parameters does not equal one.
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The log cholesky representation.
    """

    if flat.ndim != 1:
        raise ValueError(f"flat should be flat, got shape {flat.shape}")

    match method:
        case "full":
            return _tril_from_flat(flat, n)
        case "diag":
            if flat.numel() != n:
                raise ValueError(
                    f"Inocrrect number of elements for flat, got {flat.numel()} but expected {n}"
                )
            return torch.diag(flat)
        case "ball":
            if flat.numel() != 1:
                f"Inocrrect number of elements for flat, got {flat.numel()} but expected {1}"
            return flat * torch.eye(n)
        case _:
            raise ValueError(f"Got method {method} unknown")


def flat_from_log_cholesky(L: torch.Tensor, method: str = "full") -> torch.Tensor:
    """Computes flat tensor from log cholesky matrix according to choice of method.

    Args:
        flat (torch.Tensor): The flat tensor parameter.
        method (str, optional): The method, either for full, diagonal or isotropic covariance matrix. Defaults to "full".

    Raises:
        ValueError: If the method is not in ("full", "diag", "ball").

    Returns:
        torch.Tensor: The flat representation.
    """

    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")

    match method:
        case "full":
            return _flat_from_tril(L)
        case "diag":
            return L.diagonal()
        case "ball":
            return L[0, 0]
        case _:
            raise ValueError(f"Got method {method} unknown")


def build_buckets(
    trajectories: list[Trajectory],
) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
    """Builds buckets from trajectories for user convenience.

    Args:
        trajectories (list[Trajectory]): The list of individual trajectories.

    Raises:
        RuntimeError: If the construction of the buckets fails.

    Returns:
        dict[tuple[int, int], tuple[torch.Tensor, ...]]: A dictionnary of transition keys with a triplet of tensors (idxs, t0, t1).
    """

    try:
        # Process each individual trajectory
        buckets: DefaultDict[tuple[int, int], list[list[Any]]] = defaultdict(
            lambda: [[], [], []]
        )

        for i, trajectory in enumerate(trajectories):
            for (t0, s0), (t1, s1) in itertools.pairwise(trajectory):
                key = (s0, s1)
                buckets[key][0].append(i)
                buckets[key][1].append(t0)
                buckets[key][2].append(t1)

        processed_buckets = {
            key: (
                torch.tensor(vals[0], dtype=torch.int64),
                torch.tensor(vals[1], dtype=torch.float32),
                torch.tensor(vals[2], dtype=torch.float32),
            )
            for key, vals in buckets.items()
            if vals[0]  # skip empty
        }

        return processed_buckets

    except Exception as e:
        raise RuntimeError(f"Failed to construct buckets: {e}") from e
