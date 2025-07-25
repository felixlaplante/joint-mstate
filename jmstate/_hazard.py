from typing import Any

import torch

from .types import BaseHazardFn, LinkFn
from ._utils import legendre_quad


_EMPTY_CACHE: dict[str, dict[Any, torch.Tensor]] = {"half": {}, "quad": {}, "base": {}}


class HazardMixin:
    """Mixin class for hazard model computations."""

    def __init__(self, n_quad: int, cache_limit: int):
        """Initializes the class.

        Args:
            n_quad (int): Number of quadrature nodes.
            cache_limit (int): Max length of cache.
        """

        self.n_quad = n_quad
        self.cache_limit = cache_limit

        self._std_nodes, self._std_weights = legendre_quad(n_quad)
        self._cache = _EMPTY_CACHE

    def _get_cache(self, name: str, key: Any) -> torch.Tensor:
        """Gets the cache [name][key]

        Args:
            name (str): The name of the cache info.
            key (Any): The key inside the current info.

        Returns:
            torch.Tensor: The cached tensor.
        """
        return self._cache[name][key]

    def _add_cache(self, name: str, key: Any, val: torch.Tensor) -> None:
        """Add value to cache if not exceeding the limit.

        Args:
            name (str): name (str): The name of the cache info.
            key (Any): The key inside the current info.
            val (torch.Tensor): The tensor to cache.
        """

        if len(self._cache[name]) <= self.cache_limit:
            self._cache[name][key] = val

    def clear_cache(self) -> None:
        """Clears the cached tensors"""

        self._cache = _EMPTY_CACHE

    def _log_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
    ) -> torch.Tensor:
        """Computes log hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.

        Returns:
            torch.Tensor: The computed log hazard.
        """

        # Compute baseline hazard
        key = (
            id(log_base_hazard_fn),
            id(t0.untyped_storage()),
            id(t1.untyped_storage()),
        )
        try:
            base = self._get_cache("base", key)
        except:
            base = log_base_hazard_fn(t0, t1)
            self._add_cache("base", key, base)

        # Compute time-varying effects
        mod = link(t1, x, psi) @ alpha

        # Compute covariates effect
        cov = torch.tensor(0.0, dtype=torch.float32)
        if x is not None and beta is not None:
            cov = x @ beta.unsqueeze(1)

        # Compute the total
        log_hazard_vals = base + mod + cov

        return log_hazard_vals

    def _cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        c: torch.Tensor | None,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
    ) -> torch.Tensor:
        """Computes cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            c (torch.Tensor | None): Integration start or censoring time, if None, means t0.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.

        Returns:
            torch.Tensor: The computed cumulative hazard.
        """

        # Reshape for broadcasting
        t0, t1, c = (
            t0.view(-1, 1),
            t1.view(-1, 1),
            c.view(-1, 1) if c is not None else t0.view(-1, 1),
        )

        # Transform to quadrature interval
        key = (id(c.untyped_storage()), id(t1.untyped_storage()))
        try:
            half = self._get_cache("half", key)
            quad = self._get_cache("quad", key)

        except:
            half = 0.5 * (t1 - c)
            self._add_cache("half", key, half)

            # Evaluate at quadrature points
            mid = 0.5 * (c + t1)
            quad = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))
            self._add_cache("quad", key, quad)

        # Compute hazard at quadrature points
        log_hazard_vals = self._log_hazard(
            t0, quad, x, psi, alpha, beta, log_base_hazard_fn, link
        )

        # Numerical integration using Gaussian quadrature
        hazard_vals = torch.exp(torch.clamp(log_hazard_vals, min=-50.0, max=50.0))

        cum_hazard_vals = half.view(-1) * (hazard_vals @ self._std_weights)

        return cum_hazard_vals

    def _log_and_cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing log and cumulative hazard.
        """

        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

        # Transform to quadrature interval
        key = (id(t0.untyped_storage()), id(t1.untyped_storage()))
        try:
            half = self._get_cache("half", key)
            quad = self._get_cache("quad", key)

        except:
            half = 0.5 * (t1 - t0)
            self._add_cache("half", key, half)

            # Evaluate at quadrature points
            mid = 0.5 * (t0 + t1)
            quad = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))
            self._add_cache("quad", key, quad)

        # Combine with t1
        t1_and_quad = torch.cat([t1, quad], dim=1)

        # Compute log hazard at all points
        temp = self._log_hazard(
            t0, t1_and_quad, x, psi, alpha, beta, log_base_hazard_fn, link
        )

        # Extract log hazard at endpoint and quadrature points
        log_hazard_vals = temp[:, :1]  # Log hazard at t1
        hazard_vals = torch.exp(
            torch.clamp(temp[:, 1:], min=-50.0, max=50.0)
        )  # Hazard at quadrature points

        # Compute cumulative hazard using quadrature
        cum_hazard_vals = half.view(-1) * (hazard_vals @ self._std_weights)

        return log_hazard_vals.view(-1), cum_hazard_vals

    def _sample_trajectory_step(
        self,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_base_hazard_fn: BaseHazardFn,
        link: LinkFn,
        *,
        c: torch.Tensor | None = None,
        n_bissect: int,
    ) -> torch.Tensor:
        """Sample survival times using inverse transform sampling.

        Args:
            t_left (torch.Tensor): Left sampling time.
            t_right (torch.Tensor): Right censoring sampling time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_base_hazard_fn (BaseHazardFn): Base hazard function.
            link (LinkFn): Link function.
            n_bissect (int): _description_
            c (torch.Tensor | None, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: The computed pre transition times.
        """

        n = psi.shape[0]

        # Initialize for bisection search
        t0 = t_left.clone().view(-1, 1)
        t_left, t_right = t_left.view(-1, 1), t_right.view(-1, 1)

        # Generate exponential random variables
        target = -torch.log(torch.clamp(torch.rand(n), min=1e-8))

        # Bisection search for survival times
        for _ in range(n_bissect):
            t_mid = 0.5 * (t_left + t_right)

            cumulative = self._cum_hazard(
                t0, t_mid, c, x, psi, alpha, beta, log_base_hazard_fn, link
            )

            # Update search bounds
            accept_mask = cumulative < target
            t_left[accept_mask] = t_mid[accept_mask]
            t_right[~accept_mask] = t_mid[~accept_mask]

        return t_right.view(-1)
