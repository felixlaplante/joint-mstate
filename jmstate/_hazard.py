from typing import Any, cast

import torch

from .types import BaseHazardFn, LinkFn
from ._utils import legendre_quad


class HazardMixin:
    """Mixin class for hazard model computations."""

    def __init__(self, n_quad: int):
        self._std_nodes, self._std_weights = legendre_quad(n_quad)

        self._cache: dict[str, dict[Any, torch.Tensor]] = {"quad": {}, "base": {}}

    def clear_cache(self) -> None:
        """Clears the cached tensors"""

        self._cache = {"quad": {}, "base": {}}

    def _log_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor | None,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor | None,
        log_lambda0: BaseHazardFn,
        g: LinkFn,
    ) -> torch.Tensor:
        """Computes log hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseHazardFn): Base hazard function.
            g (LinkFn): Link function.

        Returns:
            torch.Tensor: The computed log hazard.
        """

        # Compute baseline hazard
        key = (id(log_lambda0), id(t0.untyped_storage()), id(t1.untyped_storage()))
        try:
            base = self._cache["base"][key]
        except:
            base = log_lambda0(t0, t1)
            self._cache["base"][key] = base

        # Compute time-varying effects
        mod = g(t1, x, psi) @ alpha

        # Compute covariates effect
        cov = (
            x @ beta.unsqueeze(1)
            if x is not None and beta is not None
            else torch.tensor(0.0, dtype=torch.float32)
        )

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
        log_lambda0: BaseHazardFn,
        g: LinkFn,
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
            log_lambda0 (BaseHazardFn): Base hazard function.
            g (LinkFn): Link function.

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
        half = 0.5 * (t1 - c)

        key = (id(c.untyped_storage()), id(t1.untyped_storage()))
        try:
            ts = self._cache["quad"][key]
        except:
            mid = 0.5 * (c + t1)

            # Evaluate at quadrature points
            ts = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))
            self._cache["quad"][key] = ts

        # Compute hazard at quadrature points
        log_hazard_vals = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)

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
        log_lambda0: BaseHazardFn,
        g: LinkFn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes both log and cumulative hazard.

        Args:
            t0 (torch.Tensor): Start time.
            t1 (torch.Tensor): End time.
            x (torch.Tensor | None): Covariates.
            psi (torch.Tensor): Inidivual parameters.
            alpha (torch.Tensor): Link linear parameters.
            beta (torch.Tensor): Covariate linear parameters.
            log_lambda0 (BaseHazardFn): Base hazard function.
            g (LinkFn): Link function.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing log and cumulative hazard.
        """

        # Reshape for broadcasting
        t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

        # Transform to quadrature interval
        half = 0.5 * (t1 - t0)

        key = (id(t0.untyped_storage()), id(t1.untyped_storage()))

        try:
            ts = self._cache["quad"][key]
        except:
            # Combine endpoint and quadrature points
            mid = 0.5 * (t0 + t1)

            ts = torch.addmm(mid, half.view(-1, 1), self._std_nodes.view(1, -1))
            self._cache["quad"][key] = ts

        # Combine with t1
        ts = torch.cat([t1, ts], dim=1)

        # Compute log hazard at all points
        temp = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)

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
        log_lambda0: BaseHazardFn,
        g: LinkFn,
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
            log_lambda0 (BaseHazardFn): Base hazard function.
            g (LinkFn): Link function.
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
                t0, t_mid, c, x, psi, alpha, beta, log_lambda0, g
            )

            # Update search bounds
            accept_mask = cumulative < target
            t_left[accept_mask] = t_mid[accept_mask]
            t_right[~accept_mask] = t_mid[~accept_mask]

        return t_right.view(-1)
