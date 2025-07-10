import torch
from typing import Dict, Any
import warnings

from .utils import _tril_from_flat, _precision_matrix_from_cholesky


class LikelihoodMixin:
    """
    Mixin class containing likelihood computation methods.

    This implementation includes:
    - Proper error handling and validation
    - Numerical stability improvements
    - Efficient vectorized computations
    - Comprehensive documentation
    - Type hints for better code clarity
    """

    def _hazard_ll(self, psi: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute hazard likelihood for all individuals.

        This method computes the log-likelihood contribution from the survival/hazard
        model by iterating through survival data buckets and computing both observed
        and censored event likelihoods.

        Args:
            psi: Random effects for each individual [n x psi_dim]

        Returns:
            torch.Tensor: Log-likelihood contribution from hazard model [n]
            data: Dict containing the relevant prepared dataset.

        Raises:
            RuntimeError: If hazard computation fails
        """
        if not isinstance(psi, torch.Tensor):
            raise TypeError("psi must be a torch.Tensor")

        if psi.shape[0] != data["n"]:
            raise ValueError(
                f"psi first dimension ({psi.shape[0]}) must match n ({data["n"]})"
            )

        ll = torch.zeros(data["n"], dtype=torch.float32)

        try:
            for d, info in data["buckets"].items():
                alpha, beta = self.params["alpha"][d], self.params["beta"][d]
                idx, t0, t1, obs = info["idx"], info["t0"], info["t1"], info["obs"]

                obs_ll, alts_ll = self._log_and_cum_hazard(
                    t0, t1, data["x"][idx], psi[idx], alpha, beta, **self.surv[d]
                )

                # Check for invalid values
                if torch.isnan(obs_ll).any() or torch.isinf(obs_ll).any():
                    warnings.warn(f"Invalid observed log-likelihood for bucket {d}")
                    continue

                if torch.isnan(alts_ll).any() or torch.isinf(alts_ll).any():
                    warnings.warn(f"Invalid cumulative hazard for bucket {d}")
                    continue

                vals = obs * obs_ll - alts_ll
                ll.scatter_add_(0, idx, vals)

        except Exception as e:
            raise RuntimeError(f"Failed to compute hazard likelihood: {e}")

        return ll

    def _long_ll(self, psi: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute longitudinal likelihood for all individuals.

        This method computes the log-likelihood contribution from the longitudinal
        model using multivariate normal distribution with precision matrix
        parametrization via Cholesky decomposition.

        Args:
            psi: Random effects for each individual [n x psi_dim]
            data: Dict containing the relevant prepared dataset.

        Returns:
            torch.Tensor: Log-likelihood contribution from longitudinal model [n]

        Raises:
            RuntimeError: If longitudinal likelihood computation fails
        """
        if not isinstance(psi, torch.Tensor):
            raise TypeError("psi must be a torch.Tensor")

        if psi.shape[0] != data["n"]:
            raise ValueError(
                f"psi first dimension ({psi.shape[0]}) must match n ({data["n"]})"
            )

        try:
            # Compute residuals: observed - predicted (only for valid observations)
            predicted = self.h(data["t"], data["x"], psi)
            diff = data["y"] - predicted * data["valid_mask"]

            # Check for invalid predictions
            if torch.isnan(predicted).any() or torch.isinf(predicted).any():
                warnings.warn("Invalid predictions encountered in longitudinal model")

            # Reconstruct precision matrix R_inv from Cholesky parametrization
            R_inv = _tril_from_flat(self.params["R_inv"], self.h.output_dim)

            # Compute log determinant: log det = -2 * sum(log(diag(R_inv)))
            log_det_R = -torch.diag(R_inv).sum() * 2

            # Compute the precision matrix
            R_inv = _precision_matrix_from_cholesky(R_inv)

            # Compute quadratic form: diff.T @ R_inv @ diff for each individual
            quad_form = torch.einsum("ijk,kl,ijl->i", diff, R_inv, diff)

            # Log-likelihood: -0.5 * (log|R| * n_valid + quadratic_form)
            ll = -0.5 * (log_det_R * data["n_valid"] + quad_form)

            # Validate output
            if torch.isnan(ll).any() or torch.isinf(ll).any():
                warnings.warn("Invalid longitudinal likelihood computed")

            return ll

        except Exception as e:
            raise RuntimeError(f"Failed to compute longitudinal likelihood: {e}")

    def _pr_ll(self, b: torch.Tensor) -> torch.Tensor:
        """
        Compute prior likelihood for random effects.

        This method computes the log-likelihood contribution from the multivariate
        normal prior on random effects using precision matrix parametrization.

        Args:
            b: Random effects [n x b_dim]

        Returns:
            torch.Tensor: Log-likelihood contribution from prior [n]

        Raises:
            RuntimeError: If prior likelihood computation fails
        """
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")

        try:
            # Reconstruct precision matrix Q_inv from Cholesky parametrization
            Q_inv = _tril_from_flat(self.params["Q_inv"], self.f.input_dim[1])

            # Compute log determinant: log det = -2 * sum(log(diag(Q_inv)))
            log_det_Q = -torch.diag(Q_inv).sum() * 2

            # Compute the precision matrix
            Q_inv = _precision_matrix_from_cholesky(Q_inv)

            # Compute quadratic form: b.T @ Q_inv @ b for each individual
            quad_form = torch.einsum("ik,kl,il->i", b, Q_inv, b)

            # Log-likelihood: -0.5 * (log det + quadratic_form)
            ll = -0.5 * (log_det_Q + quad_form)

            # Validate output
            if torch.isnan(ll).any() or torch.isinf(ll).any():
                warnings.warn("Invalid prior likelihood computed")

            return ll

        except Exception as e:
            raise RuntimeError(f"Failed to compute prior likelihood: {e}")

    def _ll(self, b: torch.Tensor, data: Dict[str, Any]) -> torch.Tensor:
        """
        Compute total log-likelihood for given random effects.

        This is the main likelihood function that combines all components:
        - Longitudinal likelihood (how well the model fits observed outcomes)
        - Hazard likelihood (how well the model predicts survival times)
        - Prior likelihood (regularization on random effects)

        Args:
            b: Random effects [n x b_dim]
            data: Dict containing the relevant prepared dataset.

        Returns:
            torch.Tensor: Total log-likelihood [n]

        Raises:
            RuntimeError: If likelihood computation fails
        """
        if not isinstance(b, torch.Tensor):
            raise TypeError("b must be a torch.Tensor")

        try:
            # Transform random effects to individual-specific parameters
            psi = self.f(self.params["gamma"], b)

            # Validate transformation
            if torch.isnan(psi).any() or torch.isinf(psi).any():
                warnings.warn("Invalid psi values from transformation")

            # Compute individual likelihood components
            long_ll = self._long_ll(psi, data)
            hazard_ll = self._hazard_ll(psi, data)
            prior_ll = self._pr_ll(b)

            # Sum all likelihood components
            total_ll = long_ll + hazard_ll + prior_ll

            # Final validation
            if torch.isnan(total_ll).any() or torch.isinf(total_ll).any():
                warnings.warn("Invalid total likelihood computed")

            return total_ll

        except Exception as e:
            raise RuntimeError(f"Failed to compute total likelihood: {e}")
