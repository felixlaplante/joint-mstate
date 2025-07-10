import torch
import warnings
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict


class HazardMixin:
    """
    Mixin class for hazard model computations.

    This implementation provides methods for computing hazard functions,
    cumulative hazards, and survival probabilities using numerical integration
    and efficient vectorized operations.
    """

    def _log_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: callable,
        g: callable,
    ) -> torch.Tensor:
        """
        Compute log hazard function.

        Args:
            t0: Start times [n x 1]
            t1: End times [n x k] where k is number of evaluation points
            x: Covariates [n x p]
            psi: Random effects [n x psi_dim]
            alpha: Hazard coefficients for g(t,x,psi)
            beta: Hazard coefficients for x
            log_lambda0: Baseline log hazard function
            g: Function g(t,x,psi) for hazard model

        Returns:
            torch.Tensor: Log hazard values [n x k]
        """
        try:
            # Validate inputs
            if not all(
                isinstance(t, torch.Tensor) for t in [t0, t1, x, psi, alpha, beta]
            ):
                raise TypeError("All inputs must be torch.Tensor")

            # Compute baseline hazard
            base = log_lambda0(t1, t0)

            # Compute time-varying effects
            mod = g(t1, x, psi)

            # Validate g output
            if torch.isnan(mod).any() or torch.isinf(mod).any():
                warnings.warn("Invalid values in g(t,x,psi)")

            # Compute log hazard
            log_hazard = (
                base + torch.einsum("ijk,k->ij", mod, alpha) + x @ beta.unsqueeze(1)
            )

            return log_hazard

        except Exception as e:
            raise RuntimeError(f"Error in log hazard computation: {e}")

    def _cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: callable,
        g: callable,
    ) -> torch.Tensor:
        """
        Compute cumulative hazard using Gaussian quadrature.

        Args:
            t0: Start times [n x 1]
            t1: End times [n x 1]
            x: Covariates [n x p]
            psi: Random effects [n x psi_dim]
            alpha: Hazard coefficients for g(t,x,psi)
            beta: Hazard coefficients for x
            log_lambda0: Baseline log hazard function
            g: Function g(t,x,psi) for hazard model

        Returns:
            torch.Tensor: Cumulative hazard values [n]
        """
        try:
            # Reshape for broadcasting
            t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

            # Transform to quadrature interval [-1, 1]
            mid = 0.5 * (t0 + t1)
            half = 0.5 * (t1 - t0)

            # Evaluate at quadrature points
            ts = mid + half * self.std_nodes

            # Compute hazard at quadrature points
            log_hazard_vals = self._log_hazard(
                t0, ts, x, psi, alpha, beta, log_lambda0, g
            )

            # Numerical integration using Gaussian quadrature
            hazard_vals = torch.exp(log_hazard_vals)

            # Check for numerical issues
            if torch.isnan(hazard_vals).any() or torch.isinf(hazard_vals).any():
                warnings.warn("Numerical issues in hazard computation")
                hazard_vals = torch.nan_to_num(hazard_vals, nan=0.0, posinf=1e10)

            cumulative = half.flatten() * (hazard_vals * self.std_weights).sum(dim=1)

            return cumulative

        except Exception as e:
            raise RuntimeError(f"Error in cumulative hazard computation: {e}")

    def _log_and_cum_hazard(
        self,
        t0: torch.Tensor,
        t1: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: callable,
        g: callable,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute both log hazard and cumulative hazard efficiently.

        This method is optimized for likelihood computation where both
        the observed log hazard and cumulative hazard are needed.

        Args:
            t0: Start times [n x 1]
            t1: End times [n x 1]
            x: Covariates [n x p]
            psi: Random effects [n x psi_dim]
            alpha: Hazard coefficients for g(t,x,psi)
            beta: Hazard coefficients for x
            log_lambda0: Baseline log hazard function
            g: Function g(t,x,psi) for hazard model

        Returns:
            Tuple of (log_hazard [n], cumulative_hazard [n])
        """
        try:
            # Reshape for broadcasting
            t0, t1 = t0.view(-1, 1), t1.view(-1, 1)

            # Transform to quadrature interval
            mid = 0.5 * (t0 + t1)
            half = 0.5 * (t1 - t0)

            # Combine endpoint and quadrature points
            ts = torch.cat([t1, mid + half * self.std_nodes], dim=1)

            # Compute log hazard at all points
            temp = self._log_hazard(t0, ts, x, psi, alpha, beta, log_lambda0, g)

            # Extract log hazard at endpoint and quadrature points
            log_hazard = temp[:, :1]  # Log hazard at t1
            quad_vals = torch.exp(temp[:, 1:])  # Hazard at quadrature points

            # Compute cumulative hazard using quadrature
            cumulative = half.flatten() * (quad_vals * self.std_weights).sum(dim=1)

            return log_hazard.flatten(), cumulative

        except Exception as e:
            raise RuntimeError(f"Error in joint hazard computation: {e}")

    def _build_buckets(
        self, T: list, C: torch.Tensor
    ) -> Dict[Tuple[Any, Any], Dict[str, torch.Tensor]]:
        """
        Build survival data buckets for efficient likelihood computation.

        This method organizes survival data into buckets based on state transitions,
        allowing for vectorized likelihood computations across indivs with
        similar transition patterns.

        Args:
            T: List of trajectory lists, where each trajectory is [(t, state), ...]
            C: Censoring times [n]

        Returns:
            Dict mapping (from_state, to_state) -> {idx, t0, t1, obs}
        """
        try:
            # Get survival transitions defined in the model
            surv_transitions = set(self.surv.keys())

            # Build alternative state mapping
            alt_map = defaultdict(list)
            for from_state, to_state in surv_transitions:
                alt_map[from_state].append(to_state)

            # Initialize buckets
            buckets = defaultdict(lambda: [[], [], [], []])

            # Process each indiv's trajectory
            for i, trajectory in enumerate(T):
                # Add censoring time as final observation
                extended_traj = trajectory + [(float(C[i]), torch.nan)]

                # Process consecutive time points
                for (t0, s0), (t1, s1) in zip(extended_traj, extended_traj[1:]):
                    # Skip invalid time intervals
                    if t0 >= t1:
                        continue

                    # Process all possible transitions from current state
                    for potential_to_state in alt_map.get(s0, []):
                        transition_key = (s0, potential_to_state)

                        # Only process defined transitions
                        if transition_key in surv_transitions:
                            buckets[transition_key][0].append(i)  # Subject index
                            buckets[transition_key][1].append(t0)  # Start time
                            buckets[transition_key][2].append(t1)  # End time
                            buckets[transition_key][3].append(
                                potential_to_state == s1
                            )  # Observed

            # Convert to tensors
            processed_buckets = {}
            for key, values in buckets.items():
                if len(values[0]) > 0:  # Only include non-empty buckets
                    processed_buckets[key] = {
                        "idx": torch.tensor(values[0], dtype=torch.int64),
                        "t0": torch.tensor(values[1], dtype=torch.float32).view(-1, 1),
                        "t1": torch.tensor(values[2], dtype=torch.float32).view(-1, 1),
                        "obs": torch.tensor(values[3], dtype=torch.bool),
                    }

            return processed_buckets

        except Exception as e:
            raise RuntimeError(f"Error building survival buckets: {e}")

    def _sample_survival_time(
        self,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        log_lambda0: callable,
        g: callable,
        t_surv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample survival times using inverse transform sampling.

        This method uses bisection search to find survival times that match
        exponentially distributed random variables, effectively sampling from
        the survival distribution defined by the hazard model.

        Args:
            t_left: Left boundary times [n x 1]
            t_right: Right boundary times [n x 1]
            x: Covariates [n x p]
            psi: Random effects [n x psi_dim]
            alpha: Hazard coefficients for g(t,x,psi)
            beta: Hazard coefficients for x
            log_lambda0: Baseline log hazard function
            g: Function g(t,x,psi) for hazard model
            t_surv: Optional conditioning survival times [n x 1]

        Returns:
            torch.Tensor: Sampled survival times [n]
        """
        try:
            n = x.shape[0]

            # Initialize for bisection search
            t0 = t_left.clone().view(-1, 1)
            t_left, t_right = t_left.view(-1, 1), t_right.view(-1, 1)

            # Generate exponential random variables
            target = -torch.log(torch.clamp(torch.rand(n), min=1e-8))

            # Adjust target if conditioning on existing survival
            if t_surv is not None:
                t_surv = t_surv.view(-1, 1)
                existing_hazard = self._cum_hazard(
                    t0, t_surv, x, psi, alpha, beta, log_lambda0, g
                )
                target += existing_hazard

            # Bisection search for survival times
            for _ in range(self.n_bissect):
                t_mid = 0.5 * (t_left + t_right)

                cumulative = self._cum_hazard(
                    t0, t_mid, x, psi, alpha, beta, log_lambda0, g
                )

                # Update search bounds
                accept_mask = cumulative < target
                t_left[accept_mask] = t_mid[accept_mask]
                t_right[~accept_mask] = t_mid[~accept_mask]

            return t_right.flatten()

        except Exception as e:
            raise RuntimeError(f"Error sampling survival times: {e}")

    def sample(
        self,
        T: List[List[Tuple[float, Any]]],
        C: torch.Tensor,
        x: torch.Tensor,
        psi: torch.Tensor,
        t_surv: Optional[Any] = None,
        max_iter: int = 100,
    ) -> List[List[Tuple[float, Any]]]:
        """
        Sample future trajectories from the fitted joint model.

        This method generates future state transitions by sampling from the
        hazard model, allowing for prediction of survival outcomes and
        state progression patterns.

        Args:
            T: List of current trajectories, where each trajectory is [(t, state), ...]
            C: Maximum observation times [n]
            x: Covariates [n x p]
            psi: Random effects [n x psi_dim]
            t_surv: Optional conditioning survival times [n]
            max_iter: Maximum iterations for sampling (prevents infinite loops)

        Returns:
            List of sampled trajectories with future transitions

        Raises:
            RuntimeError: If sampling fails
        """
        try:
            # Convert inputs to tensors
            C = torch.as_tensor(C, dtype=torch.float32)
            x = torch.as_tensor(x, dtype=torch.float32)
            psi = torch.as_tensor(psi, dtype=torch.float32)

            # Validate inputs
            if C.ndim != 1:
                raise ValueError("C must be 1-dimensional")
            if x.ndim != 2:
                raise ValueError("x must be 2-dimensional")
            if psi.ndim != 2:
                raise ValueError("psi must be 2-dimensional")
            if psi.shape[1] != self.f.output_dim:
                raise ValueError("psi must match output dimension of f")
            if not isinstance(T, list) or not all(
                isinstance(traj, list)
                and all(isinstance(event, tuple) for event in traj)
                for traj in T
            ):
                raise TypeError("T must be a list of lists of tuples")
            if not x.shape[0] == psi.shape[0] == len(T):
                raise ValueError(
                    "Number of trajectories must match number of individuals"
                )

            n = x.shape[0]

            # Initialize with copies of current trajectories
            sampled_trajectories = [list(trajectory) for trajectory in T]

            # Get initial buckets from last states
            last_states = [trajectory[-1:] for trajectory in sampled_trajectories]
            current_buckets = self._build_buckets(last_states, C)

            # Sample future transitions iteratively
            for iteration in range(max_iter):
                # Stop if no more possible transitions
                if not current_buckets:
                    break

                # Initialize candidate transition times
                n_transitions = len(current_buckets)
                t_candidates = torch.full(
                    (n, n_transitions), torch.inf, dtype=torch.float32
                )

                # Sample transition times for each possible transition
                for j, (transition_key, bucket_info) in enumerate(
                    current_buckets.items()
                ):
                    try:
                        # Get parameters for this transition
                        alpha = self.params["alpha"][transition_key]
                        beta = self.params["beta"][transition_key]

                        # Extract bucket information
                        idx = bucket_info["idx"]
                        t0 = bucket_info["t0"]
                        t1 = bucket_info["t1"]

                        # Sample transition times
                        t_sample = self._sample_survival_time(
                            t0,
                            t1 + 1e-8,  # Extend upper bound
                            x[idx],
                            psi[idx],
                            alpha,
                            beta,
                            **self.surv[transition_key],
                            t_surv=(
                                t_surv[idx]
                                if not iteration and t_surv is not None
                                else None
                            ),
                        )

                        # Store candidate times
                        t_candidates[idx, j] = t_sample

                    except Exception as e:
                        warnings.warn(
                            f"Error sampling transition {transition_key}: {e}"
                        )
                        continue

                # Find earliest transition for each indiv
                min_times, argmin_indices = torch.min(t_candidates, dim=1)

                # Identify indivuals with valid transitions
                valid_indivs = torch.nonzero(torch.isfinite(min_times)).flatten()

                # Update trajectories with new transitions
                for indiv_idx in valid_indivs:
                    indiv_idx = int(indiv_idx)
                    transition_idx = int(argmin_indices[indiv_idx])
                    transition_time = float(min_times[indiv_idx])

                    # Get the new state from the transition
                    transition_key = list(current_buckets.keys())[transition_idx]
                    new_state = transition_key[1]  # to_state

                    # Add transition to trajectory
                    sampled_trajectories[indiv_idx].append(
                        (transition_time, new_state)
                    )

                # Update buckets for next iteration
                last_states = [trajectory[-1:] for trajectory in sampled_trajectories]
                current_buckets = self._build_buckets(last_states, C)

            # Remove transitions that exceed censoring times
            final_trajectories = []
            for i, trajectory in enumerate(sampled_trajectories):
                censoring_time = float(C[i])
                filtered_trajectory = [
                    (t, s) for t, s in trajectory if t <= censoring_time
                ]

                # If trajectory was truncated, ensure it doesn't end beyond censoring
                if (
                    len(filtered_trajectory) < len(trajectory)
                    and filtered_trajectory
                    and filtered_trajectory[-1][0] > censoring_time
                ):
                    filtered_trajectory = filtered_trajectory[:-1]

                final_trajectories.append(filtered_trajectory)

            return final_trajectories

        except Exception as e:
            raise RuntimeError(f"Error in trajectory sampling: {e}")
