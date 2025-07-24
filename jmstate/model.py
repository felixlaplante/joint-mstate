import copy
import warnings
import itertools
from collections import defaultdict
from typing import Any, DefaultDict, Dict

import torch
from tqdm import tqdm

from ._hazard import HazardMixin
from ._sampler import MetropolisHastingsSampler
from .utils import *


class MultiStateJointModel(HazardMixin):
    """A class of the nonlinear multistate joint model. It feature possibility
    to simulate data, fit based on stochastic gradient with any torch.optim
    optimizer of choice.

    Args:
        HazardMixin (_type_): A mixin class to delegate hazard related methods.
    """

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        pen: Callable[[ModelParams], torch.Tensor] | None = None,
        n_quad: int = 16,
        n_bissect: int = 16,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing regression, base hazard and link functions and model dimensions.
            init_params (ModelParams): Initial values for the parameters.
            pen (Callable[[ModelParams], torch.Tensor] | None, optional): The penalization function. Defaults to None.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 16.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 16.

        Raises:
            TypeError: If pen is not None and is not callable.
        """

        # Store model components
        self.model_design = model_design
        self.params_ = copy.deepcopy(init_params)

        # Store penalization
        if pen is not None and not callable(pen):
            raise TypeError("pen must be callable or None")
        self.pen: Callable[[ModelParams], torch.Tensor] = lambda params: (
            torch.tensor(0.0, dtype=torch.float32) if pen is None else pen(params)
        )

        # Set up numerical integration
        self.n_quad = n_quad
        self._std_nodes = None
        self._std_weights = None
        self._legendre_quad(self.n_quad)

        # Set up for bissection algorithm
        self.n_bissect = n_bissect

        # Initialize attributes that will be set during fitting
        self.sampler_: MetropolisHastingsSampler | None = None
        self.fim_: torch.Tensor | None = None
        self.fit_ = False

    def _hazard_ll(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the hazard log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        ll = torch.zeros(data.size, dtype=torch.float32)

        for key, bucket in data.buckets_.items():
            idx, t0, t1, obs = bucket

            current_alpha = self.params_.alphas[key]
            current_beta = (
                self.params_.betas[key] if self.params_.betas is not None else None
            )
            current_x = data.x.index_select(0, idx) if data.x is not None else None
            current_psi = psi.index_select(0, idx)
            current_surv = self.model_design.surv[key]

            obs_ll, alts_ll = self._log_and_cum_hazard(
                t0,
                t1,
                current_x,
                current_psi,
                current_alpha,
                current_beta,
                *current_surv,
            )

            vals = obs * obs_ll - alts_ll
            ll.index_add_(0, idx, vals)

        return ll

    def _long_ll(self, psi: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the longitudinal log likelihood.

        Args:
            psi (torch.Tensor): A matrix of individual parameters.
            data (ModelData): Dataset on which likelihood is computed.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        # Compute residuals: observed - predicted (only for valid observations)
        predicted = self.model_design.h(data.valid_t_, data.x, psi)
        diffs = data.valid_y_ - predicted * data.valid_mask_

        # Reconstruct precision matrix R_inv from Cholesky parametrization and logdet
        R_inv_cholesky, R_log_eigvals = self.params_.get_cholesky_and_log_eigvals("R")

        # Compute quadratic forms
        R_quad_forms = (diffs @ R_inv_cholesky).pow(2).sum(dim=(1, 2))

        # Compute total log det for each individual
        R_log_dets = torch.einsum("ij,j->i", data.n_valid_, R_log_eigvals)

        # Log likelihood
        ll = 0.5 * (R_log_dets - R_quad_forms)

        return ll

    def _prior_ll(self, b: torch.Tensor) -> torch.Tensor:
        """Computes the prior log likelihood.

        Args:
            b (torch.Tensor): The individual random effects.

        Raises:
            RuntimeError: If the computation fails.

        Returns:
            torch.Tensor: The computed log likelihood.
        """

        # Reconstruct precision matrix R_inv from Cholesky parametrization and logdet
        Q_inv_cholesky, Q_log_eigvals = self.params_.get_cholesky_and_log_eigvals("Q")

        # Compute quadratic form
        Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=1)

        # Compute log det
        Q_log_det = Q_log_eigvals.sum()

        # Log likelihood:
        ll = 0.5 * (Q_log_det - Q_quad_form)

        return ll

    def _ll(self, b: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the total log likelihood up to a constant.

        Args:
            b (torch.Tensor): The individual random effects.
            data (ModelData): Dataset on which the likeihood is computed.

        Returns:
            torch.Tensor: The computed total log likelihood.
        """

        # Transform random effects to individual-specific parameters
        psi = self.model_design.f(self.params_.gamma, b)

        # Compute individual likelihood components
        long_ll = self._long_ll(psi, data)
        hazard_ll = self._hazard_ll(psi, data)
        prior_ll = self._prior_ll(b)

        # Sum all likelihood components
        total_ll = long_ll + hazard_ll + prior_ll

        return total_ll

    def _build_vec_rep(
        self, trajectories: list[Trajectory], c: torch.Tensor
    ) -> dict[tuple[int, int], tuple[torch.Tensor, ...]]:
        """Build vectorizable bucket representation.

        Args:
            trajectories (list[Trajectory]): The trajectories.
            c (torch.Tensor): Censoring times.

        Raises:
            ValueError: If some keys are not in self.surv.
            RuntimeError: If the building fails.

        Returns:
            dict[tuple[int, int], tuple[torch.Tensor, ...]]: The vectorizable buckets representation.
        """

        try:
            # Get survival transitions defined in the model
            trans = set(self.model_design.surv.keys())

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

    def _prepare_data(self, data: ModelData) -> None:
        """Add derived quantities.

        Args:
            data (ModelData): The current dataset.
        Raises:
            TypeError: If self.params_.betas is None and x is not None or the other way around.
        """

        # Check self.params_.betas and data.x types TypeError: If self.params_.betas is None and x is not None or the other way around.
        if (self.params_.betas is None and data.x is not None) or (
            self.params_.betas is not None and data.x is None
        ):
            raise TypeError(
                "self.params_.betas and data.x should be either both None, or neither"
            )

        # Add derived quantities
        data.valid_mask_ = ~torch.isnan(data.y)
        data.n_valid_ = data.valid_mask_.sum(dim=1)
        data.valid_t_ = torch.nan_to_num(data.t)
        data.valid_y_ = torch.nan_to_num(data.y)
        data.buckets_ = self._build_vec_rep(data.trajectories, data.c)

    def _setup_mcmc(
        self,
        data: ModelData,
        init_step_size: float,
        adapt_rate: float,
        target_accept_rate: float,
    ) -> MetropolisHastingsSampler:
        """Setup the MCMC kernel and hyperparameters.

        Args:
            data (ModelData): The dataset on which the likelihood is to be computed.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings.
            adapt_rate (float, optional): Adaptation rate for the step_size.
            target_accept_rate (float, optional): Mean acceptance target.

        Returns:
            MetropolisHastingsSampler: The intialized Markov kernel.
        """

        # Initialize random effects
        init_b = torch.zeros((data.size, self.params_.Q_dim_), dtype=torch.float32)

        # Create sampler
        sampler = MetropolisHastingsSampler(
            lambda b: self._ll(b, data),
            init_b,
            init_step_size,
            adapt_rate,
            target_accept_rate,
        )

        return sampler

    def fit(
        self,
        data: ModelData,
        optimizer: type[torch.optim.Optimizer] = torch.optim.Adam,
        optimizer_params: Dict[str, Any] = {"lr": 1e-2},
        *,
        n_iter: int = 2000,
        batch_size: int = 5,
        callback: Callable[[], None] | None = None,
        init_step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> None:
        """Fits the MultiStateJointModel.

        Args:
            data (ModelData): The dataset to learn from.
            optimizer (type[torch.optim.Optimizer], optional): The stochastic optimizer constructor. Defaults to torch.optim.Adam.
            optimizer_params (_type_, optional): Optimizer parameter dict. Defaults to {"lr": 1e-2}.
            n_iter (int, optional): Number of iterations for optimization. Defaults to 2000.
            batch_size (int, optional): Batch size used in fitting. Defaults to 5.
            callback (Callable[[], None] | None, optional): A callback function that can be used to track the optimization. Defaults to None.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            target_accept_rate (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
        """

        # Load and complete data
        x_rep = data.x.repeat(batch_size, 1) if data.x is not None else None
        t_rep = data.t if data.t.ndim == 1 else data.t.repeat(batch_size, 1)
        y_rep = data.y.repeat(batch_size, 1, 1)
        trajectories_rep = data.trajectories * batch_size
        c_rep = data.c.repeat(batch_size)

        data_rep = ModelData(x_rep, t_rep, y_rep, trajectories_rep, c_rep)

        self._prepare_data(data_rep)

        # Set up optimizer
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        optimizer_instance = optimizer(params=params_list, **optimizer_params)

        # Set up MCMC
        self.sampler_ = self._setup_mcmc(
            data_rep, init_step_size, adapt_rate, accept_target
        )

        # Warmup MCMC
        self.sampler_.warmup(init_warmup)

        # Main fitting loop
        for iteration in tqdm(range(n_iter), desc="Fitting joint model"):
            try:
                # MCMC: Sample random effects
                self.sampler_.warmup(cont_warmup)
                _, current_ll = self.sampler_.step()

                # Optimization step: Update parameters
                optimizer_instance.zero_grad()
                nll_pen = -current_ll.sum() / batch_size + self.pen(self.params_)
                nll_pen.backward()  # type: ignore

                optimizer_instance.step()

                # Execute callback
                if callback is not None:
                    callback()

            except Exception as e:
                warnings.warn(f"Error in iteration {iteration}: {e}")
                continue

        params_flat = torch.cat([p.detach().flatten() for p in params_list])
        if torch.isnan(params_flat).any() or torch.isinf(params_flat).any():
            warnings.warn("Error infering model parameters")

        # Set fit_ to True
        self.fit_ = True

    def _compute_fim(
        self,
        data: ModelData,
        *,
        n_iter_fim: int = 500,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> None:
        """Computes the Fisher Information Matrix.

        Args:
            data (ModelData): The dataset to learn from. Should be the same as used in fit.
            n_iter_fim (int, optional): Number of iterations to compute n_iter_fim. Defaults to 500.
            step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            target_accept_rate (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
        """

        if not self.fit_:
            warnings.warn(
                "Model should be fit before computing Fisher Information Matrix"
            )

        self._prepare_data(data)

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Setup
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        d = self.params_.numel
        self.fim_ = torch.zeros(d, d)

        for _ in tqdm(range(n_iter_fim), desc="Computing Fisher Information Matrix"):
            # Sample random effects
            sampler.warmup(cont_warmup)
            _, current_ll = sampler.step()
            nll_pen = -current_ll.sum() + self.pen(self.params_)

            # Clear gradients
            for p in params_list:
                if p.grad is not None:
                    p.grad.zero_()

            # Compute gradients
            nll_pen.backward()  # type: ignore

            # Collect gradient vector
            grad_chunks: list[torch.Tensor] = []
            for p in params_list:
                if p.grad is not None:
                    grad_chunks.append(p.grad.view(-1))
                else:
                    grad_chunks.append(torch.zeros(p.numel()))

            grad = torch.cat(grad_chunks)

            # Update Fisher Information Matrix
            self.fim_ += torch.outer(grad, grad) / n_iter_fim

        if torch.isnan(self.fim_).any() or torch.isinf(self.fim_).any():
            warnings.warn("Error computing Fisher Information Matrix")
            self.fim_ = None

    def get_stderror(self) -> ModelParams:
        """Returns the standard error of the parameters that can be used to
        draw confidence intervals.

        Raises:
            ValueError: If the Fisher Information Matrix could not be computed.

        Returns:
            ModelParams: The standard error in the same format as the parameters.
        """

        # Check if self.fim_ is well defined
        if self.fim_ is None:
            raise ValueError(
                "Fisher Information Matrix must be previously computed. CIs may not be computed."
            )

        # Get parameter vector
        params_list = self.params_.as_list
        params_flat = torch.cat([p.detach().flatten() for p in params_list])

        # Compute standard errors
        try:
            fim_inv = torch.linalg.pinv(self.fim_)  # type: ignore
            flat_se = torch.sqrt(fim_inv.diag())  # type: ignore

        except Exception as e:
            warnings.warn(f"Error inverting Fisher Information Matrix: {e}")
            flat_se = torch.full_like(params_flat, torch.nan)

        # Organize by parameter structure
        i = 0

        def _next(ref: torch.Tensor) -> torch.Tensor:
            nonlocal i
            n = ref.numel()
            result = flat_se[i : i + n]
            i += n
            return result

        gamma = _next(self.params_.gamma) if self.params_.gamma is not None else None

        Q_flat = _next(self.params_.Q_repr[0])
        Q_method = self.params_.Q_repr[1]

        R_flat = _next(self.params_.R_repr[0])
        R_method = self.params_.R_repr[1]

        alphas = {key: _next(val) for key, val in self.params_.alphas.items()}

        betas = (
            {key: _next(val) for key, val in self.params_.betas.items()}
            if self.params_.betas is not None
            else None
        )

        se_params = ModelParams(
            gamma, (Q_flat, Q_method), (R_flat, R_method), alphas, betas
        )

        return se_params

    def compute_surv_log_probs(
        self, sample_data: SampleData, u: torch.Tensor
    ) -> torch.Tensor:
        """Computes log probabilites of remaining event free up to time u.

        Args:
            sample_data (SampleData): The data on which to compute the probabilities.
            u (torch.Tensor): The time at which to evaluate the probabilities.

        Raises:
            ValueError: If u is of incorrect shape.

        Returns:
            torch.Tensor: The computed survival log probabilities.
        """

        # Convert to float32
        u = torch.as_tensor(u, dtype=torch.float32)

        # Check dims
        if u.ndim != 2 or u.shape[0] != sample_data.size:
            raise ValueError(
                f"u must have shape ({sample_data.size}, eval_points), got {u.shape}"
            )

        last_states = [trajectory[-1:] for trajectory in sample_data.trajectories]
        buckets = self._build_vec_rep(
            last_states, torch.full((sample_data.size,), torch.inf)
        )

        nlog_probs = torch.zeros_like(u)

        for key, bucket in buckets.items():
            for k in range(u.shape[1]):
                idx, t0, _, _ = bucket

                current_alpha = self.params_.alphas[key]
                current_beta = (
                    self.params_.betas[key] if self.params_.betas is not None else None
                )
                current_x = (
                    sample_data.x.index_select(0, idx)
                    if sample_data.x is not None
                    else None
                )
                current_psi = sample_data.psi.index_select(0, idx)
                current_surv = self.model_design.surv[key]

                t1 = u[:, k]

                alts_ll = self._cum_hazard(
                    t0,
                    t1,
                    current_x,
                    current_psi,
                    current_alpha,
                    current_beta,
                    *current_surv,
                )

                nlog_probs[:, k].index_add_(0, idx, alts_ll)

        log_probs = -nlog_probs

        return log_probs

    def sample_trajectories(
        self,
        sample_data: SampleData,
        c_max: torch.Tensor,
        max_length: int = 100,
    ) -> list[Trajectory]:
        """Sample future trajectories from the fitted joint model.

        Args:
            sample_data (SampleData): Prediction data.
            c_max (torch.Tensor): The maximum trajectory sampling time (censoring time).
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            ValueError: If all the parameters are not set.
            ValueError: If the shape of c_max is not compatible.
            RuntimeError: If the sampling fails.

        Returns:
            list[Trajectory]: The sampled trajectories.
        """

        try:
            # Convert and check if c_max matches the right shape
            c_max = torch.as_tensor(c_max, dtype=torch.float32)
            if c_max.shape != (sample_data.size,):
                raise ValueError(
                    "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
                )

            # Initialize with copies of current trajectories
            trajectories = [list(trajectory) for trajectory in sample_data.trajectories]

            # Get initial buckets from last states
            last_states = [trajectory[-1:] for trajectory in trajectories]
            current_buckets = self._build_vec_rep(last_states, c_max)

            # Sample future transitions iteratively
            for iteration in range(max_length):
                # Stop if no more possible transitions
                if not current_buckets:
                    break

                # Initialize candidate transition times
                n_transitions = len(current_buckets)
                t_candidates = torch.full(
                    (sample_data.size, n_transitions), torch.inf, dtype=torch.float32
                )

                # Sample transition times for each possible transition
                for j, (key, bucket) in enumerate(current_buckets.items()):
                    try:
                        # Extract bucket information
                        idx, t0, t1, _ = bucket

                        # Extend upper bound
                        t1 = torch.nextafter(
                            t1, torch.tensor(torch.inf, dtype=torch.float32)
                        )

                        current_alpha = self.params_.alphas[key]
                        current_beta = (
                            self.params_.betas[key]
                            if self.params_.betas is not None
                            else None
                        )
                        current_x = (
                            sample_data.x.index_select(0, idx)
                            if sample_data.x is not None
                            else None
                        )
                        current_psi = sample_data.psi.index_select(0, idx)
                        current_surv = self.model_design.surv[key]
                        current_c = (
                            sample_data.c.index_select(0, idx)
                            if not iteration and sample_data.c is not None
                            else None
                        )

                        # Sample transition times
                        t_sample = self._sample_trajectory_step(
                            t0,
                            t1,
                            current_x,
                            current_psi,
                            current_alpha,
                            current_beta,
                            *current_surv,
                            c=current_c,
                            n_bissect=self.n_bissect,
                        )

                        # Store candidate times
                        t_candidates[idx, j] = t_sample

                    except Exception as e:
                        warnings.warn(f"Error sampling transition {key}: {e}")
                        continue

                # Find earliest transition
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
                    trajectories[indiv_idx].append((transition_time, new_state))

                # Update buckets for next iteration
                last_states = [trajectory[-1:] for trajectory in trajectories]
                current_buckets = self._build_vec_rep(last_states, c_max)

            # Remove transitions that exceed censoring times
            for i, trajectory in enumerate(trajectories):
                censoring_time = float(c_max[i])
                # If trajectory was truncated, ensure it doesn't end beyond censoring
                if trajectory[-1][0] > censoring_time:
                    trajectories[i] = trajectory[:-1]

            return trajectories

        except Exception as e:
            raise RuntimeError(f"Error in trajectory sampling: {e}") from e

    def predict_surv_log_probs(
        self,
        pred_data: ModelData,
        u: torch.Tensor,
        *,
        n_iter_b: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> list[torch.Tensor]:
        """Predicts the survival (event free) probabilities for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            u (torch.Tensor): The evaluation times of the probabilities.
            n_iter_b (int): Number of iterations for random effects sampling.
            step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            TypeError: If self.params_.betas is None and x is not None or the other way around.
            ValueError: If u is of incorrect shape.
            RuntimeError: If the computation fails.

        Returns:
            list[torch.Tensor]: A list for each b of survival probabilities.
        """

        try:
            # Check self.params_.betas and data.x types
            if (self.params_.betas is None and pred_data.x is not None) or (
                self.params_.betas is not None and pred_data.x is None
            ):
                raise TypeError(
                    "self.params_.betas and data.x should be either both None, or neither"
                )

            # Convert and check if c_max matches the right shape
            u = torch.as_tensor(u, dtype=torch.float32)
            if u.ndim != 2 or u.shape[0] != pred_data.size:
                raise ValueError(
                    "u has incorrect shape, got {u.shape}, expected {(sample_data.size, eval_points)}"
                )

            # Load and complete prediction data
            self._prepare_data(pred_data)

            # Set up MCMC for prediction
            sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

            # Warmup MCMC
            sampler.warmup(init_warmup)

            # Generate predicted probabilites
            predicted_log_probs: list[torch.Tensor] = []

            for _ in tqdm(range(n_iter_b), desc="Predicting survival probabilities"):
                # Sample random effects
                sampler.warmup(cont_warmup)

                current_b, _ = sampler.step()

                # Transform to individual-specific parameters
                psi = self.model_design.f(self.params_.gamma, current_b)

                sample_data = SampleData(pred_data.x, pred_data.trajectories, psi)

                c_log_probs = self.compute_surv_log_probs(
                    sample_data, pred_data.c.view(-1, 1)
                )
                u_log_probs = self.compute_surv_log_probs(sample_data, u)
                current_log_probs = torch.clamp(u_log_probs - c_log_probs, max=0.0)

                predicted_log_probs.append(current_log_probs)

            return predicted_log_probs

        except Exception as e:
            raise RuntimeError(f"Error in survival prediction: {e}") from e

    def predict_trajectories(
        self,
        pred_data: ModelData,
        c_max: torch.Tensor,
        *,
        n_iter_b: int,
        n_iter_T: int,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        max_length: int = 100,
    ) -> list[list[list[Trajectory]]]:
        """Predict survival trajectories for new individuals.

        Args:
            pred_data (ModelData): Prediction data.
            c_max (torch.Tensor): Maximum prediction times.
            n_iter_b (int): Number of iterations for random effects sampling.
            n_iter_T (int): Number of trajectory samples per random effects sample.
            step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            accept_target (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.
            max_length (int, optional): Maximum iterations or sampling (prevents infinite loops). Defaults to 100.

        Raises:
            TypeError: If self.params_.betas is None and x is not None or the other way around.
            RuntimeError: If the prediction fails.

        Returns:
            list[list[list[Trajectory]]]: A list of lists of trajectories. First list is for a b sample, then multiples iid drawings of the trajectories.
        """

        try:
            # Check self.params_.betas and data.x types
            if (self.params_.betas is None and pred_data.x is not None) or (
                self.params_.betas is not None and pred_data.x is None
            ):
                raise TypeError(
                    "self.params_.betas and data.x should be either both None, or neither"
                )

            # Convert and check if c_max matches the right shape
            c_max = torch.as_tensor(c_max, dtype=torch.float32)
            if c_max.shape != (pred_data.size,):
                raise ValueError(
                    "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
                )

            # Load and complete prediction data
            self._prepare_data(pred_data)

            # Set up MCMC for prediction
            sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

            # Warmup MCMC
            sampler.warmup(init_warmup)

            # Prepare replicate data for trajectory sampling
            x_rep = pred_data.x.repeat(n_iter_T, 1) if pred_data.x is not None else None
            trajectories_rep = pred_data.trajectories * n_iter_T
            c_rep = pred_data.c.repeat(n_iter_T)
            c_max_rep = c_max.repeat(n_iter_T)

            # Generate predictions
            predicted_trajectories: list[list[list[Trajectory]]] = []

            for _ in tqdm(range(n_iter_b), desc="Predicting trajectories"):
                # Sample random effects
                sampler.warmup(cont_warmup)

                current_b, _ = sampler.step()

                # Transform to individual-specific parameters
                psi = self.model_design.f(self.params_.gamma, current_b)

                # Replicate for multiple trajectory samples
                psi_rep = psi.repeat(n_iter_T, 1)

                sample_data = SampleData(x_rep, trajectories_rep, psi_rep, c_rep)

                # Sample trajectories
                current_trajectories = self.sample_trajectories(
                    sample_data, c_max_rep, max_length
                )

                # Organize by trajectory iteration
                trajectory_chunks = [
                    current_trajectories[i * pred_data.size : (i + 1) * pred_data.size]
                    for i in range(n_iter_T)
                ]

                predicted_trajectories.append(trajectory_chunks)

            return predicted_trajectories

        except Exception as e:
            raise RuntimeError(f"Error in survival prediction: {e}") from e
