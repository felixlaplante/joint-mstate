import copy
import warnings
from typing import Any, Callable, Dict

import torch

from ..sampler._sampler import MetropolisHastingsSampler
from ..utils.structures import ModelData, ModelDesign, ModelParams, SampleData
from ..utils.types import Trajectory
from ._hazard import HazardMixin
from ._longitudinal import LongitudinalMixin


class MultiStateJointModel(LongitudinalMixin, HazardMixin):
    """A class of the nonlinear multistate joint model. It feature possibility
    to simulate data, fit based on stochastic gradient with any torch.optim
    optimizer of choice."""

    def __init__(
        self,
        model_design: ModelDesign,
        init_params: ModelParams,
        *,
        pen: Callable[[ModelParams], torch.Tensor] | None = None,
        n_quad: int = 32,
        n_bissect: int = 32,
        enable_likelihood_cache: bool = True,
        enable_predict_cache: bool = True,
        cache_limit: int = 256,
    ):
        """Initializes the joint model based on the user defined design.

        Args:
            model_design (ModelDesign): Model design containing regression, base hazard and link functions and model dimensions.
            init_params (ModelParams): Initial values for the parameters.
            pen (Callable[[ModelParams], torch.Tensor] | None, optional): The penalization function. Defaults to None.
            n_quad (int, optional): The used numnber of points for Gauss-Legendre quadrature. Defaults to 32.
            n_bissect (int, optional): The number of bissection steps used in transition sampling. Defaults to 32.
            enable_likelihood_cache (bool, optional): Enables cache during fit, and MCMC loops and likelihood computations in general. Defaults to True.
            enable_predict_cache (bool, optional): Enables cache during predicting steps. Defaults to True.
            cache_limit (int, optional): The max length of cache. Defaults to 256.

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

        # Info of the Mixin Classes
        super().__init__(
            n_quad=n_quad,
            n_bissect=n_bissect,
            enable_likelihood_cache=enable_likelihood_cache,
            enable_predict_cache=enable_predict_cache,
            cache_limit=cache_limit,
        )

        # Initialize attributes that will be set later
        self.fim_: torch.Tensor | None = None
        self.fit_ = False

    def _ll(self, b: torch.Tensor, data: ModelData) -> torch.Tensor:
        """Computes the total log likelihood up to a constant.

        Args:
            b (torch.Tensor): The individual random effects.
            data (ModelData): Dataset on which the likeihood is computed.

        Returns:
            torch.Tensor: The computed total log likelihood.
        """

        def _prior_ll(b: torch.Tensor) -> torch.Tensor:
            Q_inv_cholesky, Q_log_eigvals = self.params_.get_cholesky_and_log_eigvals(
                "Q"
            )

            Q_quad_form = (b @ Q_inv_cholesky).pow(2).sum(dim=1)
            Q_log_det = Q_log_eigvals.sum()

            # Log likelihood:
            ll = 0.5 * (Q_log_det - Q_quad_form)

            return ll

        # Transform random effects to individual-specific parameters
        psi = self.model_design.f(self.params_.gamma, b)

        # Compute individual likelihood components
        long_ll = super()._long_ll(psi, data)
        hazard_ll = super()._hazard_ll(psi, data)
        prior_ll = _prior_ll(b)

        # Sum all likelihood components
        total_ll = long_ll + hazard_ll + prior_ll

        return total_ll

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
        batch_size: int = 1,
        callback: (
            Callable[["MultiStateJointModel", MetropolisHastingsSampler], None] | None
        ) = None,
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
            batch_size (int, optional): Batch size used in fitting. Defaults to 1.
            callback (Callable[[], None] | None, optional): A callback function that can be used to track the optimization. Defaults to None.
            init_step_size (float, optional): Kernel standard error in Metropolis Hastings. Defaults to 0.1.
            adapt_rate (float, optional): Adaptation rate for the step_size. Defaults to 0.01.
            target_accept_rate (float, optional): Mean acceptation target. Defaults to 0.234.
            init_warmup (int, optional): The number of iteration steps used in the warmup. Defaults to 500.
            cont_warmup (int, optional): The warmup step in-between each parameter changes. Defaults to 5.

        Raises:
            ValueError: If batch_size is not greater than 0.
        """

        # Verify batch_size
        if batch_size < 1:
            raise ValueError(f"batch_size must be greater than 0, got {batch_size}")

        # Load and complete data
        x_rep = data.x.repeat(batch_size, 1) if data.x is not None else None
        t_rep = data.t if data.t.ndim == 1 else data.t.repeat(batch_size, 1)
        y_rep = data.y.repeat(batch_size, 1, 1)
        trajectories_rep = data.trajectories * batch_size
        c_rep = data.c.repeat(batch_size)

        data_rep = ModelData(x_rep, t_rep, y_rep, trajectories_rep, c_rep)

        # Prepare data
        data_rep.prepare(self.model_design)

        # Set up optimizer
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        optimizer_instance = optimizer(params=params_list, **optimizer_params)

        # Set up MCMC
        sampler = self._setup_mcmc(data_rep, init_step_size, adapt_rate, accept_target)
        sampler.warmup(init_warmup)

        def _fit():
            _, current_ll = sampler.step()

            # Optimization step: Update parameters
            optimizer_instance.zero_grad()
            nll_pen = -current_ll.sum() / batch_size + self.pen(self.params_)
            nll_pen.backward()  # type: ignore

            optimizer_instance.step()

            if callback is not None:
                callback(self, sampler)

        # Main fitting loop
        sampler.loop(n_iter, cont_warmup, _fit, desc="Fitting joint model")

        params_flat = torch.cat([p.detach().flatten() for p in params_list])
        if torch.isnan(params_flat).any() or torch.isinf(params_flat).any():
            warnings.warn("Error infering model parameters")

        # Set fit_ to True
        self.params_.require_grad(False)
        self.fit_ = True
        self.clear_cache()

    def compute_fim(
        self,
        data: ModelData,
        *,
        n_iter: int = 500,
        step_size: float = 0.1,
        adapt_rate: float = 0.01,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> None:
        """Computes the Fisher Information Matrix.

        Args:
            data (ModelData): The dataset to learn from. Should be the same as used in fit.
            n_iter (int, optional): Number of iterations to compute n_iter. Defaults to 500.
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

        # Prepare data
        data.prepare(self.model_design)

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Setup
        self.params_.require_grad(True)
        params_list = self.params_.as_list
        d = self.params_.numel
        fim = torch.zeros(d, d)

        def _compute_fim():
            nonlocal fim

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
            fim += torch.outer(grad, grad) / n_iter

        sampler.loop(
            n_iter, cont_warmup, _compute_fim, "Computing Fisher Information Matrix"
        )

        if torch.isnan(fim).any() or torch.isinf(fim).any():
            warnings.warn("Error computing Fisher Information Matrix")
        else:
            self.fim_ = fim

        self.params_.require_grad(False)
        self.clear_cache()

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
            ValueError: If u is of incorrect shape.

        Returns:
            list[torch.Tensor]: A list for each b of survival probabilities.
        """

        # Convert and check if c_max matches the right shape
        u = torch.as_tensor(u, dtype=torch.float32)
        if u.ndim != 2 or u.shape[0] != pred_data.size:
            raise ValueError(
                "u has incorrect shape, got {u.shape}, expected {(sample_data.size, eval_points)}"
            )

        # Load and complete prediction data
        pred_data.prepare(self.model_design)

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Generate predicted probabilites
        predicted_log_probs: list[torch.Tensor] = []

        def _predict_surv_log_probs():
            current_b, _ = sampler.step()

            # Transform to individual-specific parameters
            psi = self.model_design.f(self.params_.gamma, current_b)

            sample_data = SampleData(
                pred_data.x, pred_data.trajectories, psi, pred_data.c
            )

            current_log_probs = self.compute_surv_log_probs(sample_data, u)

            predicted_log_probs.append(current_log_probs)

        sampler.loop(
            n_iter_b,
            cont_warmup,
            _predict_surv_log_probs,
            "Predicting survival log probabilities",
        )

        self.clear_cache()
        return predicted_log_probs

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
            ValueError: If n_iter_T is not greater than 0.
            RuntimeError: If the prediction fails.

        Returns:
            list[list[list[Trajectory]]]: A list of lists of trajectories. First list is for a b sample, then multiples iid drawings of the trajectories.
        """

        # Convert and check if c_max matches the right shape
        c_max = torch.as_tensor(c_max, dtype=torch.float32)
        if c_max.shape != (pred_data.size,):
            raise ValueError(
                "c_max has incorrect shape, got {c_max.shape}, expected {(sample_data.size,)}"
            )

        # Load and complete prediction data

        # Set up MCMC for prediction
        sampler = self._setup_mcmc(pred_data, step_size, adapt_rate, accept_target)

        # Warmup MCMC
        sampler.warmup(init_warmup)

        # Check n_iter_T
        if n_iter_T < 1:
            raise ValueError(f"n_iter_T must be greater than 0, got {n_iter_T}")

        # Prepare replicate data for trajectory sampling
        x_rep = pred_data.x.repeat(n_iter_T, 1) if pred_data.x is not None else None
        trajectories_rep = pred_data.trajectories * n_iter_T
        c_rep = pred_data.c.repeat(n_iter_T)
        c_max_rep = c_max.repeat(n_iter_T)

        # Generate predictions
        predicted_trajectories: list[list[list[Trajectory]]] = []

        def _predict_trajectories():
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

        sampler.loop(
            n_iter_b, cont_warmup, _predict_trajectories, "Predicting trajectories"
        )

        self.clear_cache()
        return predicted_trajectories
