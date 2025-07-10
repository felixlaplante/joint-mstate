import torch
import warnings
from typing import Dict, List, Tuple, Optional, Callable, Any
from tqdm import tqdm

from .utils import Fun, _legendre_quad
from .likelihood import LikelihoodMixin
from .hazard import HazardMixin
from .sampler import MetropolisHastingsSampler


class JointModel(LikelihoodMixin, HazardMixin):
    """
    Joint model for longitudinal and survival data.

    This class implements a joint modeling framework that combines:
    - Longitudinal submodel for repeated measurements over time
    - Survival submodel for time-to-event outcomes
    - Shared random effects linking the two submodels

    The model uses MCMC for inference and supports flexible hazard functions,
    multiple competing risks, and various longitudinal response distributions.
    """

    def __init__(
        self,
        h: Fun,
        f: Fun,
        surv: Dict[Tuple[Any, Any], Dict[str, Callable]],
        n_quad: int = 16,
        n_bissect: int = 16,
    ):
        """
        Initialize the joint model.

        Args:
            h: Longitudinal model function h(t, x, psi)
            f: Random effects transformation function f(gamma, b)
            surv: Survival model specifications {(from_state, to_state): {log_lambda0, g}}
            n_quad: Number of quadrature points for numerical integration
            n_bissect: Number of bisection steps for survival time sampling

        Raises:
            TypeError: If input functions are not callable or properly structured
            ValueError: If dimensions are inconsistent
        """
        # Validate inputs
        if not isinstance(h, Fun):
            raise TypeError("h must be a Fun instance")
        if not isinstance(f, Fun):
            raise TypeError("f must be a Fun instance")
        if not isinstance(surv, dict):
            raise TypeError("surv must be a dictionary")

        # Store model components
        self.h = h  # Longitudinal model function
        self.f = f  # Random effects transformation
        self.surv = surv  # Survival model specifications

        # Initialize parameters dictionary
        self.params = {}

        # Set up numerical integration
        self.std_nodes, self.std_weights = _legendre_quad(n_quad)
        self.n_bissect = n_bissect

        # Validate model structure
        self._check()

        # Model fitting status
        self.fit_ = False

        # Initialize attributes that will be set during fitting
        self.fim = None

    def _check(self) -> None:
        """Validate the structure of the survival model specification."""
        if len(self.f.input_dim) != 2:
            raise ValueError(
                f"f should have input_dim of type tuple of length two, got {self.f.input_dim}"
            )
        if len(self.h.input_dim) != 3:
            raise ValueError(
                f"h should have input_dim of type tuple of length three, got {self.h.input_dim}"
            )
        if not isinstance(self.f.output_dim, int):
            raise TypeError(f"f must have integer output_dim")
        if not isinstance(self.h.output_dim, int):
            raise TypeError(f"h must have integer output_dim")
        if self.f.output_dim <= 0:
            raise ValueError(
                f"f output dimension must be positive, got {self.f.output_dim}"
            )
        for i, dim in enumerate(self.f.input_dim):
            if not isinstance(dim, int):
                raise TypeError(f"f input_dim[{i}] must be integer, got {type(dim)}")
            if dim <= 0:
                raise ValueError(f"f input_dim[{i}] must be positive, got {dim}")
        if self.h.output_dim <= 0:
            raise ValueError(
                f"h output dimension must be positive, got {self.f.output_dim}"
            )
        for i, dim in enumerate(self.h.input_dim):
            if not isinstance(dim, int):
                raise TypeError(f"h input_dim[{i}] must be integer, got {type(dim)}")
            if dim <= 0:
                raise ValueError(f"h input_dim[{i}] must be positive, got {dim}")
        if self.f.output_dim != self.h.input_dim[2]:
            raise ValueError(
                f"f output dimension ({self.f.output_dim}) must match h third input dimension ({self.h.input_dim[2]})"
            )

        for key, spec in self.surv.items():
            if not isinstance(key, tuple) or len(key) != 2:
                raise ValueError(f"Survival keys must be 2-tuples, got {key}")

            required_keys = {"log_lambda0", "g"}
            if not all(k in spec for k in required_keys):
                raise ValueError(
                    f"Survival spec for {key} missing required keys: {required_keys}"
                )

            if not all(callable(spec[k]) for k in required_keys):
                raise ValueError(
                    f"Survival spec for {key} must contain callable functions"
                )

    def _init_params(self) -> None:
        """Initialize model parameters for optimization."""
        try:
            # Fixed effects for random effects transformation
            self.params["gamma"] = torch.zeros(
                self.f.input_dim[0], dtype=torch.float32, requires_grad=True
            )

            # Random effects precision matrix (Cholesky parameterization)
            q_dim = self.f.input_dim[1]
            self.params["Q_inv"] = torch.zeros(
                q_dim * (q_dim + 1) // 2, dtype=torch.float32, requires_grad=True
            )

            # Longitudinal residual precision matrix (Cholesky parameterization)
            r_dim = self.h.output_dim
            self.params["R_inv"] = torch.zeros(
                r_dim * (r_dim + 1) // 2, dtype=torch.float32, requires_grad=True
            )

            # Survival model parameters
            self.params["alpha"] = {}
            self.params["beta"] = {}

            for transition_key in self.surv.keys():
                # Coefficients for time-varying effects g(t,x,psi)
                g_output_dim = self.surv[transition_key]["g"].output_dim
                self.params["alpha"][transition_key] = torch.zeros(
                    g_output_dim, dtype=torch.float32, requires_grad=True
                )

                # Coefficients for covariates x
                self.params["beta"][transition_key] = torch.zeros(
                    self.h.input_dim[1], dtype=torch.float32, requires_grad=True
                )

        except Exception as e:
            raise RuntimeError(f"Error initializing parameters: {e}")

    def _prepare_data(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        T: List[List[Tuple[float, Any]]],
        C: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Prepare and validate data for model fitting or prediction.

        Args:
            x: Covariates [n x p]
            t: Observation times [n x max_obs] or broadcastable
            y: Longitudinal outcomes [n x max_obs x output_dim]
            T: Survival trajectories [n] of [(time, state), ...]
            C: Censoring times [n]

        Returns:
            Dict containing processed and validated data

        Raises:
            ValueError: If data dimensions are inconsistent
            TypeError: If data types are incorrect
        """
        # Convert to tensors
        data = {
            "x": torch.as_tensor(x, dtype=torch.float32),
            "t": torch.as_tensor(t, dtype=torch.float32),
            "y": torch.as_tensor(y, dtype=torch.float32),
            "T": T,
            "C": torch.as_tensor(C, dtype=torch.float32),
        }

        # Validate tensor dimensions
        if data["C"].ndim != 1:
            raise ValueError("C must be 1-dimensional")
        if data["x"].ndim != 2:
            raise ValueError("x must be 2-dimensional")
        if data["y"].ndim != 3:
            raise ValueError("y must be 3-dimensional")

        # Validate trajectory structure
        if not isinstance(data["T"], list) or not all(
            isinstance(traj, list) and all(isinstance(event, tuple) for event in traj)
            for traj in data["T"]
        ):
            raise TypeError("T must be a list of lists of tuples")

        # Validate consistent number of subjects
        n_subjects = data["x"].shape[0]
        if not (
            data["y"].shape[0] == n_subjects
            and len(data["T"]) == n_subjects
            and data["C"].numel() == n_subjects
        ):
            raise ValueError("Inconsistent number of individuals")

        # Validate time dimension compatibility
        valid_t_shapes = [
            (data["y"].shape[1],),
            (1, data["y"].shape[1]),
            data["y"].shape[:2],
        ]
        if data["t"].shape not in valid_t_shapes:
            raise ValueError("t must be broadcastable with y")

        # Add derived quantities
        data["n"], data["p"] = data["x"].shape
        data["valid_mask"] = ~torch.isnan(data["y"])
        data["n_valid"] = data["valid_mask"].any(dim=2).sum(dim=1)
        data["y"] = torch.nan_to_num(data["y"])  # Replace NaN with 0
        data["buckets"] = self._build_buckets(data["T"], data["C"])

        return data

    def _setup_mcmc(
        self,
        data,
        step_size: float = 0.1,
        adapt_rate: float = 0.1,
        target_accept_rate: float = 0.234,
    ) -> MetropolisHastingsSampler:
        """
        Set up MCMC sampler for random effects.

        Args:
            step_size: Initial step size for proposals
            adapt_step_size: Whether to adapt step size during sampling
            adapt_rate: Rate of step size adaptation
            target_accept_rate: Target acceptance rate

        Returns:
            MetropolisHastingsSampler instance
        """
        # Initialize random effects
        initial_b = torch.zeros((data["n"], self.f.input_dim[1]), dtype=torch.float32)

        # Create sampler
        sampler = MetropolisHastingsSampler(
            log_prob_fn=lambda b: self._ll(b, data),
            initial_state=initial_b,
            step_size=step_size,
            adapt_rate=adapt_rate,
            target_accept_rate=target_accept_rate,
        )

        return sampler

    def _mcmc_batch(self, sampler: MetropolisHastingsSampler, batch_size: int) -> float:
        """
        Run MCMC sampling with batch collection.

        Args:
            sampler: MCMC sampler instance
            warmup: Number of warmup iterations
            batch_size: Number of batch iterations

        Returns:
            Tuple of (average_ll, final_state, final_log_prob)
        """
        # Run batch sampling
        total_ll = 0.0

        for _ in range(batch_size):
            try:
                _, curr_log_prob = sampler.step()
                total_ll += curr_log_prob.sum()
            except Exception as e:
                warnings.warn(f"Error in batch sampling: {e}")
                continue

        avg_ll = total_ll / batch_size
        return avg_ll

    def _param_list(self) -> List[torch.Tensor]:
        """Get list of all parameters for optimization."""
        params = []

        # Add non-dictionary parameters
        for key, value in self.params.items():
            if not isinstance(value, dict):
                params.append(value)

        # Add dictionary parameters
        params.extend(self.params["alpha"].values())
        params.extend(self.params["beta"].values())

        return params

    def fit(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        T: List[List[Tuple[float, int]]],
        C: torch.Tensor,
        optimizer: Callable,
        optimizer_params: Dict[str, Any],
        n_iter: int = 1000,
        batch_size: int = 1,
        callback: Optional[Callable] = None,
        n_iter_fim: int = 1000,
        step_size: float = 0.1,
        accept_step_size: float = 0.1,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
    ) -> None:
        """
        Fit the joint model using MCMC-EM algorithm.

        Args:
            x: Covariates [n x p]
            t: Observation times [n x max_obs x 1]
            y: Longitudinal outcomes [n x max_obs x output_dim]
            T: Survival trajectories [n] of [(time, state), ...]
            C: Censoring times [n]
            optimizer: Optimizer class (e.g., torch.optim.Adam)
            optimizer_params: Parameters for optimizer
            n_iter: Number of EM iterations
            batch_size: MCMC batch size per iteration
            callback: Optional callback function called after each iteration
            n_iter_fim: Iterations for Fisher Information Matrix computation
            step_size: Initial MCMC step size
            accept_step_size: Step size for acceptance rate adaptation
            accept_target: Target acceptance rate
            init_warmup: Initial warmup for the MCMC sampler.
            cont_warmup: Continuous warmup for the MCMC sampler.

        Raises:
            RuntimeError: If fitting fails
        """
        try:
            # Convert and validate inputs
            data = self._prepare_data(x, t, y, T, C)

            # Initialize parameters
            self._init_params()

            # Set up optimizer
            param_list = self._param_list()
            optimizer_instance = optimizer(params=param_list, **optimizer_params)

            # Set up MCMC
            sampler = self._setup_mcmc(data, step_size, accept_step_size, accept_target)

            # Warmup MCMC
            sampler.warmup(init_warmup)

            # Main fitting loop
            for iteration in tqdm(range(n_iter), desc="Fitting joint model"):
                try:
                    # MCMC: Sample random effects
                    sampler.warmup(cont_warmup)
                    avg_ll = self._mcmc_batch(sampler, batch_size)

                    # Optimization step: Update parameters
                    optimizer_instance.zero_grad()
                    nll = -avg_ll
                    nll.backward()

                    optimizer_instance.step()

                    # Execute callback
                    if callback is not None:
                        callback()

                except Exception as e:
                    warnings.warn(f"Error in iteration {iteration}: {e}")
                    continue

            # Compute Fisher Information Matrix
            self._compute_fim(sampler, n_iter_fim, cont_warmup)

            # Mark as fitted
            self.fit_ = True

        except Exception as e:
            raise RuntimeError(f"Error fitting joint model: {e}")

    def _compute_fim(
        self, sampler: MetropolisHastingsSampler, n_iter_fim: int, cont_warmup: int
    ) -> None:
        """Compute Fisher Information Matrix for standard errors."""
        try:
            param_list = self._param_list()
            d = sum(p.numel() for p in param_list)
            self.fim = torch.zeros(d, d)

            for _ in tqdm(
                range(n_iter_fim), desc="Computing Fisher Information Matrix"
            ):
                # Sample random effects
                sampler.warmup(cont_warmup)
                _, curr_ll = sampler.step()

                # Clear gradients
                for p in param_list:
                    if p.grad is not None:
                        p.grad.zero_()

                # Compute gradients
                ll = curr_ll.sum()
                ll.backward()

                # Collect gradient vector
                grad_parts = []
                for p in param_list:
                    if p.grad is not None:
                        grad_parts.append(p.grad.view(-1))
                    else:
                        grad_parts.append(torch.zeros(p.numel()))

                grad = torch.cat(grad_parts)

                # Update Fisher Information Matrix
                self.fim += torch.outer(grad, grad) / n_iter_fim

        except Exception as e:
            warnings.warn(f"Error computing Fisher Information Matrix: {e}")
            self.fim = None

    def stderror(self) -> Dict[str, Any]:
        """
        Compute confidence intervals for parameters.

        Returns:
            Dict containing standard errors for all parameters

        Raises:
            AssertionError: If model hasn't been fitted
            RuntimeError: If Fisher Information Matrix is unavailable
        """
        assert self.fit_, "Model must be fitted before computing confidence intervals"

        if self.fim is None:
            raise RuntimeError("Fisher Information Matrix already computed")

        try:
            # Get parameter vector
            param_list = self._param_list()
            params = torch.cat([p.detach().flatten() for p in param_list])

            # Compute standard errors
            try:
                fim_inv = torch.linalg.pinv(self.fim)
                flat_se = torch.sqrt(fim_inv.diag())
            except Exception as e:
                warnings.warn(f"Error inverting Fisher Information Matrix: {e}")
                flat_se = torch.full_like(params, torch.nan)

            # Organize by parameter structure
            se = {}
            i = 0

            for key, val in self.params.items():
                if isinstance(val, dict):
                    se[key] = {}
                    for subkey, subval in val.items():
                        n = subval.numel()
                        shape = subval.shape
                        se[key][subkey] = flat_se[i : i + n].view(shape)
                        i += n
                else:
                    n = val.numel()
                    shape = val.shape
                    se[key] = flat_se[i : i + n].view(shape)
                    i += n

            return se

        except Exception as e:
            raise RuntimeError(f"Error computing confidence intervals: {e}")

    def predict_survival(
        self,
        C_max: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        T: List[List[Tuple[float, Any]]],
        C: torch.Tensor,
        n_iter_b: int,
        n_iter_T: int,
        step_size: float = 0.1,
        accept_step_size: float = 0.1,
        accept_target: float = 0.234,
        init_warmup: int = 500,
        cont_warmup: int = 5,
        max_iter: int = 100,
    ) -> List[List[List[List[Tuple[float, Any]]]]]:
        """
        Predict survival trajectories for new individuals.

        Args:
            C_max: Maximum prediction times [n]
            x: Covariates [n x p]
            t: Observation times [n x max_obs x 1]
            y: Longitudinal outcomes [n x max_obs x output_dim]
            T: Current survival trajectories [n] of [(time, state), ...]
            C: Current censoring times [n]
            n_iter_b: Number of iterations for random effects sampling
            n_iter_T: Number of trajectory samples per random effects sample
            step_size: Initial MCMC step size
            accept_step_size: Step size for acceptance rate adaptation
            accept_target: Target acceptance rate
            init_warmup: Initial warmup for the MCMC sampler.
            cont_warmup: Continuous warmup for the MCMC sampler.
            max_iter: Maximum allowed number of iterations to append transitions.

        Returns:
            List of predicted trajectories [n_iter_b x n_iter_T x n]

        Raises:
            AssertionError: If model hasn't been fitted
            RuntimeError: If prediction fails
        """
        assert self.fit_, "Model must be fitted before prediction"

        try:
            # Load and validate prediction data
            data = self._prepare_data(x, t, y, T, C)

            # Set up MCMC for prediction
            sampler = self._setup_mcmc(data, step_size, accept_step_size, accept_target)

            # Warmup MCMC
            sampler.warmup(init_warmup)

            # Prepare replicate data for trajectory sampling
            x_rep = data["x"].repeat(n_iter_T, 1)
            T_rep = data["T"] * n_iter_T
            C_rep = data["C"].repeat(n_iter_T)
            C_max = torch.as_tensor(C_max, dtype=torch.float32)
            C_max_rep = C_max.repeat(n_iter_T)

            # Generate predictions
            predictions = []

            for _ in tqdm(range(n_iter_b), desc="Predicting trajectories"):
                # Sample random effects
                sampler.warmup(cont_warmup)

                curr_b, _ = sampler.step()

                # Transform to individual-specific parameters
                psi = self.f(self.params["gamma"], curr_b)

                # Replicate for multiple trajectory samples
                psi_rep = psi.repeat(n_iter_T, 1)

                # Sample trajectories
                sampled_trajectories = self.sample(
                    T_rep, C_max_rep, x_rep, psi_rep, C_rep, max_iter
                )

                # Organize by trajectory iteration
                trajectory_chunks = [
                    sampled_trajectories[i * data["n"] : (i + 1) * data["n"]]
                    for i in range(n_iter_T)
                ]

                predictions.append(trajectory_chunks)

            return predictions

        except Exception as e:
            raise RuntimeError(f"Error in survival prediction: {e}")
