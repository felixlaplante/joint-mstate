import torch
from typing import Callable, Tuple
import warnings


class MetropolisHastingsSampler:
    """
    A robust Metropolis-Hastings sampler with adaptive step size.
    """

    def __init__(
        self,
        log_prob_fn: Callable[[torch.Tensor], torch.Tensor],
        initial_state: torch.Tensor,
        step_size: float = 0.1,
        adapt_rate: float = 0.1,
        target_accept_rate: float = 0.234,
        min_step_size: float = 1e-6,
        max_step_size: float = 10.0,
    ):
        """
        Initialize the Metropolis-Hastings sampler.

        Args:
            log_prob_fn: Function that computes log probability of states
            initial_state: Starting state for the chain
            adapt_step_size: Whether to adapt step size during sampling
            adapt_rate: Rate of step size adaptation
            target_accept_rate: Target acceptance rate for adaptation
            min_step_size: Minimum allowed step size
            max_step_size: Maximum allowed step size
        """
        self.log_prob_fn = log_prob_fn
        self.adapt_rate = adapt_rate
        self.target_accept_rate = target_accept_rate
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size

        # Initialize state
        self.current_state = initial_state.clone().detach()
        self.step_size = torch.tensor(step_size, dtype=torch.float32)

        # Compute initial log probability
        try:
            self.current_log_prob = self.log_prob_fn(self.current_state)
        except Exception as e:
            raise ValueError(f"Failed to compute initial log probability: {e}")

        # Statistics tracking
        self.n_samples = 0
        self.n_accepted = 0

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate input parameters."""
        if not callable(self.log_prob_fn):
            raise TypeError("log_prob_fn must be callable")

        if not isinstance(self.current_state, torch.Tensor):
            raise TypeError("initial_state must be a torch.Tensor")

        if self.step_size <= 0:
            raise ValueError("step_size must be positive")

        if not 0 < self.target_accept_rate < 1:
            raise ValueError("target_accept_rate must be between 0 and 1")

        if self.adapt_rate <= 0:
            raise ValueError("adapt_rate must be positive")

    def step(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one Metropolis-Hastings step.

        Returns:
            Tuple of (current_state, current_log_prob, was_accepted)
        """
        # Detach current state to avoid gradient accumulation
        self.current_state = self.current_state.detach()
        self.current_log_prob = self.current_log_prob.detach()

        # Generate proposal
        noise = torch.randn_like(self.current_state)
        proposed_state = self.current_state + noise * self.step_size

        # Compute proposal log probability
        try:
            proposed_log_prob = self.log_prob_fn(proposed_state)
        except Exception as e:
            warnings.warn(f"Failed to compute proposal log probability: {e}")
            return self.current_state, self.current_log_prob, False

        # Check for invalid log probabilities
        if torch.isnan(proposed_log_prob).any() or torch.isinf(proposed_log_prob).any():
            warnings.warn("Invalid log probability encountered in proposal")
            return self.current_state, self.current_log_prob, False

        # Compute acceptance probability
        log_prob_diff = proposed_log_prob - self.current_log_prob

        # Vectorized acceptance decision
        log_uniform = torch.log(torch.clamp(torch.rand_like(log_prob_diff), min=1e-8))
        accept_mask = log_uniform < log_prob_diff

        # Update accepted states
        if accept_mask.any():
            self.current_state = torch.where(
                (
                    accept_mask.unsqueeze(-1)
                    if accept_mask.dim() < self.current_state.dim()
                    else accept_mask
                ),
                proposed_state,
                self.current_state,
            )
            self.current_log_prob = torch.where(
                accept_mask, proposed_log_prob, self.current_log_prob
            )

        # Update statistics
        self.n_samples += 1
        accepted = accept_mask.float().mean().item()
        self.n_accepted += accepted

        # Adapt step size
        self._adapt_step_size(accepted)

        return self.current_state, self.current_log_prob

    def warmup(self, warmup: int) -> None:
        """Warmups the MCMC without returning anything."""
        if not isinstance(warmup, int):
                raise TypeError(f"warmup must be an integer, got {type(warmup).__name__}")
        if warmup < 0:
                raise ValueError("warmup must be a non-negative integer")
        
        with torch.no_grad():
            for _ in range(warmup):
                self.step()

    def _adapt_step_size(self, accept_rate: float):
        """Adapt step size based on acceptance rate."""
        adaptation = (
            torch.tensor(accept_rate - self.target_accept_rate) * self.adapt_rate
        )
        self.step_size = torch.clamp(
            self.step_size * torch.exp(adaptation),
            min=self.min_step_size,
            max=self.max_step_size,
        )

    @property
    def acceptance_rate(self) -> float:
        """Current acceptance rate."""
        return self.n_accepted / max(self.n_samples, 1)

    @property
    def current_step_size(self) -> float:
        """Current step size."""
        return self.step_size.item()

    def diagnostics(self) -> dict:
        """Get diagnostic information about the sampler."""
        return {
            "n_samples": self.n_samples,
            "n_accepted": self.n_accepted,
            "acceptance_rate": self.acceptance_rate,
            "current_step_size": self.current_step_size,
            "target_accept_rate": self.target_accept_rate,
        }
