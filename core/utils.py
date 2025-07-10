import torch
import numpy as np
from typing import Tuple


def _tril_from_flat(flat: torch.Tensor, n: int) -> torch.Tensor:
    """
    Reconstruct a lower triangular matrix from its flattened representation.

    Args:
        flat: Flattened lower triangular matrix elements
        n: Dimension of the square matrix

    Returns:
        torch.Tensor: Lower triangular matrix [n x n]
    """
    L = torch.zeros(n, n, dtype=flat.dtype)
    iu = torch.tril_indices(n, n)
    L[iu[0], iu[1]] = flat
    return L


def _precision_matrix_from_cholesky(L: torch.Tensor) -> torch.Tensor:
    """
    Compute precision matrix from Cholesky factor.

    Args:
        L: Lower triangular Cholesky factor

    Returns:
        torch.Tensor: Precision matrix L @ L.T
    """
    # Exponentiate diagonal elements for positive definiteness
    L_exp = L.clone()
    diag_indices = torch.arange(L.shape[0])
    L_exp[diag_indices, diag_indices] = torch.exp(L[diag_indices, diag_indices])

    # Compute precision matrix
    precision = L_exp @ L_exp.T

    return precision


def _legendre_quad(n_quad: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get Legendre quadrature nodes and weights.

    Args:
        n_quad: Number of quadrature points

    Returns:
        Tuple of (nodes, weights) as torch tensors
    """
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    return (
        torch.tensor(nodes, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    )


class Fun:
    """
    A callable wrapper for functions with specified input and output dimensions.

    This class provides a clean interface for mathematical functions used in
    the joint model, ensuring proper dimension tracking and validation.
    """

    def __init__(self, fun, input_dim, output_dim):
        """
        Initialize the function wrapper.

        Args:
            fun: The callable function
            input_dim: Input dimension specification
            output_dim: Output dimension specification
        """
        self.fun = fun
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Validate that function is callable
        if not callable(fun):
            raise TypeError("fun must be callable")

    def __call__(self, *args):
        """Call the wrapped function with input validation."""
        try:
            result = self.fun(*args)
            return result
        except Exception as e:
            raise RuntimeError(f"Error in function call: {e}")
