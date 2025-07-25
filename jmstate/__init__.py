"""Multi-state joint modeling package.

This package provides tools for multi-state joint modeling with PyTorch.
"""

from .model import MultiStateJointModel
from . import types
from . import utils
from . import base_hazards

__version__ = "0.1.0"
__author__ = "Félix Laplante"
__email__ = "felixlaplante0@gmail.com"
__license__ = "MIT"

__all__ = ["MultiStateJointModel", "types", "utils", "base_hazards"]
