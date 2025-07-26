from typing import Any, Callable, TypeAlias

import torch

# Type Aliases
RegressionFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
LinkFn: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor | None, torch.Tensor], torch.Tensor
]
IndividualEffectsFn: TypeAlias = Callable[
    [torch.Tensor | None, torch.Tensor], torch.Tensor
]
BaseHazardFn: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
Trajectory: TypeAlias = list[tuple[float, Any]]
ClockMethod: TypeAlias = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
