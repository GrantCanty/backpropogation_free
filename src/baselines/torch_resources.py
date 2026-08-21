"""Public resource accounting for optional PyTorch controls."""

from __future__ import annotations

from torch import Tensor, nn
from torch.optim import Optimizer


def tensor_nbytes(tensor: Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def module_nbytes(module: nn.Module) -> int:
    return sum(tensor_nbytes(parameter) for parameter in module.parameters())


def optimizer_nbytes(optimizer: Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, Tensor):
                total += tensor_nbytes(value)
    return total
