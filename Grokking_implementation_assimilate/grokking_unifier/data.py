"""
Unified data generation for grokking experiments.

Supports modular arithmetic operations with a registry pattern
so users can easily add new operations.

Repos unified:
  - 7vik/grokking (modular arithmetic datasets)
  - neelnanda-io/Grokking (mod addition, specific splits)
  - Tikquuss/grokking_fda (various group operations)
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Callable


# ---------------------------------------------------------------------------
# Operation Registry
# ---------------------------------------------------------------------------

_OPERATION_REGISTRY: Dict[str, Callable] = {}


def register_operation(name: str):
    """Decorator to register a new modular arithmetic operation.

    Example::

        @register_operation("my_custom_op")
        def _my_op(a: int, b: int, p: int) -> int:
            return (a ** 2 + b) % p
    """
    def decorator(fn: Callable):
        _OPERATION_REGISTRY[name] = fn
        return fn
    return decorator


def get_available_operations() -> List[str]:
    """Return list of all registered operation names."""
    return list(_OPERATION_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Built-in Operations
# ---------------------------------------------------------------------------

@register_operation("addition")
def _op_addition(a: int, b: int, p: int) -> int:
    return (a + b) % p


@register_operation("subtraction")
def _op_subtraction(a: int, b: int, p: int) -> int:
    return (a - b) % p


@register_operation("multiplication")
def _op_multiplication(a: int, b: int, p: int) -> int:
    return (a * b) % p


@register_operation("division")
def _op_division(a: int, b: int, p: int) -> Optional[int]:
    """Division as a * b^{-1} mod p.  Returns None when b == 0."""
    if b == 0:
        return None  # filtered out during dataset construction
    return (a * pow(b, p - 2, p)) % p


@register_operation("x2_plus_y2")
def _op_x2_plus_y2(a: int, b: int, p: int) -> int:
    return (a * a + b * b) % p


@register_operation("x2_plus_xy_plus_y2")
def _op_x2_plus_xy_plus_y2(a: int, b: int, p: int) -> int:
    return (a * a + a * b + b * b) % p


@register_operation("x3_plus_xy")
def _op_x3_plus_xy(a: int, b: int, p: int) -> int:
    return (a * a * a + a * b) % p


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ModularArithmeticDataset(Dataset):
    """Dataset for modular arithmetic tasks.

    Generates all valid ``(a, b)`` pairs with ``a, b ∈ [0, p)`` and computes
    ``target = op(a, b) mod p``.

    Parameters
    ----------
    operation : str
        Name of the registered operation (see ``get_available_operations()``).
    p : int
        The prime modulus.
    train_fraction : float
        Fraction of total pairs used for training.
    seed : int
        Random seed for the train / val split.
    """

    def __init__(
        self,
        operation: str = "addition",
        p: int = 97,
        train_fraction: float = 0.3,
        seed: int = 42,
    ):
        super().__init__()
        self.operation = operation
        self.p = p
        self.train_fraction = train_fraction
        self.seed = seed

        if operation not in _OPERATION_REGISTRY:
            raise ValueError(
                f"Unknown operation '{operation}'. "
                f"Available: {get_available_operations()}"
            )

        op_fn = _OPERATION_REGISTRY[operation]

        # Generate all valid pairs
        inputs, targets = [], []
        for a in range(p):
            for b in range(p):
                result = op_fn(a, b, p)
                if result is not None:
                    inputs.append([a, b])
                    targets.append(result)

        self.inputs = torch.tensor(inputs, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.num_samples = len(self.inputs)

        # Deterministic train / val split
        rng = np.random.RandomState(seed)
        indices = rng.permutation(self.num_samples)
        split = int(self.num_samples * train_fraction)
        self.train_indices = indices[:split]
        self.val_indices = indices[split:]

    # -- PyTorch Dataset interface ------------------------------------------

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]

    # -- Convenience helpers ------------------------------------------------

    def get_train_loader(
        self, batch_size: int = 512, shuffle: bool = True
    ) -> DataLoader:
        return DataLoader(
            Subset(self, self.train_indices),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    def get_val_loader(self, batch_size: int = 512) -> DataLoader:
        return DataLoader(
            Subset(self, self.val_indices),
            batch_size=batch_size,
            shuffle=False,
        )

    def summary(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "p": self.p,
            "total_samples": self.num_samples,
            "train_samples": len(self.train_indices),
            "val_samples": len(self.val_indices),
            "train_fraction": self.train_fraction,
        }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"ModularArithmeticDataset(op={s['operation']}, p={s['p']}, "
            f"train={s['train_samples']}, val={s['val_samples']})"
        )
