"""
grokking_unifier — A unified framework for studying neural network grokking.

Brings together insights from:
  • 7vik/grokking          – modular arithmetic datasets & Transformer
  • neelnanda-io/Grokking  – mechanistic interpretability setup
  • ironjr/grokfast        – gradient filtering to accelerate grokking
  • nmallinar/rfm-grokking – MLP / kernel baselines
  • Tikquuss/grokking_fda  – FDA-style activation diagnostics

Quick-start::

    from grokking_unifier import (
        ModularArithmeticDataset,
        create_model,
        Trainer,
        GrokfastEMA,
        WeightNormCallback,
    )

    ds  = ModularArithmeticDataset("addition", p=97, train_fraction=0.3)
    mdl = create_model("transformer", p=97)
    opt = torch.optim.AdamW(mdl.parameters(), lr=1e-3, weight_decay=1.0)
    tr  = Trainer(mdl, ds.get_train_loader(), ds.get_val_loader(), opt,
                  callbacks=[WeightNormCallback()])
    tr.train(epochs=3000)
"""

from .data import (
    ModularArithmeticDataset,
    register_operation,
    get_available_operations,
)
from .models import (
    create_model,
    register_model,
    get_available_models,
    GrokkingTransformer,
    GrokkingMLP,
)
from .optimizers import GrokfastEMA, GrokfastMA
from .metrics import (
    MetricCallback,
    WeightNormCallback,
    GradientNormCallback,
    PerLayerNormCallback,
    ActivationVarianceCallback,
    register_metric,
    create_metric,
    get_available_metrics,
)
from .trainer import Trainer

__version__ = "0.1.0"

__all__ = [
    # Data
    "ModularArithmeticDataset",
    "register_operation",
    "get_available_operations",
    # Models
    "create_model",
    "register_model",
    "get_available_models",
    "GrokkingTransformer",
    "GrokkingMLP",
    # Optimizers
    "GrokfastEMA",
    "GrokfastMA",
    # Metrics
    "MetricCallback",
    "WeightNormCallback",
    "GradientNormCallback",
    "PerLayerNormCallback",
    "ActivationVarianceCallback",
    "register_metric",
    "create_metric",
    "get_available_metrics",
    # Trainer
    "Trainer",
]
