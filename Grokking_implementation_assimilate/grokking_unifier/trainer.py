"""
Unified training loop for grokking experiments.

The :class:`Trainer` class orchestrates the training, evaluation,
Grokfast filtering, and metric-callback lifecycle in one place.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
import time

try:
    from tqdm.auto import tqdm  # auto picks notebook vs terminal widget
except ImportError:
    tqdm = None

try:
    from .metrics import MetricCallback
except (ImportError, ValueError):
    from metrics import MetricCallback


class Trainer:
    """Unified trainer for grokking experiments.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader, val_loader : DataLoader
        Data loaders for training and validation.
    optimizer : torch.optim.Optimizer
        Optimiser instance (typically AdamW with weight_decay=1.0).
    callbacks : list[MetricCallback], optional
        Metric callbacks to run at the end of each epoch.
    grokfast : GrokfastEMA | GrokfastMA | None
        Optional gradient filter applied after ``loss.backward()``.
    device : str
        ``"auto"`` picks CUDA if available, else CPU.
    log_interval : int
        Print a milestone line every *log_interval* epochs (used as
        fallback when tqdm is unavailable).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        callbacks: Optional[List[MetricCallback]] = None,
        grokfast=None,
        device: str = "auto",
        log_interval: int = 100,
    ):
        if device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.callbacks = callbacks or []
        self.grokfast = grokfast
        self.log_interval = log_interval

        # Core history (always recorded)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch": [],
            "elapsed_sec": [],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for inputs, targets in loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = F.cross_entropy(logits, targets)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.grokfast is not None:
                        self.grokfast.apply()
                    self.optimizer.step()

                total_loss += loss.item() * inputs.shape[0]
                total_correct += (logits.argmax(-1) == targets).sum().item()
                total_samples += inputs.shape[0]

        return total_loss / total_samples, total_correct / total_samples

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self, epochs: int, verbose: bool = True) -> Dict[str, Any]:
        """Run the full training loop for *epochs* epochs.

        When ``tqdm`` is installed a live progress bar is shown with
        train/val loss and accuracy updated every epoch.  Otherwise
        milestone lines are printed every ``log_interval`` epochs.

        Returns
        -------
        dict
            The combined history dictionary.
        """
        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_start(model=self.model)

        start = time.time()

        use_tqdm = verbose and tqdm is not None
        epoch_iter = range(1, epochs + 1)

        if use_tqdm:
            pbar = tqdm(epoch_iter, desc="Training", unit="ep")
        else:
            pbar = epoch_iter

        for epoch in pbar:
            train_loss, train_acc = self._run_epoch(
                self.train_loader, train=True
            )
            val_loss, val_acc = self._run_epoch(
                self.val_loader, train=False
            )

            elapsed = time.time() - start

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["epoch"].append(epoch)
            self.history["elapsed_sec"].append(elapsed)

            for cb in self.callbacks:
                cb.on_epoch_end(
                    model=self.model,
                    epoch=epoch,
                    train_loss=train_loss,
                    train_acc=train_acc,
                    val_loss=val_loss,
                    val_acc=val_acc,
                )

            # --- Live progress update ---
            if use_tqdm:
                pbar.set_postfix(
                    tr_loss=f"{train_loss:.4f}",
                    tr_acc=f"{train_acc:.4f}",
                    va_loss=f"{val_loss:.4f}",
                    va_acc=f"{val_acc:.4f}",
                )
            elif verbose and (epoch % self.log_interval == 0 or epoch == 1):
                print(
                    f"Epoch {epoch:5d}/{epochs} │ "
                    f"Train {train_loss:.4f} / {train_acc:.4f} │ "
                    f"Val {val_loss:.4f} / {val_acc:.4f} │ "
                    f"{elapsed:.1f}s"
                )

        # Notify callbacks
        for cb in self.callbacks:
            cb.on_train_end(model=self.model)

        if verbose:
            print(f"\n[DONE] Training complete in {elapsed:.1f}s")

        return self.get_all_history()

    def get_all_history(self) -> Dict[str, Any]:
        """Merge trainer history with all callback histories."""
        combined = dict(self.history)
        for cb in self.callbacks:
            for key, values in cb.get_history().items():
                combined[f"cb/{key}"] = values
        return combined
