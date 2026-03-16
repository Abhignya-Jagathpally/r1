"""
Base class for temporal models with training infrastructure.

Provides:
  - Standard training loop with validation
  - Early stopping
  - Learning rate scheduling (CosineAnnealingLR, ReduceLROnPlateau)
  - Gradient clipping and accumulation
  - Mixed precision training (AMP)
  - Checkpoint management
  - MLflow integration for experiment tracking
"""

import os
import json
from typing import Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LinearLR,
)
from torch.utils.data import DataLoader
import numpy as np

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 32
    num_epochs: int = 100
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1

    # Scheduling
    scheduler_type: str = "cosine"  # 'cosine', 'plateau', 'linear'
    warmup_epochs: int = 5

    # Mixed precision
    use_amp: bool = True
    scaler_init_scale: float = 65536.0

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # 'min' or 'max'

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    save_frequency: int = 5  # Save every N epochs

    # MLflow
    use_mlflow: bool = False
    experiment_name: str = "mm_temporal"

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class CheckpointManager:
    """Manage model checkpoints."""

    def __init__(self, checkpoint_dir: str, model_name: str = "model"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.best_metric_value = None
        self.best_epoch = 0

    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        is_best: bool = False,
    ) -> str:
        """Save checkpoint and return path."""
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
        }

        # Save latest
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, latest_path)

        # Save best
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            self.best_epoch = epoch
            logger.info(f"Saved best checkpoint at epoch {epoch}: {best_path}")

        return str(latest_path)

    def load_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer) -> int:
        """Load best checkpoint. Returns starting epoch."""
        best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"

        if not best_path.exists():
            logger.warning(f"No checkpoint found at {best_path}")
            return 0

        checkpoint = torch.load(best_path, map_location=model.device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])

        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        return checkpoint["epoch"]


class BaseTemporalModel(nn.Module):
    """
    Base class for temporal MM models.

    Provides training loop, validation, scheduling, mixed precision, etc.

    Architecture:
    ┌─────────────────────────────────────┐
    │    BaseTemporalModel (nn.Module)    │
    ├─────────────────────────────────────┤
    │ - forward(x, times, ...)            │
    │ - compute_loss(y_pred, y_true)      │
    │ - train_epoch(train_loader)         │
    │ - validate(val_loader)              │
    │ - fit(train_loader, val_loader)     │
    │ - configure_optimization()          │
    └─────────────────────────────────────┘
             ▲           ▲           ▲
             │           │           │
         DeepHit      TFT      DynamicSurvival
    """

    def __init__(
        self,
        config: TrainingConfig,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = torch.device(config.device)

        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[Any] = None
        self.scaler: Optional[GradScaler] = None
        self.checkpoint_manager: Optional[CheckpointManager] = None

        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Initialize MLflow if enabled."""
        if not self.config.use_mlflow or not HAS_MLFLOW:
            return

        mlflow.set_experiment(self.config.experiment_name)
        mlflow.log_params(asdict(self.config))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Implement in subclass."""
        raise NotImplementedError("Subclass must implement forward()")

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """Implement in subclass."""
        raise NotImplementedError("Subclass must implement compute_loss()")

    def configure_optimization(self) -> None:
        """Setup optimizer, scheduler, scaler."""
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        if self.config.scheduler_type == "cosine":
            total_steps = self.config.num_epochs
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6,
            )
        elif self.config.scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        elif self.config.scheduler_type == "linear":
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_epochs,
            )

        if self.config.use_amp:
            self.scaler = GradScaler(
                init_scale=self.config.scaler_init_scale
            )

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            batch = self._move_batch_to_device(batch)

            if batch_idx % self.config.accumulation_steps == 0:
                self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.config.use_amp and self.scaler:
                with autocast():
                    y_pred = self.forward(**batch)
                    loss = self.compute_loss(y_pred, batch)

                # Backward with scaled loss
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if batch_idx % self.config.accumulation_steps == self.config.accumulation_steps - 1:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                y_pred = self.forward(**batch)
                loss = self.compute_loss(y_pred, batch)
                loss.backward()

                if batch_idx % self.config.accumulation_steps == self.config.accumulation_steps - 1:
                    torch.nn.utils.clip_grad_norm_(
                        self.parameters(),
                        self.config.gradient_clip_norm,
                    )
                    self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}

    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate model."""
        self.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)

                if self.config.use_amp:
                    with autocast():
                        y_pred = self.forward(**batch)
                        loss = self.compute_loss(y_pred, batch)
                else:
                    y_pred = self.forward(**batch)
                    loss = self.compute_loss(y_pred, batch)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {"val_loss": avg_loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train model with early stopping.

        Returns training history.
        """
        self.to(self.device)
        self.configure_optimization()

        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(
                checkpoint_dir,
                model_name=self.__class__.__name__,
            )

        history = {
            "train_loss": [],
            "val_loss": [],
            "best_epoch": 0,
            "best_value": float('inf'),
        }

        patience_counter = 0
        best_val_loss = float('inf')

        logger.info(f"Starting training for {self.config.num_epochs} epochs")

        for epoch in range(self.config.num_epochs):
            # Training
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["train_loss"])

            # Validation
            val_metrics = self.validate(val_loader)
            history["val_loss"].append(val_metrics["val_loss"])

            # Scheduling (cosine or linear)
            if self.scheduler and self.config.scheduler_type != "plateau":
                self.scheduler.step()

            # Check for improvement
            current_val_loss = val_metrics["val_loss"]
            is_best = current_val_loss < best_val_loss

            if is_best:
                best_val_loss = current_val_loss
                patience_counter = 0
                history["best_epoch"] = epoch
                history["best_value"] = current_val_loss
            else:
                patience_counter += 1

            # Reduce LR on plateau
            if self.scheduler and self.config.scheduler_type == "plateau":
                self.scheduler.step(current_val_loss)

            # Save checkpoint
            if self.checkpoint_manager and epoch % self.config.save_frequency == 0:
                metrics = {**train_metrics, **val_metrics}
                self.checkpoint_manager.save_checkpoint(
                    epoch,
                    self,
                    self.optimizer,
                    self.scheduler,
                    metrics,
                    is_best=is_best,
                )

            # Log to MLflow
            if self.config.use_mlflow and HAS_MLFLOW:
                mlflow.log_metrics({
                    "train_loss": train_metrics["train_loss"],
                    "val_loss": val_metrics["val_loss"],
                }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train: {train_metrics['train_loss']:.4f} | "
                f"Val: {val_metrics['val_loss']:.4f} | "
                f"Patience: {patience_counter}/{self.config.early_stopping_patience}"
            )

            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        return history

    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        if isinstance(batch, dict):
            return {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
        return batch.to(self.device)
