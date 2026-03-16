"""
Dynamic Survival Model: Landmarking & Time-Varying Covariates.

Handles:
  - Conditional survival P(T > t + Δ | T > t, H(t))
  - Landmark superposition: predict from multiple landmarks
  - Time-varying risk factors
  - RNN-based dynamic update

Reference:
  - van Houwelingen & Putter (2007) - Dynamic Predicting Survival
  - Jewell et al. (2009) - Partly Conditional Survival Models for Longitudinal Data

Architecture:
┌───────────────────────────────────────┐
│   Dynamic Survival Model               │
├───────────────────────────────────────┤
│                                       │
│  Landmark-based Approach:             │
│                                       │
│  For prediction from landmark time t: │
│                                       │
│  ┌──────────────────────────────────┐ │
│  │ Select history up to landmark    │ │
│  │  H(t) = {measurements up to t}  │ │
│  └─────────────┬────────────────────┘ │
│                │                      │
│  ┌─────────────▼────────────────────┐ │
│  │ RNN with recent covariate info   │ │
│  │  - LSTM encodes H(t)            │ │
│  │  - Attention on recent changes  │ │
│  └─────────────┬────────────────────┘ │
│                │                      │
│  ┌─────────────▼────────────────────┐ │
│  │ Conditional Risk Head            │ │
│  │  P(T > t+Δ | T>t, H(t))        │ │
│  └───────────────────────────────────┘ │
│                                       │
│  Superposition: Average over landmarks
│
└───────────────────────────────────────┘
"""

from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model_base import BaseTemporalModel, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class DynamicSurvivalConfig(TrainingConfig):
    """Dynamic survival configuration."""

    # Architecture
    num_features: int = 10
    lstm_hidden_dim: int = 128
    fc_dim: int = 256
    num_attention_heads: int = 4
    dropout: float = 0.1

    # Landmarks
    landmark_times: Tuple[int, ...] = (0, 3, 6, 12)  # months
    prediction_horizon: int = 12  # predict Δ months ahead
    use_superposition: bool = True

    # Loss
    loss_type: str = "bce"  # 'bce' or 'cox'


class LandmarkWindow(nn.Module):
    """
    Extract and process data around landmark time.

    For landmark t and history H(t):
      - Include observations in [t - lookback, t]
      - Create time offsets relative to landmark
      - Mask future observations
    """

    def __init__(
        self,
        lookback_days: int = 365,
    ):
        super().__init__()
        self.lookback_days = lookback_days

    def forward(
        self,
        times: torch.Tensor,  # (batch, seq_len)
        values: torch.Tensor,  # (batch, seq_len, features)
        landmark_time: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract observations in [landmark - lookback, landmark].

        Args:
            times: Observation times in months
            values: Feature values
            landmark_time: Landmark time in months

        Returns:
            window_values: (batch, window_len, features)
            window_times: (batch, window_len) relative to landmark
        """
        batch_size, seq_len, features = values.shape

        # Create mask for observations in window
        mask = (times >= (landmark_time - self.lookback_days / 30.0)) & (times <= landmark_time)

        # Relative times
        window_times = times - landmark_time  # Negative before landmark

        # Zero out masked positions
        window_values = values * mask.unsqueeze(-1).float()

        return window_values, window_times


class ConditionalRiskHead(nn.Module):
    """
    Head for conditional survival probability P(T > t + Δ | T > t, H(t)).

    Uses Cox-like proportional hazards or simple MLP.
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim // 2)
        self.fc_risk = nn.Linear(fc_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) - encoded history

        Returns:
            risk_scores: (batch, 1) - log hazard or logit
        """
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        log_hazard = self.fc_risk(h)
        return log_hazard


class DynamicSurvivalModel(BaseTemporalModel):
    """
    Dynamic survival model with landmark-based predictions.

    Estimates conditional probability P(T > t + Δ | T > t, H(t))
    where H(t) is patient history up to landmark time.
    """

    def __init__(self, config: DynamicSurvivalConfig):
        super().__init__(config, config.num_features, 1)
        self.config = config

        # Landmark window
        self.landmark_window = LandmarkWindow(lookback_days=365)

        # LSTM encoder for history
        self.lstm = nn.LSTM(
            config.num_features,
            config.lstm_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=config.dropout if config.lstm_hidden_dim > 1 else 0.0,
        )

        # Attention on recent observations
        self.attention = nn.MultiheadAttention(
            config.lstm_hidden_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )

        # Time encoding
        self.time_embedding = nn.Linear(1, config.lstm_hidden_dim)

        # Shared representation
        self.fc_shared = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, config.fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fc_dim, config.fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Conditional risk head
        self.risk_head = ConditionalRiskHead(
            config.fc_dim,
            config.fc_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        seq_lengths: torch.Tensor,
        landmark_times: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            times: (batch, seq_len) observation times in months
            seq_lengths: (batch,) actual sequence lengths
            landmark_times: (batch,) or None - use config landmark_times

        Returns:
            predictions: (batch, num_landmarks) conditional survival probs
        """
        batch_size = x.size(0)

        if landmark_times is None:
            # Use predefined landmarks
            landmark_times = torch.FloatTensor(self.config.landmark_times).to(x.device)

        predictions = []

        # Predict from each landmark
        for landmark_t in landmark_times:
            # Extract window around landmark
            window_values, window_times = self.landmark_window(
                times,
                x,
                landmark_t.item(),
            )

            # Encode history
            packed = pack_padded_sequence(
                window_values,
                seq_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)

            # Time-aware attention on recent observations
            time_emb = self.time_embedding(
                window_times.unsqueeze(-1)
            )  # (batch, seq_len, lstm_hidden_dim)
            lstm_out_with_time = lstm_out + time_emb

            # Self-attention
            attn_out, _ = self.attention(
                lstm_out_with_time,
                lstm_out_with_time,
                lstm_out_with_time,
            )

            # Get final representation
            final_h = h_n[-1]  # (batch, lstm_hidden_dim)

            # Shared representation
            shared_rep = self.fc_shared(final_h)  # (batch, fc_dim)

            # Conditional risk
            risk = self.risk_head(shared_rep)  # (batch, 1)

            # Convert risk to survival probability
            if self.config.loss_type == "bce":
                survival_prob = torch.sigmoid(risk)  # P(T > t + Δ)
            else:
                # Cox-like: exp(-hazard * time)
                survival_prob = torch.exp(-F.softplus(risk) * self.config.prediction_horizon / 12.0)

            predictions.append(survival_prob)

        # Stack predictions
        if self.config.use_superposition and len(predictions) > 1:
            # Average over landmarks
            output = torch.cat(predictions, dim=1).mean(dim=1, keepdim=True)
        else:
            output = torch.cat(predictions, dim=1)

        return output

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute conditional survival loss.

        Args:
            y_pred: (batch, num_landmarks) predicted survival probabilities
            batch: Dict with event times and indicators

        Returns:
            loss
        """
        event_times = batch["y"]  # (batch,)
        events = batch["events"]  # (batch,) binary indicator

        if self.config.loss_type == "bce":
            # Convert to future events: did event occur within prediction_horizon?
            future_events = torch.zeros_like(events)

            for i, (time, event) in enumerate(zip(event_times, events)):
                if event == 1 and time <= self.config.prediction_horizon:
                    future_events[i] = 1

            # BCE loss
            y_pred_mean = y_pred.mean(dim=1) if y_pred.dim() > 1 else y_pred.squeeze()
            loss = F.binary_cross_entropy(
                torch.clamp(y_pred_mean, 1e-7, 1 - 1e-7),
                future_events.float(),
                weight=1.0 - events,  # Weight censored samples
            )

        elif self.config.loss_type == "cox":
            # Negative Cox partial likelihood
            loss = self._negative_cox_loss(y_pred, event_times, events)

        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")

        return loss

    def _negative_cox_loss(
        self,
        log_hazards: torch.Tensor,
        times: torch.Tensor,
        events: torch.Tensor,
    ) -> torch.Tensor:
        """Negative Cox partial likelihood."""
        # Simple implementation - full Cox loss is more complex
        n = len(times)

        loss = torch.tensor(0.0, device=log_hazards.device)

        for i in range(n):
            if events[i] == 0:
                continue

            # Risk set: j with T_j >= T_i
            risk_set_mask = times >= times[i]
            risk_scores = log_hazards[risk_set_mask]

            # Log partial likelihood
            log_lik = log_hazards[i] - torch.logsumexp(risk_scores, dim=0)
            loss += -log_lik

        return loss / (torch.sum(events) + 1e-8)

    def get_conditional_survival(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        seq_lengths: torch.Tensor,
        landmark_times: torch.Tensor,
        prediction_horizons: Optional[Tuple[int, ...]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Get conditional survival for multiple prediction horizons.

        Returns:
            Dict mapping horizon -> (batch,) survival probabilities
        """
        if prediction_horizons is None:
            prediction_horizons = (3, 6, 12)

        results = {}

        for horizon in prediction_horizons:
            # Temporarily set prediction horizon
            original_horizon = self.config.prediction_horizon
            self.config.prediction_horizon = horizon

            with torch.no_grad():
                survival = self.forward(x, times, seq_lengths, landmark_times)
                results[horizon] = survival.squeeze(-1)

            self.config.prediction_horizon = original_horizon

        return results
