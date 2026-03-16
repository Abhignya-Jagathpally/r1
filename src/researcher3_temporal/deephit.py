"""
DeepHit: Deep Learning for Competing Risks.

Handles:
  - Competing risks (progression, death, relapse)
  - Discrete-time survival with ranking loss
  - Negative log-likelihood
  - Time-varying risk curves

Reference: Lee et al. (2018) - DeepHit: A Deep Learning Approach for
Survival Analysis with Competing Risks

Architecture:
┌──────────────────────────────────────┐
│         DeepHit Model                │
├──────────────────────────────────────┤
│                                      │
│  Input: sequences + times + events  │
│                                      │
│  ┌──────────────────────────────────┐
│  │ Shared LSTM Encoder              │
│  │  - Process temporal sequences   │
│  │  - Learn patient representation │
│  └────────┬─────────────────────────┘
│           │
│  ┌────────▼─────────────────────────┐
│  │ Cause-Specific Subnetworks       │
│  │  - Progression subnet            │
│  │  - Death subnet                  │
│  │  - Relapse subnet                │
│  │                                  │
│  │ Each outputs P(T ∈ [t,t+1] | t)  │
│  └────────┬─────────────────────────┘
│           │
│  ┌────────▼─────────────────────────┐
│  │ Survival Probabilities           │
│  │  S(t) = 1 - Σ P(event|t)        │
│  └──────────────────────────────────┘
│
│ Loss: Ranking Loss + NLL
│
└──────────────────────────────────────┘
"""

from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model_base import BaseTemporalModel, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class DeepHitConfig(TrainingConfig):
    """DeepHit configuration."""

    # Architecture
    num_features: int = 10
    lstm_hidden_dim: int = 128
    shared_fc_dim: int = 256
    cause_fc_dim: int = 128
    num_causes: int = 3  # progression, death, relapse

    # Time discretization
    num_time_steps: int = 60  # discretize time into bins
    time_bins: Optional[Tuple[int, ...]] = None  # e.g., (0, 3, 6, 12, 24, 60)

    # Loss
    alpha: float = 0.5  # weight for ranking loss vs NLL
    dropout: float = 0.1


class CauseSpecificNetwork(nn.Module):
    """
    Cause-specific subnetwork for one competing risk.

    Outputs P(T ∈ [t, t+1] | T ≥ t, cause) for each time step.
    """

    def __init__(
        self,
        input_dim: int,
        fc_dim: int,
        num_time_steps: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc_out = nn.Linear(fc_dim, num_time_steps)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim)

        Returns:
            (batch, num_time_steps) - probabilities P(T ∈ [t,t+1])
        """
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        logits = self.fc_out(h)
        return torch.softmax(logits, dim=-1)


class DeepHit(BaseTemporalModel):
    """
    DeepHit model for MM competing risks.

    Predicts multiple event types (progression, death, relapse)
    with discrete-time survival framework.
    """

    def __init__(self, config: DeepHitConfig):
        super().__init__(config, config.num_features, config.num_causes)
        self.config = config

        # Create time bins if not provided
        if config.time_bins is None:
            self.time_bins = torch.linspace(0, 60, config.num_time_steps + 1)
        else:
            self.time_bins = torch.FloatTensor(config.time_bins)

        # LSTM encoder
        self.lstm = nn.LSTM(
            config.num_features,
            config.lstm_hidden_dim,
            batch_first=True,
            dropout=config.dropout if config.lstm_hidden_dim > 1 else 0.0,
        )

        # Shared representation
        self.fc_shared = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, config.shared_fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.shared_fc_dim, config.shared_fc_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        # Cause-specific networks
        self.cause_networks = nn.ModuleList([
            CauseSpecificNetwork(
                config.shared_fc_dim,
                config.cause_fc_dim,
                config.num_time_steps,
                dropout=config.dropout,
            )
            for _ in range(config.num_causes)
        ])

    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            seq_lengths: (batch,) actual sequence lengths

        Returns:
            (batch, num_causes, num_time_steps) - probabilities for each cause
        """
        batch_size = x.size(0)

        # LSTM encoding
        packed = pack_padded_sequence(
            x,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (h_n, _) = self.lstm(packed)
        lstm_out = h_n[-1]  # (batch, lstm_hidden_dim)

        # Shared representation
        shared_rep = self.fc_shared(lstm_out)  # (batch, shared_fc_dim)

        # Cause-specific predictions
        cause_probs = []
        for cause_net in self.cause_networks:
            prob = cause_net(shared_rep)  # (batch, num_time_steps)
            cause_probs.append(prob)

        # Stack into (batch, num_causes, num_time_steps)
        output = torch.stack(cause_probs, dim=1)

        return output

    def compute_loss(
        self,
        y_pred: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute DeepHit loss: ranking loss + negative log-likelihood.

        Args:
            y_pred: (batch, num_causes, num_time_steps)
            batch: Dict with 'y' (event times), 'events' (indicators), 'event_types'

        Returns:
            Combined loss
        """
        event_times = batch["y"]  # (batch,)
        events = batch["events"]  # (batch,) binary
        event_types = batch["event_types"]  # (batch,) cause index

        batch_size, num_causes, num_time_steps = y_pred.shape

        # Discretize event times to bins
        time_indices = self._discretize_times(event_times)

        # NLL Loss
        nll_loss = torch.tensor(0.0, device=y_pred.device)
        ranking_loss = torch.tensor(0.0, device=y_pred.device)

        for i in range(batch_size):
            if events[i] == 0:
                # Censored: P(T > t)
                t_idx = time_indices[i]
                cause_probs = y_pred[i]  # (num_causes, num_time_steps)

                # Survival probability
                survival_prob = 1.0 - torch.sum(cause_probs[:, :t_idx], dim=0)
                survival_prob = torch.clamp(survival_prob, 1e-7, 1.0)
                nll_loss += -torch.log(survival_prob[-1])

            else:
                # Event: P(T ∈ [t, t+1], cause)
                t_idx = time_indices[i]
                cause_idx = int(event_types[i].item())

                # Get probability of event at this time and cause
                event_prob = y_pred[i, cause_idx, t_idx]
                event_prob = torch.clamp(event_prob, 1e-7, 1.0)
                nll_loss += -torch.log(event_prob)

                # Ranking loss: P(event at t < t') for all unobserved times
                for t_prime in range(t_idx + 1, num_time_steps):
                    surv_t = torch.sum(y_pred[i, :, :t_idx])
                    event_t_prime = y_pred[i, cause_idx, t_prime]
                    ranking_loss += F.relu(event_t_prime - surv_t + 0.1)

        nll_loss = nll_loss / batch_size
        ranking_loss = ranking_loss / (batch_size * num_time_steps) if ranking_loss > 0 else ranking_loss

        total_loss = (1 - self.config.alpha) * nll_loss + self.config.alpha * ranking_loss

        return total_loss

    def _discretize_times(self, event_times: torch.Tensor) -> torch.LongTensor:
        """Convert continuous times to discrete bins."""
        # Searchsorted to find which bin each event time falls into
        bins = self.time_bins.to(event_times.device)
        indices = torch.searchsorted(bins, event_times.squeeze())
        return torch.clamp(indices, 0, self.config.num_time_steps - 1)

    def get_survival_curves(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute survival curves S(t) = P(T > t) for each cause.

        Returns:
            (batch, num_causes, num_time_steps) survival probabilities
        """
        with torch.no_grad():
            cause_probs = self.forward(x, seq_lengths)  # (B, causes, time_steps)

            # Cumulative incidence: P(T ≤ t, cause)
            cumulative_probs = torch.cumsum(cause_probs, dim=2)

            # Overall survival: 1 - sum of cumulative incidences
            overall_survival = 1.0 - torch.sum(cumulative_probs, dim=1, keepdim=True)
            overall_survival = torch.clamp(overall_survival, 0, 1)

            # Cause-specific survival
            cause_survival = (
                overall_survival * cause_probs / (torch.sum(cause_probs, dim=1, keepdim=True) + 1e-8)
            )

            return cause_survival

    def get_risk_scores(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor,
        time_horizon: int,
    ) -> torch.Tensor:
        """
        Get cumulative risk at time horizon.

        Args:
            x: (batch, seq_len, features)
            seq_lengths: (batch,)
            time_horizon: months

        Returns:
            (batch, num_causes) cumulative risk probabilities
        """
        with torch.no_grad():
            cause_probs = self.forward(x, seq_lengths)  # (B, causes, time_steps)

            # Find index closest to time_horizon
            time_idx = min(
                int(time_horizon * self.config.num_time_steps / 60),
                self.config.num_time_steps - 1
            )

            # Cumulative probability at time_idx
            cumulative_risk = torch.sum(cause_probs[:, :, :time_idx+1], dim=2)
            cumulative_risk = torch.clamp(cumulative_risk, 0, 1)

            return cumulative_risk
