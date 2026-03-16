"""
Temporal Fusion Transformer (TFT) for MM progression forecasting.
 
Handles:
  - Variable-length irregular time series
  - Multi-horizon predictions (3, 6, 12 months)
  - Attention-based interpretability
  - Feature importance via attention weights
 
Reference: Lim et al. (2021) - Temporal Fusion Transformers for Interpretable
Multi-horizon Time Series Forecasting
 
Architecture:
┌────────────────────────────────────┐
│   Temporal Fusion Transformer      │
├────────────────────────────────────┤
│                                    │
│  Input: (batch, seq_len, features) │
│         + time information         │
│         + static features          │
│                                    │
│  ┌─────────────────────────────────┐
│  │ Embedding Layer                 │
│  │  - Continuous var embedding    │
│  │  - Categorical encoding        │
│  │  - Time encoding               │
│  └──────┬──────────────────────────┘
│         │
│  ┌──────▼──────────────────────────┐
│  │ LSTM Encoder (past)             │
│  │  - Encode historical context   │
│  └──────┬──────────────────────────┘
│         │
│  ┌──────▼──────────────────────────┐
│  │ Temporal Self-Attention         │
│  │  - Context attention           │
│  │  - Temporal feature selection  │
│  └──────┬──────────────────────────┘
│         │
│  ┌──────▼──────────────────────────┐
│  │ LSTM Decoder (future)          │
│  │  - Generate forecast context   │
│  └──────┬──────────────────────────┘
│         │
│  ┌──────▼──────────────────────────┐
│  │ Multi-Horizon Prediction Head   │
│  │  - Output (batch, horizons)    │
│  └──────────────────────────────────┘
│
└────────────────────────────────────┘
"""
 
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
 
from .model_base import BaseTemporalModel, TrainingConfig
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class TFTConfig(TrainingConfig):
    """TFT-specific configuration."""
 
    # Architecture
    num_features: int = 10
    num_static_features: int = 5
    embedding_dim: int = 64
    lstm_hidden_dim: int = 128
    num_attention_heads: int = 4
    num_transformer_layers: int = 2
    ffn_dim: int = 256
 
    # Prediction
    prediction_horizons: Tuple[int, ...] = (3, 6, 12)  # months
    dropout: float = 0.1
 
    # Loss
    loss_type: str = "mse"  # 'mse', 'huber', 'quantile'
    quantile_levels: Tuple[float, ...] = (0.1, 0.5, 0.9)  # for quantile loss
 
 
class VariableSelectionNetwork(nn.Module):
    """
    Feature selection network: learns which features are important.
 
    Uses gated mechanism: output = v_x * x where v_x is learned gate.
    """
 
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) or (batch, input_dim)
 
        Returns:
            (batch, seq_len, output_dim) or (batch, output_dim)
        """
        v = torch.sigmoid(self.fc1(x))
        x_processed = self.fc2(x)
        return self.dropout(v * x_processed)
 
 
class TemporalAttentionLayer(nn.Module):
    """
    Multi-head attention with relative positional encoding.
    """
 
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0
 
        self.scale = self.head_dim ** -0.5
 
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
 
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, dim)
            mask: (batch, seq_len) binary mask for padding
 
        Returns:
            output: (batch, seq_len, dim)
            attention: (batch, num_heads, seq_len, seq_len)
        """
        B, N, C = x.shape
 
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
 
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
 
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))
 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
 
        # Aggregate values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
 
        return x, attn
 
 
class TemporalFusionTransformer(BaseTemporalModel):
    """
    Temporal Fusion Transformer for MM progression forecasting.
 
    Multi-horizon predictions with attention-based interpretability.
    """
 
    def __init__(self, config: TFTConfig):
        super().__init__(config, config.num_features, len(config.prediction_horizons))
        self.config = config
 
        # Embeddings
        self.time_embedding = nn.Linear(1, config.embedding_dim)
        self.feature_embedding = nn.Linear(config.num_features, config.embedding_dim)
        self.static_embedding = nn.Linear(
            config.num_static_features,
            config.embedding_dim
        )
 
        # Variable selection
        self.temporal_vsn = VariableSelectionNetwork(
            config.embedding_dim,
            config.embedding_dim,
            dropout=config.dropout,
        )
        self.static_vsn = VariableSelectionNetwork(
            config.embedding_dim,
            config.embedding_dim,
            dropout=config.dropout,
        )
 
        # LSTM Encoder (past)
        self.lstm_encoder = nn.LSTM(
            config.embedding_dim,
            config.lstm_hidden_dim,
            batch_first=True,
            dropout=config.dropout if config.lstm_hidden_dim > 1 else 0.0,
        )
 
        # LSTM Decoder (future)
        self.lstm_decoder = nn.LSTM(
            config.embedding_dim,
            config.lstm_hidden_dim,
            batch_first=True,
            dropout=config.dropout if config.lstm_hidden_dim > 1 else 0.0,
        )
 
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.lstm_hidden_dim,
            nhead=config.num_attention_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_transformer_layers,
        )
 
        # Prediction head
        self.fc_out = nn.Sequential(
            nn.Linear(config.lstm_hidden_dim, config.ffn_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, len(config.prediction_horizons)),
        )
 
        self.attention_weights = None
 
    def forward(
        self,
        x: torch.Tensor,
        times: torch.Tensor,
        seq_lengths: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, features)
            times: (batch, seq_len, 1) time in months
            seq_lengths: (batch,) actual sequence lengths
            static: (batch, static_features) optional static features
 
        Returns:
            predictions: (batch, num_horizons)
        """
        batch_size, seq_len, _ = x.shape
 
        # Ensure times has correct shape
        if times.dim() == 2:
            times = times.unsqueeze(-1)
 
        # Normalize times to [0, 1]
        times_normalized = times / (times.max() + 1e-8)
 
        # Embeddings
        time_emb = self.time_embedding(times_normalized)  # (B, seq_len, emb_dim)
        feature_emb = self.feature_embedding(x)  # (B, seq_len, emb_dim)
        temporal_emb = feature_emb + time_emb
 
        # Variable selection for temporal
        temporal_selected = self.temporal_vsn(temporal_emb)
 
        # Static features
        if static is not None:
            static_emb = self.static_embedding(static)  # (B, emb_dim)
            static_selected = self.static_vsn(static_emb)
            static_selected = static_selected.unsqueeze(1).expand(
                -1, seq_len, -1
            )  # (B, seq_len, emb_dim)
            temporal_selected = temporal_selected + static_selected
 
        # LSTM Encoder
        packed = pack_padded_sequence(
            temporal_selected,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        encoder_out, (h_n, c_n) = self.lstm_encoder(packed)
        encoder_out, _ = pad_packed_sequence(encoder_out, batch_first=True)
 
        # Transformer for temporal attention
        # Create padding mask
        mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < seq_lengths.unsqueeze(1)
        transformer_out = self.transformer(
            encoder_out,
            src_key_padding_mask=~mask,
        )
 
        # Get final hidden state
        batch_idx = torch.arange(batch_size, device=x.device)
        final_hidden = transformer_out[batch_idx, seq_lengths - 1]  # (B, lstm_hidden_dim)
 
        # Predictions
        predictions = self.fc_out(final_hidden)  # (B, num_horizons)
 
        return predictions
 
    def compute_loss(self, y_pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        """
        Compute loss based on configuration.
 
        Supports MSE, Huber, and quantile loss.
        y_pred: (batch, num_horizons), y_true: (batch,) scalar event times.
        We broadcast y_true across horizons so each horizon predicts the same target.
        """
        y_true = batch["y"]
 
        # Expand y_true to match multi-horizon output shape
        if y_pred.dim() == 2 and y_true.dim() == 1:
            y_true = y_true.unsqueeze(1).expand_as(y_pred)
 
        if self.config.loss_type == "mse":
            return F.mse_loss(y_pred, y_true)
 
        elif self.config.loss_type == "huber":
            return F.huber_loss(y_pred, y_true, delta=1.0)
 
        elif self.config.loss_type == "quantile":
            # Quantile loss for prediction intervals
            losses = []
            for i, q in enumerate(self.config.quantile_levels):
                diff = y_true - y_pred[:, i:i+1]
                loss_q = torch.where(
                    diff >= 0,
                    q * diff,
                    (q - 1) * diff,
                )
                losses.append(loss_q.mean())
            return torch.stack(losses).mean()
 
        else:
            raise ValueError(f"Unknown loss_type: {self.config.loss_type}")
 
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Return attention weights for interpretability."""
        return self.attention_weights