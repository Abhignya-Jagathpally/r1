"""
Multimodal Fusion Network: Clinical + Temporal + Genomic + Imaging.
 
Late fusion architecture with attention-based modality weighting.
 
Handles:
  - Clinical tabular features (static & dynamic)
  - Temporal sequences (lab results, imaging findings)
  - Genomic features (optional - gene expression, mutations)
  - Imaging features (optional - radiomics, segmentation results)
  - Learnable fusion with attention
  - Ablation studies (study contribution of each modality)
 
Architecture:
┌────────────────────────────────────────┐
│     Multimodal Fusion Network           │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────────────────────────────┐  │
│  │ Clinical Features (tabular)      │  │
│  │  - Demographics                  │  │
│  │  - Laboratory tests              │  │
│  │  - Staging info                  │  │
│  └───────────┬──────────────────────┘  │
│              │                         │
│  ┌───────────▼──────────────────────┐  │
│  │ Temporal Encoder                 │  │
│  │  - LSTM on sequences             │  │
│  │  - Attention on time intervals  │  │
│  └───────────┬──────────────────────┘  │
│              │                         │
│  ┌───────────▼──────────────────────┐  │
│  │ Genomic Branch (optional)        │  │
│  │  - Gene embeddings               │  │
│  │  - Pathway representations       │  │
│  └───────────┬──────────────────────┘  │
│              │                         │
│  ┌───────────▼──────────────────────┐  │
│  │ Imaging Branch (optional)        │  │
│  │  - Radiomics features            │  │
│  │  - Radiologist assessments       │  │
│  └───────────┬──────────────────────┘  │
│              │                         │
│  ┌───────────▼──────────────────────┐  │
│  │ Attention-based Fusion           │  │
│  │  - Learn modality weights        │  │
│  │  - Cross-modal interactions      │  │
│  └───────────┬──────────────────────┘  │
│              │                         │
│  ┌───────────▼──────────────────────┐  │
│  │ Prediction Head                  │  │
│  │  - Task-specific output          │  │
│  └────────────────────────────────────┘
│
│ Supports ablation: disable modalities for analysis
│
└────────────────────────────────────────┘
"""
 
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
import logging
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
 
from .model_base import BaseTemporalModel, TrainingConfig
 
logger = logging.getLogger(__name__)
 
 
@dataclass
class MultimodalFusionConfig(TrainingConfig):
    """Multimodal fusion configuration."""
 
    # Modality dimensions
    num_temporal_features: int = 10
    num_clinical_features: int = 8
    num_genomic_features: int = 0
    num_imaging_features: int = 0
 
    # Architecture
    temporal_lstm_dim: int = 128
    clinical_fc_dim: int = 64
    genomic_fc_dim: int = 64
    imaging_fc_dim: int = 64
    fusion_dim: int = 256
 
    # Attention fusion
    num_fusion_heads: int = 4
    use_cross_attention: bool = True
 
    # Ablation
    ablate_temporal: bool = False
    ablate_clinical: bool = False
    ablate_genomic: bool = False
    ablate_imaging: bool = False
 
    dropout: float = 0.1
    output_dim: int = 1
 
 
class TemporalBranch(nn.Module):
    """Encoder for temporal sequences."""
 
    def __init__(
        self,
        num_features: int,
        lstm_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.lstm = nn.LSTM(
            num_features,
            lstm_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
 
        self.attention = nn.MultiheadAttention(
            lstm_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
 
        self.fc = nn.Sequential(
            nn.Linear(lstm_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_features)
            seq_lengths: (batch,)
 
        Returns:
            (batch, output_dim)
        """
        # LSTM encoding
        packed = pack_padded_sequence(
            x,
            seq_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        lstm_out, (h_n, c_n) = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
 
        # Attention pooling
        attn_out, _ = self.attention(
            lstm_out,
            lstm_out,
            lstm_out,
        )
 
        # Get final state
        batch_idx = torch.arange(len(seq_lengths), device=x.device)
        final_rep = attn_out[batch_idx, seq_lengths - 1]  # (batch, lstm_dim)
 
        # MLP
        output = self.fc(final_rep)
 
        return output
 
 
class ClinicalBranch(nn.Module):
    """Encoder for static clinical features."""
 
    def __init__(
        self,
        num_features: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
 
        Returns:
            (batch, output_dim)
        """
        return self.fc(x)
 
 
class GenomicBranch(nn.Module):
    """Encoder for genomic features."""
 
    def __init__(
        self,
        num_features: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.embedding = nn.Sequential(
            nn.Linear(num_features, num_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features * 2, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
 
        Returns:
            (batch, output_dim)
        """
        return self.embedding(x)
 
 
class ImagingBranch(nn.Module):
    """Encoder for imaging features (radiomics, etc)."""
 
    def __init__(
        self,
        num_features: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_features, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features)
 
        Returns:
            (batch, output_dim)
        """
        return self.fc(x)
 
 
class AttentionFusion(nn.Module):
    """
    Attention-based fusion of multiple modalities.
 
    Learns weights for each modality based on context.
    """
 
    def __init__(
        self,
        num_modalities: int,
        feature_dim: int,
        output_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
 
        self.num_modalities = num_modalities
 
        # Modality-specific queries
        self.modality_queries = nn.Parameter(
            torch.randn(num_modalities, feature_dim)
        )
        nn.init.xavier_uniform_(self.modality_queries)
 
        # Cross-modal attention
        self.attention = nn.MultiheadAttention(
            feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
 
        # Fusion MLP
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * num_modalities, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
 
    def forward(self, modality_embeddings: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            modality_embeddings: List of (batch, feature_dim) tensors
 
        Returns:
            fused: (batch, output_dim)
            weights: (num_modalities,) or (batch, num_modalities) attention weights
        """
        batch_size = modality_embeddings[0].size(0)
 
        # Stack modalities
        stacked = torch.stack(modality_embeddings, dim=1)  # (batch, num_mod, feature_dim)
 
        # Attention-based fusion
        # Use queries to weight modalities
        queries = self.modality_queries.unsqueeze(0).expand(batch_size, -1, -1)
        attn_out, attn_weights = self.attention(queries, stacked, stacked)
 
        # Global weights per modality
        modality_weights = attn_weights.mean(dim=(0, 2))  # Average over batch and query positions
        modality_weights = F.softmax(modality_weights, dim=0)
 
        # Concatenate all modalities
        concat = torch.cat(modality_embeddings, dim=1)
 
        # Final fusion
        fused = self.fc(concat)
 
        return fused, modality_weights
 
 
class MultimodalFusionNet(BaseTemporalModel):
    """
    Multimodal fusion network combining multiple data modalities.
 
    Supports temporal, clinical, genomic, and imaging features
    with attention-based fusion and ablation studies.
    """
 
    def __init__(self, config: MultimodalFusionConfig):
        super().__init__(config, 1, config.output_dim)
        self.config = config
 
        # Create active modalities
        self.active_modalities: List[str] = []
 
        # Temporal branch
        if config.num_temporal_features > 0 and not config.ablate_temporal:
            self.temporal_branch = TemporalBranch(
                config.num_temporal_features,
                config.temporal_lstm_dim,
                config.fusion_dim,
                dropout=config.dropout,
            )
            self.active_modalities.append("temporal")
        else:
            self.temporal_branch = None
 
        # Clinical branch
        if config.num_clinical_features > 0 and not config.ablate_clinical:
            self.clinical_branch = ClinicalBranch(
                config.num_clinical_features,
                config.fusion_dim,
                dropout=config.dropout,
            )
            self.active_modalities.append("clinical")
        else:
            self.clinical_branch = None
 
        # Genomic branch
        if config.num_genomic_features > 0 and not config.ablate_genomic:
            self.genomic_branch = GenomicBranch(
                config.num_genomic_features,
                config.fusion_dim,
                dropout=config.dropout,
            )
            self.active_modalities.append("genomic")
        else:
            self.genomic_branch = None
 
        # Imaging branch
        if config.num_imaging_features > 0 and not config.ablate_imaging:
            self.imaging_branch = ImagingBranch(
                config.num_imaging_features,
                config.fusion_dim,
                dropout=config.dropout,
            )
            self.active_modalities.append("imaging")
        else:
            self.imaging_branch = None
 
        # Fusion
        num_active = len(self.active_modalities)
        if num_active > 0:
            self.fusion = AttentionFusion(
                num_active,
                config.fusion_dim,
                config.fusion_dim,
                num_heads=config.num_fusion_heads,
                dropout=config.dropout,
            )
 
            # Prediction head
            self.pred_head = nn.Sequential(
                nn.Linear(config.fusion_dim, config.fusion_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_dim, config.output_dim),
            )
        else:
            self.fusion = None
            self.pred_head = None
 
        logger.info(f"Multimodal network with active modalities: {self.active_modalities}")
 
    def forward(
        self,
        x: torch.Tensor,
        seq_lengths: torch.Tensor,
        static: Optional[torch.Tensor] = None,
        genomic: Optional[torch.Tensor] = None,
        imaging: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, num_temporal_features) temporal sequences
            seq_lengths: (batch,) sequence lengths
            static: (batch, num_clinical_features) static features
            genomic: (batch, num_genomic_features) optional
            imaging: (batch, num_imaging_features) optional
 
        Returns:
            predictions: (batch, output_dim)
        """
        embeddings: List[torch.Tensor] = []
 
        # Temporal modality
        if self.temporal_branch is not None:
            temporal_emb = self.temporal_branch(x, seq_lengths)
            embeddings.append(temporal_emb)
 
        # Clinical modality
        if self.clinical_branch is not None and static is not None:
            clinical_emb = self.clinical_branch(static)
            embeddings.append(clinical_emb)
 
        # Genomic modality
        if self.genomic_branch is not None and genomic is not None:
            genomic_emb = self.genomic_branch(genomic)
            embeddings.append(genomic_emb)
 
        # Imaging modality
        if self.imaging_branch is not None and imaging is not None:
            imaging_emb = self.imaging_branch(imaging)
            embeddings.append(imaging_emb)
 
        # Fusion
        if self.fusion is not None and len(embeddings) > 0:
            fused, weights = self.fusion(embeddings)
            self.modality_weights = weights
 
            # Prediction
            predictions = self.pred_head(fused)
        else:
            predictions = embeddings[0] if len(embeddings) > 0 else torch.zeros(1, self.config.output_dim)
 
        return predictions
 
    def compute_loss(
        self,
        y_pred: torch.Tensor,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute loss (task-specific, implement in subclass)."""
        y_true = batch["y"]
        # Match shapes: y_pred may be (batch, 1) while y_true is (batch,)
        if y_pred.dim() == 2 and y_true.dim() == 1:
            y_pred = y_pred.squeeze(-1)
        return F.mse_loss(y_pred, y_true)
 
    def get_modality_weights(self) -> Optional[torch.Tensor]:
        """Return learned modality weights from last forward pass."""
        if hasattr(self, "modality_weights"):
            return self.modality_weights
        return None
 
    def ablate_modality(self, modality: str) -> None:
        """Disable a modality for analysis."""
        if modality == "temporal":
            self.temporal_branch = None
        elif modality == "clinical":
            self.clinical_branch = None
        elif modality == "genomic":
            self.genomic_branch = None
        elif modality == "imaging":
            self.imaging_branch = None
        logger.info(f"Ablated modality: {modality}")
 
    def enable_modality(self, modality: str) -> None:
        """Re-enable a modality."""
        if modality == "temporal" and self.config.num_temporal_features > 0:
            self.temporal_branch = TemporalBranch(
                self.config.num_temporal_features,
                self.config.temporal_lstm_dim,
                self.config.fusion_dim,
                dropout=self.config.dropout,
            )
        elif modality == "clinical" and self.config.num_clinical_features > 0:
            self.clinical_branch = ClinicalBranch(
                self.config.num_clinical_features,
                self.config.fusion_dim,
                dropout=self.config.dropout,
            )
        logger.info(f"Enabled modality: {modality}")
 