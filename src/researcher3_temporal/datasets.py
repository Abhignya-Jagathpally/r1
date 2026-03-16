"""
PyTorch Datasets for longitudinal MM data.

Handles:
  - Variable-length sequences with temporal information
  - Survival outcomes with censoring
  - Multi-modal data (clinical, genomic, imaging)
  - Custom collate functions for sequence padding/packing
"""

from typing import Optional, Tuple, Dict, Any, List, Callable
from dataclasses import dataclass
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

logger = logging.getLogger(__name__)


@dataclass
class SequenceSample:
    """Single longitudinal sample."""

    patient_id: str
    sequence: torch.Tensor  # (seq_len, features)
    times: torch.Tensor  # (seq_len,) - time in months
    label: torch.Tensor  # target (scalar or multi-outcome)
    event: Optional[torch.Tensor] = None  # event indicator (0=censored, 1=event)
    event_type: Optional[torch.Tensor] = None  # for competing risks
    static_features: Optional[torch.Tensor] = None  # (static_dim,)
    sample_weight: float = 1.0


class LongitudinalDataset(Dataset):
    """
    Variable-length longitudinal sequences for MM patients.

    Data format:
    - sequences: list of (seq_len, feature_dim) arrays
    - times: list of (seq_len,) time arrays in months
    - targets: (n_samples,) array

    Architecture:
    ┌──────────────────────────────────┐
    │  LongitudinalDataset             │
    ├──────────────────────────────────┤
    │ sequences: list[ndarray]         │ (variable length)
    │ times: list[ndarray]             │
    │ targets: ndarray                 │
    │                                  │
    │ __getitem__ -> SequenceSample    │
    └──────────────────────────────────┘
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        times: List[np.ndarray],
        targets: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        normalize: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            sequences: List of (seq_len, feature_dim) arrays
            times: List of (seq_len,) time arrays
            targets: (n_samples,) target values
            patient_ids: Optional patient identifiers
            normalize: Normalize sequences to zero mean, unit variance
        """
        assert len(sequences) == len(times) == len(targets)

        self.sequences = sequences
        self.times = times
        self.targets = targets
        self.patient_ids = patient_ids or [f"patient_{i}" for i in range(len(sequences))]

        if normalize:
            self._normalize_sequences()

        logger.info(
            f"Loaded dataset with {len(sequences)} samples, "
            f"feature_dim={sequences[0].shape[1]}"
        )

    def _normalize_sequences(self) -> None:
        """Normalize all sequences to zero mean, unit variance."""
        all_data = np.concatenate(self.sequences, axis=0)
        self.mean = np.mean(all_data, axis=0, keepdims=True)
        self.std = np.std(all_data, axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0  # Avoid division by zero

        self.sequences = [
            (seq - self.mean) / self.std
            for seq in self.sequences
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> SequenceSample:
        """Get single sample."""
        seq = torch.FloatTensor(self.sequences[idx])
        times = torch.FloatTensor(self.times[idx])
        label = torch.FloatTensor([self.targets[idx]])

        return SequenceSample(
            patient_id=self.patient_ids[idx],
            sequence=seq,
            times=times,
            label=label,
        )


class SurvivalDataset(LongitudinalDataset):
    """
    Longitudinal dataset with survival outcomes.

    Includes event times and censoring indicators.
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        times: List[np.ndarray],
        event_times: np.ndarray,
        events: np.ndarray,
        patient_ids: Optional[List[str]] = None,
        event_types: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        """
        Initialize survival dataset.

        Args:
            sequences: List of (seq_len, feature_dim) arrays
            times: List of (seq_len,) observation times
            event_times: (n_samples,) time to event or censoring
            events: (n_samples,) binary event indicators
            patient_ids: Optional identifiers
            event_types: (n_samples,) event type for competing risks
            normalize: Normalize sequences
        """
        super().__init__(sequences, times, event_times, patient_ids, normalize)

        assert len(events) == len(event_times)
        self.events = events
        self.event_types = event_types

        n_events = np.sum(events)
        n_censored = len(events) - n_events
        logger.info(
            f"Survival dataset: {n_events} events, {n_censored} censored "
            f"({100*n_events/len(events):.1f}% event rate)"
        )

    def __getitem__(self, idx: int) -> SequenceSample:
        """Get sample with event and censoring info."""
        sample = super().__getitem__(idx)

        sample.event = torch.FloatTensor([self.events[idx]])
        if self.event_types is not None:
            sample.event_type = torch.LongTensor([self.event_types[idx]])

        return sample


class MultimodalDataset(SurvivalDataset):
    """
    Multi-modal dataset: temporal + static clinical features + optional genomic.

    Architecture:
    ┌─────────────────────────────────┐
    │   MultimodalDataset             │
    ├─────────────────────────────────┤
    │ temporal: sequences + times      │
    │ static: clinical features       │
    │ genomic: (optional) embeddings  │
    │ imaging: (optional) features    │
    │                                 │
    │ Combines all modalities         │
    └─────────────────────────────────┘
    """

    def __init__(
        self,
        sequences: List[np.ndarray],
        times: List[np.ndarray],
        event_times: np.ndarray,
        events: np.ndarray,
        static_features: Optional[np.ndarray] = None,
        genomic_features: Optional[np.ndarray] = None,
        imaging_features: Optional[np.ndarray] = None,
        patient_ids: Optional[List[str]] = None,
        event_types: Optional[np.ndarray] = None,
        normalize: bool = True,
    ):
        """
        Initialize multimodal dataset.

        Args:
            sequences: List of (seq_len, feature_dim) temporal arrays
            times: List of (seq_len,) observation times
            event_times: (n_samples,) event times
            events: (n_samples,) event indicators
            static_features: (n_samples, static_dim) clinical features
            genomic_features: (n_samples, genomic_dim) optional
            imaging_features: (n_samples, imaging_dim) optional
            patient_ids: Optional identifiers
            event_types: (n_samples,) event type for competing risks
            normalize: Normalize features
        """
        super().__init__(
            sequences, times, event_times, events,
            patient_ids, event_types, normalize
        )

        self.static_features = static_features
        self.genomic_features = genomic_features
        self.imaging_features = imaging_features

        if self.static_features is not None:
            assert len(self.static_features) == len(sequences)

    def __getitem__(self, idx: int) -> SequenceSample:
        """Get multi-modal sample."""
        sample = super().__getitem__(idx)

        if self.static_features is not None:
            sample.static_features = torch.FloatTensor(self.static_features[idx])

        return sample

    def get_modality(self, idx: int, modality: str) -> Optional[np.ndarray]:
        """Get specific modality for sample."""
        if modality == "static":
            return self.static_features[idx] if self.static_features is not None else None
        elif modality == "genomic":
            return self.genomic_features[idx] if self.genomic_features is not None else None
        elif modality == "imaging":
            return self.imaging_features[idx] if self.imaging_features is not None else None
        return None


def pad_sequence_batch(
    sequences: List[torch.Tensor],
    batch_first: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length sequences in batch.

    Args:
        sequences: List of (seq_len, features)
        batch_first: If True, return (batch, seq_len, features)

    Returns:
        padded: (batch, max_len, features) or (max_len, batch, features)
        lengths: (batch,) actual sequence lengths
    """
    lengths = torch.LongTensor([s.size(0) for s in sequences])
    padded = pad_sequence(sequences, batch_first=batch_first, padding_value=0.0)

    return padded, lengths


def create_survival_collate_fn(
    pack_sequences: bool = False,
) -> Callable:
    """
    Create custom collate function for survival data.

    Args:
        pack_sequences: Use PackedSequence for RNNs

    Returns:
        Collate function
    """
    def collate_fn(batch: List[SequenceSample]) -> Dict[str, Any]:
        """Collate survival batch."""
        sequences = [s.sequence for s in batch]
        times = [s.times for s in batch]
        labels = torch.cat([s.label for s in batch], dim=0)
        events = torch.cat([s.event for s in batch], dim=0) if batch[0].event is not None else None
        event_types = None

        if batch[0].event_type is not None:
            event_types = torch.cat([s.event_type for s in batch], dim=0)

        # Pad sequences
        padded_seq, seq_lengths = pad_sequence_batch(sequences, batch_first=True)
        padded_times, _ = pad_sequence_batch(times, batch_first=True)

        # Pad times to match sequences
        padded_times = padded_times[:, :padded_seq.size(1)]

        result = {
            "x": padded_seq,
            "times": padded_times,
            "seq_lengths": seq_lengths,
            "y": labels,
        }

        if events is not None:
            result["events"] = events

        if event_types is not None:
            result["event_types"] = event_types

        # Static features
        if batch[0].static_features is not None:
            static = torch.stack([s.static_features for s in batch], dim=0)
            result["static"] = static

        # Packed sequences for RNNs
        if pack_sequences:
            sorted_lengths, sorted_idx = torch.sort(seq_lengths, descending=True)
            sorted_x = padded_seq[sorted_idx]
            packed_x = pack_padded_sequence(
                sorted_x,
                sorted_lengths.cpu(),
                batch_first=True,
                enforce_sorted=True,
            )
            result["x_packed"] = packed_x
            result["sorted_idx"] = sorted_idx

        return result

    return collate_fn
