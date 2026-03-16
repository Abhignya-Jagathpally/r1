"""
Example usage of MM temporal and fusion models.

Demonstrates:
  1. Data loading with SurvivalDataset
  2. Training TFT for multi-horizon forecasting
  3. DeepHit for competing risks
  4. Dynamic survival with landmarks
  5. Multimodal fusion
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets import SurvivalDataset, MultimodalDataset, create_survival_collate_fn
from temporal_fusion_transformer import TemporalFusionTransformer, TFTConfig
from deephit import DeepHit, DeepHitConfig
from dynamic_survival import DynamicSurvivalModel, DynamicSurvivalConfig
from multimodal_fusion import MultimodalFusionNet, MultimodalFusionConfig


def generate_sample_data(n_samples: int = 100):
    """Generate synthetic MM patient data."""
    seq_len = 24  # months of observations
    n_features = 10

    # Longitudinal sequences
    sequences = [
        np.random.randn(np.random.randint(5, seq_len), n_features)
        for _ in range(n_samples)
    ]

    # Observation times
    times = [
        np.linspace(0, s.shape[0], s.shape[0])
        for s in sequences
    ]

    # Event times and indicators
    event_times = np.random.exponential(12, n_samples)
    events = np.random.binomial(1, 0.6, n_samples)
    event_types = np.random.randint(0, 3, n_samples)

    # Static clinical features
    clinical_features = np.random.randn(n_samples, 8)

    # Optional: genomic and imaging features
    genomic_features = np.random.randn(n_samples, 100)
    imaging_features = np.random.randn(n_samples, 64)

    return {
        "sequences": sequences,
        "times": times,
        "event_times": event_times,
        "events": events,
        "event_types": event_types,
        "clinical": clinical_features,
        "genomic": genomic_features,
        "imaging": imaging_features,
    }


def example_tft():
    """Example: Temporal Fusion Transformer."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Temporal Fusion Transformer (TFT)")
    print("=" * 60)

    # Generate data
    data = generate_sample_data(n_samples=100)

    # Create dataset
    dataset = SurvivalDataset(
        sequences=data["sequences"],
        times=data["times"],
        event_times=data["event_times"],
        events=data["events"],
        event_types=data["event_types"],
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=create_survival_collate_fn(),
    )

    # Configure TFT
    config = TFTConfig(
        num_features=10,
        num_static_features=8,
        embedding_dim=64,
        lstm_hidden_dim=128,
        num_attention_heads=4,
        num_transformer_layers=2,
        prediction_horizons=(3, 6, 12),
        num_epochs=2,  # Quick demo
        learning_rate=1e-3,
        early_stopping_patience=10,
        use_amp=False,  # Disable for demo
        use_mlflow=False,
    )

    # Create and train model
    model = TemporalFusionTransformer(config)
    print("Model created. Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Single forward pass demo
    batch = next(iter(loader))
    with torch.no_grad():
        predictions = model(
            x=batch["x"],
            times=batch["times"],
            seq_lengths=batch["seq_lengths"],
            static=batch.get("static"),
        )
    print(f"Input shape: {batch['x'].shape}")
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions (3, 6, 12 month horizons): {predictions[0].numpy()}")


def example_deephit():
    """Example: DeepHit for competing risks."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: DeepHit (Competing Risks)")
    print("=" * 60)

    # Generate data
    data = generate_sample_data(n_samples=100)

    # Create dataset
    dataset = SurvivalDataset(
        sequences=data["sequences"],
        times=data["times"],
        event_times=data["event_times"],
        events=data["events"],
        event_types=data["event_types"],
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=create_survival_collate_fn(),
    )

    # Configure DeepHit
    config = DeepHitConfig(
        num_features=10,
        num_causes=3,  # progression, death, relapse
        num_time_steps=60,
        lstm_hidden_dim=128,
        alpha=0.5,
        num_epochs=2,
        learning_rate=1e-3,
        use_amp=False,
        use_mlflow=False,
    )

    # Create model
    model = DeepHit(config)
    print("Model created. Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        cause_probs = model(
            x=batch["x"],
            seq_lengths=batch["seq_lengths"],
        )

    print(f"Input shape: {batch['x'].shape}")
    print(f"Output shape (batch, causes, time_steps): {cause_probs.shape}")

    # Get survival curves
    survival = model.get_survival_curves(batch["x"], batch["seq_lengths"])
    print(f"Survival curves shape: {survival.shape}")

    # Risk scores at 6 months
    risk_6m = model.get_risk_scores(batch["x"], batch["seq_lengths"], time_horizon=6)
    print(f"6-month risk scores shape: {risk_6m.shape}")
    print(f"Sample risk (progression, death, relapse): {risk_6m[0].numpy()}")


def example_dynamic_survival():
    """Example: Dynamic Survival Model."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Dynamic Survival (Landmarks)")
    print("=" * 60)

    # Generate data
    data = generate_sample_data(n_samples=100)

    # Create dataset
    dataset = SurvivalDataset(
        sequences=data["sequences"],
        times=data["times"],
        event_times=data["event_times"],
        events=data["events"],
        event_types=data["event_types"],
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=create_survival_collate_fn(),
    )

    # Configure model
    config = DynamicSurvivalConfig(
        num_features=10,
        lstm_hidden_dim=128,
        landmark_times=(0, 3, 6, 12),
        prediction_horizon=12,
        use_superposition=True,
        num_epochs=2,
        learning_rate=1e-3,
        use_amp=False,
        use_mlflow=False,
    )

    # Create model
    model = DynamicSurvivalModel(config)
    print("Model created. Number of parameters:", sum(p.numel() for p in model.parameters()))

    # Forward pass
    batch = next(iter(loader))
    with torch.no_grad():
        conditional_survival = model(
            x=batch["x"],
            times=batch["times"],
            seq_lengths=batch["seq_lengths"],
        )

    print(f"Input shape: {batch['x'].shape}")
    print(f"Conditional survival shape: {conditional_survival.shape}")
    print(f"Sample: P(T > 12 | T > landmark): {conditional_survival[0].numpy()}")


def example_multimodal():
    """Example: Multimodal Fusion."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multimodal Fusion Network")
    print("=" * 60)

    # Generate data
    data = generate_sample_data(n_samples=100)

    # Create multimodal dataset
    dataset = MultimodalDataset(
        sequences=data["sequences"],
        times=data["times"],
        event_times=data["event_times"],
        events=data["events"],
        event_types=data["event_types"],
        static_features=data["clinical"],
        genomic_features=data["genomic"],
        imaging_features=data["imaging"],
    )

    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=create_survival_collate_fn(),
    )

    # Configure model
    config = MultimodalFusionConfig(
        num_temporal_features=10,
        num_clinical_features=8,
        num_genomic_features=100,
        num_imaging_features=64,
        temporal_lstm_dim=128,
        fusion_dim=256,
        output_dim=1,
        num_epochs=2,
        learning_rate=1e-3,
        use_amp=False,
        use_mlflow=False,
    )

    # Create model
    model = MultimodalFusionNet(config)
    print("Model created. Number of parameters:", sum(p.numel() for p in model.parameters()))
    print(f"Active modalities: {model.active_modalities}")

    # Forward pass with all modalities
    batch = next(iter(loader))
    with torch.no_grad():
        predictions = model(
            x=batch["x"],
            seq_lengths=batch["seq_lengths"],
            static=batch.get("static"),
        )

    print(f"Temporal shape: {batch['x'].shape}")
    print(f"Clinical shape: {batch.get('static').shape if 'static' in batch else 'None'}")
    print(f"Output shape: {predictions.shape}")
    print(f"Output value: {predictions[0].item():.4f}")

    # Get modality weights
    weights = model.get_modality_weights()
    print(f"Modality weights: {weights}")

    # Ablation: disable genomic modality
    print("\nAblation study: removing genomic modality...")
    model.ablate_modality("genomic")
    with torch.no_grad():
        predictions_no_genomic = model(
            x=batch["x"],
            seq_lengths=batch["seq_lengths"],
            static=batch.get("static"),
        )
    print(f"Output without genomic: {predictions_no_genomic[0].item():.4f}")


if __name__ == "__main__":
    print("Multiple Myeloma Temporal & Fusion Models - Examples")
    print("====================================================\n")

    # Run examples
    example_tft()
    example_deephit()
    example_dynamic_survival()
    example_multimodal()

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
