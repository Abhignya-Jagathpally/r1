# Multiple Myeloma Temporal & Fusion Models

Advanced PyTorch implementations for predicting MM disease progression, competing risks (progression/death/relapse), and multi-horizon forecasting with interpretability.

## Features

### Models

1. **Temporal Fusion Transformer (TFT)**
   - Multi-horizon predictions (3, 6, 12 months)
   - Attention-based interpretability for feature importance
   - Handles variable-length irregular time series
   - Gated feature selection networks

2. **DeepHit: Competing Risks**
   - Multiple event types: progression, death, relapse
   - Discrete-time survival framework
   - Ranking loss + NLL for proper calibration
   - Cause-specific hazard curves

3. **Dynamic Survival Model**
   - Landmark-based conditional survival: P(T > t+Δ | T>t, H(t))
   - Superposition over multiple landmarks
   - Time-varying covariate adjustment
   - RNN with attention on recent observations

4. **Multimodal Fusion Network**
   - Late fusion: temporal + clinical + genomic + imaging
   - Attention-based modality weighting
   - Ablation support for feature importance
   - Flexible modality combinations

### Training Infrastructure

- **Mixed Precision Training**: AMP support for faster computation
- **Early Stopping**: Configurable patience and validation metrics
- **Learning Rate Scheduling**: Cosine, plateau reduction, linear warmup
- **Gradient Clipping**: Prevent training instability
- **Gradient Accumulation**: Support for small effective batch sizes
- **Checkpoint Management**: Save/load best models automatically
- **MLflow Integration**: Experiment tracking and hyperparameter logging

### Datasets

- **LongitudinalDataset**: Variable-length sequences with automatic normalization
- **SurvivalDataset**: Event times, censoring, competing risks
- **MultimodalDataset**: Combines temporal, clinical, genomic, imaging modalities
- **Custom Collate Functions**: Sequence padding, packing, masking for RNNs

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Create Dataset

```python
import numpy as np
from researcher3_temporal import (
    SurvivalDataset,
    create_survival_collate_fn,
)
from torch.utils.data import DataLoader

# Generate sample data
n_samples = 100
seq_len = 24
n_features = 10

sequences = [np.random.randn(np.random.randint(5, seq_len), n_features)
             for _ in range(n_samples)]
times = [np.linspace(0, s.shape[0], s.shape[0])
         for s in sequences]
event_times = np.random.exponential(12, n_samples)  # months
events = np.random.binomial(1, 0.6, n_samples)      # 60% event rate
event_types = np.random.randint(0, 3, n_samples)    # 3 event types

# Create dataset
dataset = SurvivalDataset(
    sequences=sequences,
    times=times,
    event_times=event_times,
    events=events,
    event_types=event_types,
)

# Create loader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=create_survival_collate_fn(pack_sequences=True),
)
```

### 2. Train Temporal Fusion Transformer

```python
from researcher3_temporal import TemporalFusionTransformer, TFTConfig

# Configure model
config = TFTConfig(
    num_features=10,
    num_static_features=5,
    embedding_dim=64,
    lstm_hidden_dim=128,
    num_attention_heads=4,
    prediction_horizons=(3, 6, 12),
    num_epochs=50,
    learning_rate=1e-3,
    use_mlflow=True,
    experiment_name="mm_tft",
)

# Create model
model = TemporalFusionTransformer(config)

# Train
history = model.fit(
    train_loader,
    val_loader,
    checkpoint_dir="./checkpoints",
)
```

### 3. Train DeepHit for Competing Risks

```python
from researcher3_temporal import DeepHit, DeepHitConfig

config = DeepHitConfig(
    num_features=10,
    num_causes=3,
    num_time_steps=60,
    alpha=0.5,  # Balance ranking loss and NLL
    num_epochs=100,
)

model = DeepHit(config)
history = model.fit(train_loader, val_loader)

# Get survival curves
survival_curves = model.get_survival_curves(x, seq_lengths)

# Get risk scores at 6 months
risk_scores = model.get_risk_scores(x, seq_lengths, time_horizon=6)
```

### 4. Dynamic Survival with Landmarks

```python
from researcher3_temporal import DynamicSurvivalModel, DynamicSurvivalConfig

config = DynamicSurvivalConfig(
    num_features=10,
    landmark_times=(0, 3, 6, 12),
    prediction_horizon=12,
    use_superposition=True,
)

model = DynamicSurvivalModel(config)
history = model.fit(train_loader, val_loader)

# Conditional survival: P(T > t+12 | T > t, H(t))
conditional_survival = model.get_conditional_survival(
    x, times, seq_lengths, landmark_times,
    prediction_horizons=(3, 6, 12),
)
```

### 5. Multimodal Fusion

```python
from researcher3_temporal import MultimodalFusionNet, MultimodalFusionConfig

config = MultimodalFusionConfig(
    num_temporal_features=10,
    num_clinical_features=8,
    num_genomic_features=100,
    num_imaging_features=64,
    fusion_dim=256,
    output_dim=1,
)

model = MultimodalFusionNet(config)

# Forward pass with all modalities
predictions = model(
    x=temporal_data,              # (batch, seq_len, 10)
    seq_lengths=seq_lengths,      # (batch,)
    static=clinical_data,         # (batch, 8)
    genomic=genomic_data,         # (batch, 100)
    imaging=imaging_data,         # (batch, 64)
)

# Get learned modality importance
weights = model.get_modality_weights()

# Ablation study
model.ablate_modality("genomic")
predictions_without_genomic = model(...)
```

## Architecture Overview

### Temporal Fusion Transformer

```
Input (batch, seq_len, features)
         |
         v
Embedding + Time Encoding
         |
         v
Variable Selection Network
         |
         v
LSTM Encoder
         |
         v
Transformer (self-attention)
         |
         v
LSTM Decoder
         |
         v
Prediction Head (multi-horizon)
         |
         v
Output (batch, num_horizons)
```

### DeepHit

```
Input (batch, seq_len, features)
         |
         v
LSTM Encoder
         |
         v
Shared Representation
         |
    +---------+---------+
    |         |         |
    v         v         v
  Cause1    Cause2    Cause3
  Network   Network   Network
    |         |         |
    +---------+---------+
         |
         v
Survival Probabilities
(batch, num_causes, num_time_steps)
```

### Dynamic Survival

```
Input (sequence, times)
         |
         v
Landmark Window Selection
(for each landmark t)
         |
         v
LSTM Encode H(t)
         |
         v
Attention Pooling
         |
         v
Conditional Risk P(T > t+Δ | T>t)
         |
         v
Superposition (average over landmarks)
         |
         v
Output: Conditional Survival Prob
```

### Multimodal Fusion

```
Temporal        Clinical       Genomic       Imaging
   Data           Data          Data          Data
    |              |             |             |
    v              v             v             v
  LSTM      MLP Branch    Embedding      Radiomics
 Branch     (static)      Network        Encoder
    |              |             |             |
    +------+-------+------+------+
           |
           v
    Attention Fusion
    (learn modality weights)
           |
           v
    Prediction Head
           |
           v
         Output
```

## Loss Functions

### TFT
- **MSE Loss**: Standard regression loss for point predictions
- **Huber Loss**: Robust to outliers
- **Quantile Loss**: Prediction intervals for uncertainty quantification

### DeepHit
- **Ranking Loss**: Enforce correct ordering of survival times
- **NLL Loss**: Proper likelihood for calibration
- **Combined**: α * ranking_loss + (1-α) * nll_loss

### Dynamic Survival
- **BCE Loss**: Binary cross-entropy with event indicators
- **Cox Loss**: Partial likelihood for proper calibration
- **Weighted**: Downweight censored samples

## Configuration Guide

### TrainingConfig
- `learning_rate`: 1e-3 to 1e-4 (start lower for TFT)
- `num_epochs`: 50-100 for validation stability
- `early_stopping_patience`: 20-30 epochs
- `gradient_clip_norm`: 1.0 (standard value)
- `batch_size`: 32-64 (larger for stable training)
- `scheduler_type`: "cosine" (recommended) or "plateau"

### TFTConfig
- `embedding_dim`: 64-128 (higher for complex patterns)
- `lstm_hidden_dim`: 128-256
- `num_attention_heads`: 4-8
- `num_transformer_layers`: 2-4
- `dropout`: 0.1-0.3

### DeepHitConfig
- `num_time_steps`: 60 (bins for discretized time)
- `alpha`: 0.5 (balance ranking vs NLL)
- `num_causes`: 3 (progression, death, relapse)

### MultimodalFusionConfig
- Enable modalities via `num_*_features > 0`
- Ablate with `ablate_*: True`
- `fusion_dim`: 256-512 (bottleneck dimension)

## Interpretability

### Attention Weights
All models expose attention mechanisms for interpretability:

```python
# TFT feature importance
model.eval()
with torch.no_grad():
    predictions = model(x, times, seq_lengths, static)
    attention_weights = model.attention_weights

# Visualize
import matplotlib.pyplot as plt
plt.heatmap(attention_weights[0].cpu().numpy())
plt.xlabel("Time step")
plt.ylabel("Feature")
plt.title("TFT Feature Attention")
```

### Modality Importance (Multimodal)
```python
weights = model.get_modality_weights()
print(f"Temporal: {weights[0]:.3f}")
print(f"Clinical: {weights[1]:.3f}")
print(f"Genomic: {weights[2]:.3f}")
```

## Performance Tips

1. **Normalize Input Features**: All datasets auto-normalize by default
2. **Handle Imbalance**: Use sample_weight in loss computation
3. **Sequence Padding**: Uses packed sequences for efficiency (≤10% overhead)
4. **Mixed Precision**: 2-3x speedup with minimal accuracy loss
5. **Learning Rate**: Use cosine annealing with warmup for stability

## Citation

If using these models, cite:

- TFT: Lim et al. (2021) - "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- DeepHit: Lee et al. (2018) - "DeepHit: A Deep Learning Approach for Survival Analysis with Competing Risks"
- Landmarks: van Houwelingen & Putter (2007) - "Dynamic Predicting Survival"

## License

Part of Multiple Myeloma Digital Twin Pipeline
