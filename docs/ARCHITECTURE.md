# RAD-X System Architecture & ML Pipeline

## Overview
RAD-X is a deep learning multi-label classification system designed to detect 14 thoracic disease categories plus a "No Finding" baseline from frontal chest radiographs.

```
                           +------------------------+
                           |  Chest Radiograph (.png|jpg)
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           |     Preprocessing      |
                           |   224x224 RGB Tensor   |
                           |  ImageNet Normalization|
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           |    DenseNet121 Backbone|
                           |   Feature Extractor    |
                           +-----------+------------+
                                       |
                                       v
                           +------------------------+
                           | 15-Class Linear Head   |
                           | Logits -> Sigmoid      |
                           +-----------+------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +------------------------+              +------------------------+
      | Continuous Probabilities|              | Calibrated Thresholds  |
      |   P(Disease_i) in [0,1] |              | tau_i (Validation F1)  |
      +------------+-----------+              +------------+-----------+
                   |                                       |
                   +-------------------+-------------------+
                                       |
                                       v
                           +------------------------+
                           | Binary Classifications |
                           | P(Disease_i) >= tau_i  |
                           +------------------------+
```

---

## 1. Model Architecture
- **Backbone**: DenseNet121 initialized with torchvision ImageNet weights.
- **Classifier Head**: Replaced 1000-class ImageNet head with `nn.Linear(1024, 15)`.
- **Output Activation**: Independent Sigmoid units for multi-label classification:
  $$\hat{y}_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$$

## 2. Loss Function & Class Imbalance Handling
Because thoracic diseases are rare (severe positive/negative skew), standard Binary Cross-Entropy results in trivial majority-class prediction.
RAD-X employs **Weighted Binary Cross Entropy (`BCEWithLogitsLoss`)**:
$$\mathcal{L} = -\sum_{i=1}^{15} \left[ w_i \cdot y_i \log(\sigma(z_i)) + (1 - y_i) \log(1 - \sigma(z_i)) \right]$$
where the positive class weight is computed strictly on the training partition:
$$w_i = \min\left( \frac{N - N_i^+}{N_i^+}, 10.0 \right)$$
Capping at `10.0` prevents unstable gradient spikes for ultra-rare pathologies (e.g. Hernia).

## 3. Training & Fine-Tuning Strategy
1. **Baseline Phase (`ml/train.py`)**:
   - Optimizer: Adam ($lr = 10^{-4}$).
   - Learning Rate Scheduler: `ReduceLROnPlateau` ($factor = 0.5$, $patience = 1$).
   - Early Stopping: $patience = 3$.
2. **Two-Stage Fine-Tuning (`ml/finetune.py`)**:
   - **Stage 1**: Backbone frozen (`requires_grad = False`), training classifier head only ($lr = 10^{-4}$) for 2 epochs.
   - **Stage 2**: Full end-to-end unfreezing with differential learning rates:
     - Feature extractor backbone: $10^{-5}$
     - Linear classifier head: $10^{-5}$
     - Scheduler: `ReduceLROnPlateau` on validation loss.

## 4. Decision Threshold Calibration
Rather than arbitrarily fixing a global 0.5 threshold across all pathologies:
- Decision boundaries $\tau_i$ are tuned exclusively on the **Validation set** (2,179 images) using precision-recall curves to maximize F1-score:
  $$\tau_i = \arg\max_\tau F_1(\tau; \text{Val})$$
- The tuned thresholds are frozen and tested once on the untouched **Holdout Test set** (2,121 images).
