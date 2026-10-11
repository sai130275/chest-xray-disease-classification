# RAD-X: Chest X-Ray Disease Classification

**RAD-X** is a deep learning project that uses chest X-ray images to classify 14 thoracic diseases and identify images labeled as **No Finding**. It uses transfer learning with DenseNet121 and PyTorch to perform multi-label classification.

## Features

- Multi-label classification of 14 thoracic diseases and No Finding
- DenseNet121 pretrained on ImageNet
- Image preprocessing and data augmentation
- Class-weighted Binary Cross-Entropy loss
- Patient-level training, validation, and test splits
- Learning-rate scheduling and early stopping
- Per-class AUC-ROC, F1-score, precision, and recall evaluation
- Training history and performance visualizations

## Tech Stack

- Python
- PyTorch
- Torchvision
- NumPy and Pandas
- Scikit-learn
- Matplotlib
- Pillow

## Dataset

The project uses a subset of the NIH Chest X-ray dataset containing **14,999 images from 3,923 patients**.

The dataset includes labels for:

Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, and No Finding.

The dataset and image archive are not included in this repository. Obtain the authorized dataset separately and place the images and `Chest X Ray.csv` in the expected project structure.

## Model Performance

The staged fine-tuning experiment produced the following held-out test results:

| Metric | Score |
|---|---:|
| Mean AUC-ROC | 0.7420 |
| Macro F1-score | 0.2085 |
| Macro Precision | 0.1914 |
| Macro Recall | 0.2572 |

Performance varies considerably across disease classes. These results represent an educational experiment, not a clinically validated diagnostic system.

## Project Structure

```text
RAD-X/
├── dataset.py
├── train.py
├── evaluate.py
├── finetune.py
├── check_data.py
├── test_dataset.py
├── test_split.py
├── extract_images.py
├── requirements.txt
├── Chest X Ray.csv
├── disease_metrics.csv
├── finetuned_test_metrics.csv
├── scheduler_test_metrics.csv
├── final_test_metrics.csv
├── training_history.png
├── finetune_history.png
├── final_test_auc.png
├── final_test_roc.png
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/sai130275/chest-xray-disease-classification.git
cd chest-xray-disease-classification
```

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset CSV and corresponding images are available in the locations expected by `dataset.py`.

## Usage

Train the baseline model:

```bash
python train.py
```

Run staged fine-tuning:

```bash
python finetune.py
```

Evaluate a trained model:

```bash
python evaluate.py
```

**Note:** `evaluate.py` must reference the checkpoint you intend to evaluate. The current evaluation script may need its checkpoint and output filenames updated before evaluating a different experiment.

## Methodology

1. Load image metadata and disease labels.
2. Encode labels into multi-hot vectors.
3. Split the dataset by patient to prevent patient overlap.
4. Resize and normalize images.
5. Train DenseNet121 using class-weighted binary cross-entropy.
6. Select model checkpoints using validation performance.
7. Select per-class classification thresholds using validation data.
8. Evaluate on the held-out test set.

## Limitations

- Severe class imbalance affects rare disease categories.
- Labels derived from radiology reports may contain noise.
- Performance is inconsistent across disease classes.
- Threshold selection may be unstable for rare labels.
- The model has not undergone clinical validation.

## Disclaimer

RAD-X is developed for educational and research purposes only. It must not be used for medical diagnosis, treatment decisions, or other clinical purposes.
