"""
RAD-X Configuration Module
Handles project paths, dataset configuration, model parameters, and default thresholds.
Supports path overriding via environment variables for flexible deployment.
"""

import os
from pathlib import Path

# Project root resolution
# Defaults to the root repository folder (parent of ml/ directory)
PROJECT_ROOT = Path(
    os.getenv("RADX_ROOT", Path(__file__).resolve().parent.parent)
).resolve()

# Data Paths
DATA_DIR = Path(os.getenv("RADX_DATA_DIR", PROJECT_ROOT / "data")).resolve()

# Fallback to PROJECT_ROOT if data/ is not yet populated
def get_csv_path() -> Path:
    env_path = os.getenv("RADX_CSV_PATH")
    if env_path:
        return Path(env_path).resolve()
    if (DATA_DIR / "Chest X Ray.csv").exists():
        return DATA_DIR / "Chest X Ray.csv"
    if (PROJECT_ROOT / "Chest X Ray.csv").exists():
        return PROJECT_ROOT / "Chest X Ray.csv"
    return DATA_DIR / "Chest X Ray.csv"

def get_image_dir() -> Path:
    env_path = os.getenv("RADX_IMAGE_DIR")
    if env_path:
        return Path(env_path).resolve()
    if (DATA_DIR / "images").is_dir():
        return DATA_DIR / "images"
    if (PROJECT_ROOT / "images").is_dir():
        return PROJECT_ROOT / "images"
    return DATA_DIR / "images"

CSV_PATH = get_csv_path()
IMAGE_DIR = get_image_dir()

# Model Checkpoint Paths
MODELS_DIR = Path(os.getenv("RADX_MODELS_DIR", PROJECT_ROOT / "models")).resolve()
BEST_MODEL_NAME = "chest_xray_finetuned_best.pth"

def get_best_model_path() -> Path:
    env_path = os.getenv("RADX_MODEL_PATH")
    if env_path:
        return Path(env_path).resolve()
    if (MODELS_DIR / BEST_MODEL_NAME).exists():
        return MODELS_DIR / BEST_MODEL_NAME
    if (PROJECT_ROOT / BEST_MODEL_NAME).exists():
        return PROJECT_ROOT / BEST_MODEL_NAME
    return MODELS_DIR / BEST_MODEL_NAME

BEST_MODEL_PATH = get_best_model_path()

# Outputs Directory
OUTPUTS_DIR = Path(os.getenv("RADX_OUTPUTS_DIR", PROJECT_ROOT / "outputs")).resolve()
METRICS_DIR = OUTPUTS_DIR / "metrics"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
TRAINING_HISTORY_DIR = OUTPUTS_DIR / "training_history"

# Ensure output subdirectories exist
for d in (METRICS_DIR, GRAPHS_DIR, TRAINING_HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Preprocessing & Normalization constants (ImageNet standard)
IMAGE_SIZE = (224, 224)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Target disease classes (alphabetical order as fitted by MultiLabelBinarizer)
DISEASE_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"
]

# Validation thresholds computed from precision-recall curve F1 maximization (evaluate.py)
VALIDATION_THRESHOLDS = {
    "Atelectasis": 0.5752,
    "Cardiomegaly": 0.6858,
    "Consolidation": 0.6810,
    "Edema": 0.5286,
    "Effusion": 0.7964,
    "Emphysema": 0.3839,
    "Fibrosis": 0.6182,
    "Hernia": 0.1571,
    "Infiltration": 0.5793,
    "Mass": 0.5274,
    "No Finding": 0.2685,
    "Nodule": 0.4892,
    "Pleural_Thickening": 0.5444,
    "Pneumonia": 0.1885,
    "Pneumothorax": 0.6157
}

# Empirical test metrics from finetuned evaluation (finetuned_test_metrics.csv)
TEST_METRICS = {
    "Atelectasis": {"auc": 0.7435, "f1": 0.2659, "precision": 0.1877, "recall": 0.4562},
    "Cardiomegaly": {"auc": 0.8548, "f1": 0.2400, "precision": 0.3750, "recall": 0.1765},
    "Consolidation": {"auc": 0.7493, "f1": 0.0734, "precision": 0.1081, "recall": 0.0556},
    "Edema": {"auc": 0.8414, "f1": 0.0000, "precision": 0.0000, "recall": 0.0000},
    "Effusion": {"auc": 0.8096, "f1": 0.3957, "precision": 0.3834, "recall": 0.4088},
    "Emphysema": {"auc": 0.8565, "f1": 0.2883, "precision": 0.1963, "recall": 0.5424},
    "Fibrosis": {"auc": 0.7377, "f1": 0.1087, "precision": 0.1000, "recall": 0.1190},
    "Hernia": {"auc": 0.6420, "f1": 0.0000, "precision": 0.0000, "recall": 0.0000},
    "Infiltration": {"auc": 0.6953, "f1": 0.3212, "precision": 0.2736, "recall": 0.3889},
    "Mass": {"auc": 0.6999, "f1": 0.0787, "precision": 0.0737, "recall": 0.0843},
    "No Finding": {"auc": 0.7016, "f1": 0.7478, "precision": 0.6501, "recall": 0.8802},
    "Nodule": {"auc": 0.6273, "f1": 0.1606, "precision": 0.1399, "recall": 0.1887},
    "Pleural_Thickening": {"auc": 0.6916, "f1": 0.1451, "precision": 0.1284, "recall": 0.1667},
    "Pneumonia": {"auc": 0.7038, "f1": 0.0469, "precision": 0.0316, "recall": 0.0909},
    "Pneumothorax": {"auc": 0.7762, "f1": 0.2555, "precision": 0.2229, "recall": 0.2991}
}
