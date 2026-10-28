"""
RAD-X Inference Predictor
Encapsulates DenseNet121 model loading, preprocessing, and multi-label inference.
"""

import io
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from ml.config import (
    BEST_MODEL_PATH,
    DISEASE_CLASSES,
    VALIDATION_THRESHOLDS,
    TEST_METRICS,
    IMAGE_SIZE,
    NORM_MEAN,
    NORM_STD,
)


class RadXPredictor:
    """
    Singleton-style predictor for DenseNet121 chest X-ray multi-label classification.
    Loads model weights once into memory on startup.
    """
    _instance: Optional["RadXPredictor"] = None

    def __init__(self, checkpoint_path: Optional[Union[Path, str]] = None):
        self.device = torch.device(
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
        self.classes = DISEASE_CLASSES
        self.thresholds = VALIDATION_THRESHOLDS
        self.metrics = TEST_METRICS

        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
        ])

        target_path = Path(checkpoint_path) if checkpoint_path else BEST_MODEL_PATH
        if not target_path.exists():
            # Check models/ folder relative to current working dir
            alt_path = Path("models") / "chest_xray_finetuned_best.pth"
            if alt_path.exists():
                target_path = alt_path
            else:
                raise FileNotFoundError(f"Model checkpoint not found at: {target_path}")

        self.checkpoint_path = target_path
        self.model = self._load_model(target_path)
        self.is_ready = True

    def _load_model(self, checkpoint_path: Path) -> nn.Module:
        num_classes = len(self.classes)
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """
        Validates and converts image bytes into normalized PyTorch tensor (1, 3, 224, 224).
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Invalid or corrupted image file: {str(e)}")

        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs multi-label inference on provided image bytes.
        Returns probabilities, threshold comparisons, and research metadata.
        """
        input_tensor = self.preprocess_image(image_bytes)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        results: List[Dict[str, Any]] = []
        positive_findings: List[str] = []

        for idx, disease in enumerate(self.classes):
            prob = float(probs[idx])
            threshold = float(self.thresholds.get(disease, 0.5))
            is_positive = bool(prob >= threshold)
            disease_metric = self.metrics.get(disease, {})

            if is_positive and disease != "No Finding":
                positive_findings.append(disease)

            results.append({
                "disease": disease,
                "probability": round(prob, 4),
                "threshold": round(threshold, 4),
                "is_positive": is_positive,
                "test_auc": disease_metric.get("auc"),
                "test_f1": disease_metric.get("f1"),
                "test_precision": disease_metric.get("precision"),
                "test_recall": disease_metric.get("recall"),
            })

        # Sort results: positive findings first, then by highest probability
        sorted_results = sorted(
            results,
            key=lambda x: (not x["is_positive"], -x["probability"])
        )

        no_finding_prob = float(probs[self.classes.index("No Finding")])
        no_finding_threshold = float(self.thresholds.get("No Finding", 0.5))
        is_no_finding = bool(no_finding_prob >= no_finding_threshold)

        return {
            "predictions": sorted_results,
            "raw_predictions": results,
            "positive_findings": positive_findings,
            "is_no_finding": is_no_finding,
            "model_info": {
                "architecture": "DenseNet121",
                "checkpoint": self.checkpoint_path.name,
                "device": str(self.device),
                "classes_count": len(self.classes),
            },
            "disclaimer": (
                "EDUCATIONAL & RESEARCH PROTOTYPE ONLY: This model output is generated by an academic "
                "deep learning research prototype. It is NOT clinical diagnostic advice and must not "
                "be relied upon for patient diagnosis, staging, triage, or clinical decision-making. "
                "Always consult a board-certified radiologist or qualified physician."
            )
        }

# Global singleton accessor
def get_predictor(checkpoint_path: Optional[Union[Path, str]] = None) -> RadXPredictor:
    if RadXPredictor._instance is None:
        RadXPredictor._instance = RadXPredictor(checkpoint_path)
    return RadXPredictor._instance

if __name__ == "__main__":
    predictor = get_predictor()
    print(f"Predictor initialized successfully.")
    print(f"Device: {predictor.device}")
    print(f"Model Checkpoint: {predictor.checkpoint_path}")
    print(f"Number of classes: {len(predictor.classes)}")
