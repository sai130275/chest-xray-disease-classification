"""
Unit tests for DenseNet121 model initialization, checkpoint loading,
and forward-pass inference sanity checks.
"""

import torch
import torch.nn as nn
from torchvision import models
from ml.config import BEST_MODEL_PATH, DISEASE_CLASSES

def test_model_initialization_and_forward_pass():
    num_classes = len(DISEASE_CLASSES)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    model.eval()

    # Create dummy batch of 2 images: shape (2, 3, 224, 224)
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = torch.sigmoid(logits)

    assert logits.shape == (2, num_classes), f"Expected logits shape (2, {num_classes}), got {logits.shape}"
    assert probs.shape == (2, num_classes), f"Expected probs shape (2, {num_classes}), got {probs.shape}"
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Sigmoid outputs must be bounded in [0, 1]"

def test_checkpoint_loading_and_weights():
    assert BEST_MODEL_PATH.exists(), f"Best checkpoint not found at: {BEST_MODEL_PATH}"

    num_classes = len(DISEASE_CLASSES)
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    state_dict = torch.load(BEST_MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    # Forward pass with real checkpoint weights
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        probs = torch.sigmoid(model(dummy_input))

    assert probs.shape == (1, num_classes)
    assert not torch.isnan(probs).any(), "Model produced NaN probabilities"

if __name__ == "__main__":
    test_model_initialization_and_forward_pass()
    test_checkpoint_loading_and_weights()
    print("test_model.py: ALL TESTS PASSED.")
