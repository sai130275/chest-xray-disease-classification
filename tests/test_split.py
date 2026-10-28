"""
Unit tests for DataLoader shapes, data types, and batch generation.
"""

import torch
from ml.dataset import get_dataloaders
from ml.config import IMAGE_SIZE

def test_dataloaders_batch_contracts():
    batch_size = 16
    train_loader, val_loader, test_loader, mlb = get_dataloaders(batch_size=batch_size)

    assert len(train_loader) > 0
    assert len(val_loader) > 0
    assert len(test_loader) > 0

    # Inspect one training batch
    images, labels = next(iter(train_loader))

    assert images.shape == (batch_size, 3, IMAGE_SIZE[0], IMAGE_SIZE[1]), (
        f"Expected image batch shape ({batch_size}, 3, 224, 224), got {images.shape}"
    )
    assert images.dtype == torch.float32, f"Expected float32 images, got {images.dtype}"
    assert labels.shape == (batch_size, 15), f"Expected labels shape ({batch_size}, 15), got {labels.shape}"
    assert labels.dtype == torch.float32, f"Expected float32 labels, got {labels.dtype}"
    assert ((labels == 0.0) | (labels == 1.0)).all(), "Binary multi-label values must be either 0.0 or 1.0"

    # Inspect one test batch
    test_images, test_labels = next(iter(test_loader))
    assert test_images.shape[1:] == (3, 224, 224)
    assert test_labels.shape[1] == 15

if __name__ == "__main__":
    test_dataloaders_batch_contracts()
    print("test_split.py: ALL TESTS PASSED.")
