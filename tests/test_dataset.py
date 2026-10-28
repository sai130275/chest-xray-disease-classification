"""
Unit tests for RAD-X dataset loading, patient-level splits, and label encoding.
"""

from pathlib import Path
from ml.config import CSV_PATH, IMAGE_DIR, DISEASE_CLASSES
from ml.dataset import load_data, ChestXRayDataset

def test_data_paths_exist():
    assert Path(CSV_PATH).exists(), f"CSV file missing at {CSV_PATH}"
    assert Path(IMAGE_DIR).is_dir(), f"Images directory missing at {IMAGE_DIR}"

def test_load_data_splits_and_patient_leakage():
    train_df, val_df, test_df, mlb = load_data()

    total_images = len(train_df) + len(val_df) + len(test_df)
    assert total_images == 14999, f"Expected 14,999 total images, got {total_images}"
    assert len(train_df) == 10699, f"Expected 10,699 train images, got {len(train_df)}"
    assert len(val_df) == 2179, f"Expected 2,179 val images, got {len(val_df)}"
    assert len(test_df) == 2121, f"Expected 2,121 test images, got {len(test_df)}"

    train_patients = set(train_df["Patient ID"])
    val_patients = set(val_df["Patient ID"])
    test_patients = set(test_df["Patient ID"])

    assert len(train_patients) == 2746, f"Expected 2,746 train patients, got {len(train_patients)}"
    assert len(val_patients) == 588, f"Expected 588 val patients, got {len(val_patients)}"
    assert len(test_patients) == 589, f"Expected 589 test patients, got {len(test_patients)}"

    # Patient isolation assertions
    assert train_patients.isdisjoint(val_patients), "Patient leakage between train and val sets!"
    assert train_patients.isdisjoint(test_patients), "Patient leakage between train and test sets!"
    assert val_patients.isdisjoint(test_patients), "Patient leakage between val and test sets!"

    # Class labels assertions
    classes = list(mlb.classes_)
    assert len(classes) == 15, f"Expected 15 classes, got {len(classes)}"
    assert classes == DISEASE_CLASSES, "Class label ordering mismatch!"

if __name__ == "__main__":
    test_data_paths_exist()
    test_load_data_splits_and_patient_leakage()
    print("test_dataset.py: ALL TESTS PASSED.")
