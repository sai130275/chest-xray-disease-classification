"""
Backwards-compatible test_dataset script.
Points to tests.test_dataset.
"""
from tests.test_dataset import test_data_paths_exist, test_load_data_splits_and_patient_leakage

if __name__ == "__main__":
    test_data_paths_exist()
    test_load_data_splits_and_patient_leakage()
    print("test_dataset.py: ALL CHECKS PASSED.")