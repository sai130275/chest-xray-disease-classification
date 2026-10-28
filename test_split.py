"""
Backwards-compatible test_split script.
Points to tests.test_split.
"""
from tests.test_split import test_dataloaders_batch_contracts

if __name__ == "__main__":
    test_dataloaders_batch_contracts()
    print("test_split.py: ALL CHECKS PASSED.")