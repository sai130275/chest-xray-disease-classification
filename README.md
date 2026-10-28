# RAD-X: Deep Learning-Based Multi-Label Thoracic Disease Classification

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20Research-green.svg)](LICENSE)

> [!IMPORTANT]
> **RESEARCH AND EDUCATIONAL PROTOTYPE ONLY**
> RAD-X is developed strictly for machine learning research and educational demonstration. It does **not** constitute a medical device, clinical diagnostic aid, or therapeutic decision-making tool. It has not undergone clinical trials or FDA/CE regulatory clearance.

---

## 1. Project Overview

RAD-X provides an end-to-end multi-label classification pipeline and clinical research dashboard for detecting **14 thoracic disease categories** plus a **"No Finding"** baseline from frontal chest radiographs.

Built on **DenseNet121** with transfer learning from ImageNet, the model handles severe positive class imbalance via **weighted binary cross-entropy loss** and dynamic learning rate scheduling. Decision boundaries are tuned strictly on the validation set using precision-recall curves to optimize the $F_1$ score for each individual pathology.

---

## 2. Model Performance (Holdout Test Set)

Evaluated on the **untouched holdout test set** of **2,121 images** (589 unique patients) with **zero patient leakage**:

- **Mean AUC-ROC**: **`0.7383`**
- **Macro $F_1$**: **`0.2201`**
- **Macro Precision**: **`0.2045`**
- **Macro Recall**: **`0.2458`**

### Disease-by-Disease Test Results

| Pathology / Class | Test AUC-ROC | $F_1$ Score | Precision | Recall | Calibrated Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Emphysema** | **0.8565** | 0.2883 | 0.1963 | 0.5424 | 0.3839 |
| **Cardiomegaly** | **0.8548** | 0.2400 | 0.3750 | 0.1765 | 0.6858 |
| **Edema** | **0.8414** | 0.0000 | 0.0000 | 0.0000 | 0.5286 |
| **Effusion** | **0.8096** | 0.3957 | 0.3834 | 0.4088 | 0.7964 |
| **Pneumothorax** | **0.7762** | 0.2555 | 0.2229 | 0.2991 | 0.6157 |
| **Consolidation** | **0.7493** | 0.0734 | 0.1081 | 0.0556 | 0.6810 |
| **Atelectasis** | **0.7435** | 0.2659 | 0.1877 | 0.4562 | 0.5752 |
| **Fibrosis** | **0.7377** | 0.1087 | 0.1000 | 0.1190 | 0.6182 |
| **Pneumonia** | **0.7038** | 0.0469 | 0.0316 | 0.0909 | 0.1885 |
| **No Finding** | **0.7016** | 0.7478 | 0.6501 | 0.8802 | 0.2685 |
| **Mass** | **0.6999** | 0.0787 | 0.0737 | 0.0843 | 0.5274 |
| **Infiltration** | **0.6953** | 0.3212 | 0.2736 | 0.3889 | 0.5793 |
| **Pleural Thickening** | **0.6916** | 0.1451 | 0.1284 | 0.1667 | 0.5444 |
| **Hernia** | **0.6420** | 0.0000 | 0.0000 | 0.0000 | 0.1571 |
| **Nodule** | **0.6273** | 0.1606 | 0.1399 | 0.1887 | 0.4892 |

---

## 3. Repository Structure

```
RAD-X/
├── ml/                      # Core Machine Learning Pipeline
│   ├── config.py            # Central configuration & path management
│   ├── dataset.py           # Patient-level stratified split & DataLoaders
│   ├── train.py             # Baseline training with ReduceLROnPlateau
│   ├── finetune.py          # 2-stage transfer learning fine-tuning
│   ├── evaluate.py          # Threshold calibration & test set evaluation
│   ├── check_data.py        # Dataset integrity & missing image verification
│   └── extract_images.py    # Selective extraction utility for archive.zip
├── backend/                 # FastAPI Inference Service
│   ├── app.py               # REST API endpoints (GET /health, POST /predict)
│   ├── predictor.py         # Singleton predictor with Metal (MPS)/CPU support
│   └── requirements.txt     # Backend-specific dependencies
├── frontend/                # React + Vite + TypeScript Application
│   ├── src/                 # Dashboard UI, chart visualizations & disclaimer
│   ├── package.json         # Node scripts & dependencies
│   └── vite.config.ts       # Vite bundler configuration
├── models/                  # Checkpoints (Excluded from Git)
│   ├── chest_xray_finetuned_best.pth  # Active production model
│   └── README.md            # Checkpoint registry
├── data/                    # Dataset Storage
│   ├── Chest X Ray.csv      # Metadata annotations (14,999 rows)
│   ├── images/              # Extracted radiograph images (14,999 images)
│   └── README.md            # Data documentation
├── outputs/                 # Evaluation Artifacts (Tracked)
│   ├── metrics/             # CSV metric logs (disease_metrics, finetuned, etc.)
│   ├── graphs/              # ROC curves, AUC bar charts, loss curves
│   └── training_history/    # CSV loss & learning rate histories
├── tests/                   # Pytest Automated Test Suite
│   ├── test_dataset.py      # Patient isolation & label encoding tests
│   ├── test_split.py        # DataLoader batch shape & type assertions
│   ├── test_model.py        # Forward-pass & checkpoint weight tests
│   └── test_backend.py      # FastAPI TestClient API & inference tests
├── docs/                    # Architectural & API Documentation
│   ├── ARCHITECTURE.md      # Mathematical formulations & network details
│   └── API.md               # REST API schemas & curl examples
├── .gitignore               # Comprehensive Git exclusions
├── requirements.txt         # Consolidated Python dependencies
└── README.md                # Project documentation
```

---

## 4. Setup & Installation

### 1. Python Virtual Environment
```bash
# Clone the repository
git clone https://github.com/sai130275/chest-xray-disease-classification.git
cd "ML project/Chest-xray/Project"

# Activate existing virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## 5. Running Tests

Run the complete 11-test suite verifying dataset integrity, patient leakage, model forward pass, and FastAPI endpoints:

```bash
source venv/bin/activate
PYTHONPATH=. pytest tests/
```

Expected output:
```
tests/test_backend.py ...... [ 54%]
tests/test_dataset.py ..     [ 72%]
tests/test_model.py ..       [ 90%]
tests/test_split.py .        [100%]
====================== 11 passed in ~12s ======================
```

---

## 6. Launching the System

### Step 1: Start Backend API
```bash
source venv/bin/activate
uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```
- API Health: [http://localhost:8000/health](http://localhost:8000/health)
- API Docs (Swagger): [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 2: Start Frontend UI
In a separate terminal:
```bash
cd frontend
npm run dev
```
- Open browser at: [http://localhost:5173](http://localhost:5173)

---

## 7. Patient-Level Splitting Methodology

Chest X-ray datasets frequently feature multiple radiographs taken from the same patient over time. Splitting randomly at the image level introduces severe **patient identity leakage**, artificially inflating test accuracy.

RAD-X partitions unique **Patient IDs** strictly before image allocation:
- **Training Set**: 10,699 images (2,746 patients — 70.0%)
- **Validation Set**: 2,179 images (588 patients — 15.0%)
- **Holdout Test Set**: 2,121 images (589 patients — 15.0%)

Patient overlap assertions:
$$\text{Train Patients} \cap \text{Val Patients} = \emptyset$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
$$\text{Val Patients} \cap \text{Test Patients} = \emptyset$$

---

## 8. Git Safety & Large File Exclusions

The following large binary assets are kept **strictly local** and excluded via `.gitignore`:
- `images/` and `data/images/` (~15,000 files)
- `archive.zip` (~42 GB)
- Model checkpoints `*.pth` and `models/*.pth` (~27 MB each)
- Python virtual environment `venv/`
- Node dependencies `frontend/node_modules/`
