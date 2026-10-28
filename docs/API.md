# RAD-X Inference API Documentation

FastAPI backend service for multi-label chest X-ray disease prediction.

## Base URL
Default local development: `http://localhost:8000`

---

## Endpoints

### 1. Health Check
`GET /health`

Verifies that the API service is active, the DenseNet121 model is loaded in memory, and reports the runtime accelerator device.

#### Response `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps",
  "checkpoint": "chest_xray_finetuned_best.pth",
  "classes_count": 15
}
```

---

### 2. Model Metadata
`GET /metadata`

Retrieves target disease classes, validation-calibrated decision thresholds, and test performance figures.

#### Response `200 OK`
```json
{
  "architecture": "DenseNet121",
  "num_classes": 15,
  "classes": ["Atelectasis", "Cardiomegaly", "..."],
  "thresholds": {
    "Atelectasis": 0.5752,
    "Cardiomegaly": 0.6858
  },
  "test_metrics": { ... },
  "disclaimer": "RESEARCH USE ONLY..."
}
```

---

### 3. Predict Multi-Label Thoracic Diseases
`POST /predict`

Accepts an uploaded chest radiograph image and executes multi-label inference.

#### Request Parameters
- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: Binary image file (JPEG, PNG, WebP). Maximum size: 15 MB.

#### Example `curl` Command
```bash
curl -X POST -F "file=@data/images/00000001_000.png" http://localhost:8000/predict
```

#### Response `200 OK`
```json
{
  "predictions": [
    {
      "disease": "Cardiomegaly",
      "probability": 0.8770,
      "threshold": 0.6858,
      "is_positive": true,
      "test_auc": 0.8548,
      "test_f1": 0.2400,
      "test_precision": 0.3750,
      "test_recall": 0.1765
    },
    {
      "disease": "Effusion",
      "probability": 0.4970,
      "threshold": 0.7964,
      "is_positive": false,
      "test_auc": 0.8096,
      "test_f1": 0.3957,
      "test_precision": 0.3834,
      "test_recall": 0.4088
    }
  ],
  "positive_findings": ["Cardiomegaly"],
  "is_no_finding": false,
  "model_info": {
    "architecture": "DenseNet121",
    "checkpoint": "chest_xray_finetuned_best.pth",
    "device": "mps",
    "classes_count": 15
  },
  "disclaimer": "EDUCATIONAL & RESEARCH PROTOTYPE ONLY..."
}
```

#### Error Codes
- `400 Bad Request`: Invalid MIME type, unreadable image bytes, or corrupted file.
- `413 Payload Too Large`: Uploaded file exceeds 15 MB.
- `503 Service Unavailable`: Model weights not loaded or failed initialization.
