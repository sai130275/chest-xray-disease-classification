
import csv
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch.nn as nn

from torchvision import models
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score
)
from dataset import get_dataloaders


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Device:", device)

_, val_loader, test_loader, mlb = get_dataloaders(batch_size=32)
num_classes = len(mlb.classes_)


# Load final weighted checkpoint
model = models.densenet121(weights=None)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)
model.load_state_dict(
    torch.load(
        "chest_xray_finetuned_best.pth",
        map_location="cpu",
        weights_only=True
    )
)
model = model.to(device)
model.eval()


def predict(loader):
    labels_all, probs_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(device))
            probs = torch.sigmoid(outputs)

            labels_all.append(labels.numpy())
            probs_all.append(probs.cpu().numpy())

    return np.concatenate(labels_all), np.concatenate(probs_all)


# Select thresholds using validation data only
y_val, p_val = predict(val_loader)
thresholds = []

for i, disease in enumerate(mlb.classes_):
    precision_curve, recall_curve, threshold_curve = (
        precision_recall_curve(y_val[:, i], p_val[:, i])
    )

    if len(threshold_curve) == 0:
        threshold = 0.5
    else:
        f1_curve = (
            2 * precision_curve[:-1] * recall_curve[:-1]
            / (precision_curve[:-1] + recall_curve[:-1] + 1e-8)
        )
        threshold = float(threshold_curve[np.argmax(f1_curve)])

    thresholds.append(threshold)
    print(f"{disease}: validation threshold={threshold:.4f}")


# Evaluate once on the untouched test set
y_true, y_prob = predict(test_loader)
y_pred = (y_prob >= np.array(thresholds)).astype(int)


# Disease-wise metrics
auc_scores = []
rows = []

for i, disease in enumerate(mlb.classes_):
    try:
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
    except ValueError:
        auc = np.nan

    f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    precision = precision_score(
        y_true[:, i], y_pred[:, i], zero_division=0
    )
    recall = recall_score(
        y_true[:, i], y_pred[:, i], zero_division=0
    )

    auc_scores.append(auc)
    rows.append([
        disease,
        round(float(auc), 4) if np.isfinite(auc) else "",
        round(float(f1), 4),
        round(float(precision), 4),
        round(float(recall), 4),
        round(thresholds[i], 4)
    ])


with open("finetuned_test_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Disease", "AUC_ROC", "F1",
        "Precision", "Recall", "Validation_Threshold"
    ])
    writer.writerows(rows)


mean_auc = np.nanmean(auc_scores)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
macro_precision = precision_score(
    y_true, y_pred, average="macro", zero_division=0
)
macro_recall = recall_score(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n===== FINAL TEST RESULTS =====")
print(f"Mean AUC-ROC: {mean_auc:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Macro Precision: {macro_precision:.4f}")
print(f"Macro Recall:    {macro_recall:.4f}")


# AUC by disease
plt.figure(figsize=(12, 6))
plt.bar(mlb.classes_, auc_scores)
plt.xticks(rotation=60, ha="right")
plt.ylabel("AUC-ROC")
plt.ylim(0, 1)
plt.title("Final Test AUC-ROC by Disease")
plt.tight_layout()
plt.savefig("finetuned_test_auc.png", dpi=300, bbox_inches="tight")
plt.close()


# ROC curves
plt.figure(figsize=(10, 8))
for i, disease in enumerate(mlb.classes_):
    if np.isfinite(auc_scores[i]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f"{disease} ({auc_scores[i]:.2f})")

plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Final Test ROC Curves")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("finetuned_test_roc.png", dpi=300, bbox_inches="tight")
plt.close()

print("\nGenerated:")
print("finetuned_test_metrics.csv")
print("finetuned_test_auc.png")
print("finetuned_test_roc.png")
