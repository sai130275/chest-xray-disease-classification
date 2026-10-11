
import copy
import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import models
from dataset import get_dataloaders


# Configuration
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

EPOCHS = 6
PATIENCE = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_CAP = 10.0

BEST_MODEL_PATH = "chest_xray_final_best.pth"

print("Device:", device)


# Load the three data splits
train_loader, val_loader, test_loader, mlb = get_dataloaders(
    batch_size=BATCH_SIZE
)

num_classes = len(mlb.classes_)
print("Classes:", mlb.classes_)


# Calculate positive-class weights from training data only
positive_counts = torch.zeros(num_classes)
total_samples = 0

for _, labels in train_loader:
    positive_counts += labels.sum(dim=0)
    total_samples += labels.size(0)

negative_counts = total_samples - positive_counts

pos_weights = (
    negative_counts / positive_counts.clamp(min=1)
).clamp(max=WEIGHT_CAP).to(device)

print("\nPositive-class weights:")
for name, weight in zip(mlb.classes_, pos_weights.cpu().tolist()):
    print(f"{name}: {weight:.2f}")


# Initialize pretrained DenseNet121
model = models.densenet121(weights="DEFAULT")
model.classifier = nn.Linear(
    model.classifier.in_features, num_classes
)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
optimizer = torch.optim.Adam(
    model.parameters(), lr=LEARNING_RATE
)


# Training with best-checkpoint saving and early stopping
best_val_loss = float("inf")
best_state = None
patience_counter = 0
history = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    history.append([epoch + 1, train_loss, val_loss])

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, BEST_MODEL_PATH)
        patience_counter = 0
        print("  Best checkpoint saved.")
    else:
        patience_counter += 1
        print(f"  No improvement: {patience_counter}/{PATIENCE}")

    if patience_counter >= PATIENCE:
        print("Early stopping triggered.")
        break


# Save training history
with open("final_training_history.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Validation Loss"])
    writer.writerows(history)


# Plot loss curves
epochs_run = [row[0] for row in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs_run, [row[1] for row in history],
         marker="o", label="Training Loss")
plt.plot(epochs_run, [row[2] for row in history],
         marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted BCE Loss")
plt.title("Final Model Training History")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("final_training_history.png", dpi=300,
            bbox_inches="tight")
plt.close()

print("\nTraining complete.")
print(f"Best validation loss: {best_val_loss:.4f}")
print(f"Checkpoint: {BEST_MODEL_PATH}")
print("The test set was not used for training or checkpoint selection.")
