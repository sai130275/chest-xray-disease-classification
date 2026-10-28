"""
RAD-X Model Fine-Tuning Module
Executes two-stage transfer learning fine-tuning:
Stage 1: Frozen backbone, fine-tuning classification head.
Stage 2: End-to-end full network fine-tuning with differential learning rates.
"""

import copy
import csv
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import models

try:
    from ml.dataset import get_dataloaders
    from ml.config import (
        MODELS_DIR,
        TRAINING_HISTORY_DIR,
        GRAPHS_DIR,
    )
except ImportError:
    from dataset import get_dataloaders
    from config import (
        MODELS_DIR,
        TRAINING_HISTORY_DIR,
        GRAPHS_DIR,
    )

def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    BATCH_SIZE = 32
    EPOCHS = 6
    PATIENCE = 2
    HEAD_LR = 1e-4
    BACKBONE_LR = 1e-5

    model_in = MODELS_DIR / "chest_xray_scheduler_best.pth"
    if not model_in.exists():
        # Fallback to root if not yet moved
        model_in = "chest_xray_scheduler_best.pth"

    model_out = MODELS_DIR / "chest_xray_finetuned_best.pth"

    train_loader, val_loader, _, mlb = get_dataloaders(batch_size=BATCH_SIZE)
    n = len(mlb.classes_)

    # Compute class weights from training data only
    positive = torch.zeros(n)
    total = 0
    for _, y in train_loader:
        positive += y.sum(0)
        total += y.size(0)
    pos_weight = ((total - positive) / positive.clamp(min=1)).clamp(max=10).to(device)

    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, n)
    model.load_state_dict(torch.load(model_in, map_location="cpu", weights_only=True))
    model = model.to(device)

    # Stage 1: train classifier head while backbone is frozen
    for p in model.features.parameters():
        p.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=HEAD_LR)
    history = []
    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    def run_epoch(loader, training):
        model.train() if training else model.eval()
        loss_sum = 0.0
        with torch.set_grad_enabled(training):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if training:
                    optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                if training:
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item()
        return loss_sum / len(loader)

    print("Stage 1: training classifier head")
    for epoch in range(2):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        history.append([epoch + 1, "head", tr, vl])
        print(f"Head epoch {epoch+1}/2 | Train {tr:.4f} | Val {vl:.4f}")

        if vl < best_loss:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())

    # Stage 2: fine-tune the full network at a lower learning rate
    for p in model.features.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.features.parameters(), "lr": BACKBONE_LR},
        {"params": model.classifier.parameters(), "lr": HEAD_LR / 10}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=1, min_lr=1e-7
    )
    patience_count = 0

    print("Stage 2: fine-tuning full network")
    for epoch in range(EPOCHS):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        scheduler.step(vl)
        history.append([epoch + 1, "full", tr, vl])
        print(f"Fine-tune epoch {epoch+1}/{EPOCHS} | Train {tr:.4f} | Val {vl:.4f}")

        if vl < best_loss:
            best_loss = vl
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, model_out)
            patience_count = 0
            print("  Best checkpoint saved.")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping.")
                break

    # Ensure a checkpoint exists even if full fine-tuning never improves
    torch.save(best_state, model_out)

    csv_out = TRAINING_HISTORY_DIR / "finetune_history.csv"
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Stage", "Train Loss", "Validation Loss"])
        writer.writerows(history)

    plt.figure(figsize=(8, 5))
    for stage in ("head", "full"):
        rows = [r for r in history if r[1] == stage]
        if rows:
            plt.plot(range(1, len(rows)+1), [r[2] for r in rows], marker="o", label=f"{stage} train")
            plt.plot(range(1, len(rows)+1), [r[3] for r in rows], marker="o", linestyle="--", label=f"{stage} validation")
    plt.xlabel("Epoch within stage")
    plt.ylabel("Weighted BCE loss")
    plt.title("Staged DenseNet121 Fine-tuning")
    plt.legend()
    plt.tight_layout()
    plot_out = GRAPHS_DIR / "finetune_history.png"
    plt.savefig(plot_out, dpi=300)
    plt.close()

    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Saved model: {model_out}")
    print(f"History CSV: {csv_out}")
    print(f"History Plot: {plot_out}")
    print("Test set was not used.")

if __name__ == "__main__":
    main()
