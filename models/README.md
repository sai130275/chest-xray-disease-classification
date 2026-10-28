# Model Checkpoints

This directory houses PyTorch model weight checkpoints (`.pth`).

## Checkpoint Registry

| Checkpoint Name | Description | Status |
| :--- | :--- | :--- |
| `chest_xray_finetuned_best.pth` | **Primary Production Model**: DenseNet121 after 2-stage transfer learning fine-tuning. Best validation loss: **0.1494**. | **Active / Backend Default** |
| `chest_xray_scheduler_best.pth` | DenseNet121 with positive class weighting and ReduceLROnPlateau scheduler. | Pre-finetune baseline |
| `chest_xray_weighted_best.pth` | DenseNet121 trained with positive class weighting without scheduler. | Comparison checkpoint |
| `chest_xray_best.pth` | Initial unweighted DenseNet121 baseline checkpoint. | Historical baseline |
| `chest_xray_final_best.pth` | Intermediate stage final evaluation checkpoint. | Historical checkpoint |
| `chest_xray_model.pth` | Initial exploratory model checkpoint. | Exploration checkpoint |

## Note on Git Exclusions
All `.pth` and `.pt` binary weights are strictly excluded from version control via `.gitignore`.
Do not commit large model weights to the Git repository.
