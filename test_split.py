from dataset import get_dataloaders

train_loader, val_loader, mlb = get_dataloaders()

print("Train batches:", len(train_loader))
print("Validation batches:", len(val_loader))
print("Classes:", mlb.classes_)

images, labels = next(iter(train_loader))

print("Image shape:", images.shape)
print("Label shape:", labels.shape)