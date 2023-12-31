import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Environment variables
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE=128
NUM_EPOCHS=100
NUM_WORKERS=6
CHECKPOINT_FILE="b3.pth.tar"
PIN_MEMORY=True
SAVE_MODEL=True
LOAD_MODEL=True

# Data augmentation
train_transforms = A.Compose([
    A.Resize(width=150, height=150),
    A.RandomCrop(height=120, width=120),
    A.Normalize(
        mean=[0.3199, 0.2240, 0.1609],
        std= [0.3020,0.2183, 0.1741],
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(width=150, height=150),
    A.Normalize(
        mean = [0.3199, 0.2240, 0.1609],
        std = [0.3020,0.2183, 0.1741],
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])
