import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR

BATCH_SIZE = 32
SEED = 42

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=transform
)

total_size = len(full_dataset)

train_size = int(0.70 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset,
    [train_size, val_size, test_size],
    generator=generator
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", full_dataset.classes)
print("Class mapping:", full_dataset.class_to_idx)
print("Total images:", total_size)
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))