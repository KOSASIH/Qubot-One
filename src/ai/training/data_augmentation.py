# src/ai/training/data_augmentation.py

from torchvision import transforms

def augment_data():
    """Define data augmentation transformations."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
