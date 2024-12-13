# src/ai/models/transfer_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import os

class TransferLearningModel:
    def __init__(self, model_name, num_classes, feature_extract=True):
        """Initialize the transfer learning model.

        Args:
            model_name (str): Name of the pre-trained model (e.g., 'resnet', 'vgg').
            num_classes (int): Number of output classes for the new task.
            feature_extract (bool): If True, freeze the convolutional base.
        """
        self.model = self.initialize_model(model_name, num_classes, feature_extract)

    def initialize_model(self, model_name, num_classes, feature_extract):
        """Initialize the pre-trained model.

        Args:
            model_name (str): Name of the pre-trained model.
            num_classes (int): Number of output classes.
            feature_extract (bool): If True, freeze the convolutional base.

        Returns:
            nn.Module: The initialized model.
        """
        if model_name == "resnet":
            model = models.resnet50(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_classes)
        elif model_name == "vgg":
            model = models.vgg16(pretrained=True)
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(in_features, num_classes)
        else:
            raise ValueError("Model not recognized. Please choose 'resnet' or 'vgg'.")

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.fc.parameters():  # Unfreeze the final layer
                param.requires_grad = True

        return model

    def train(self, data_loader, num_epochs, learning_rate=0.001):
        """Train the transfer learning model.

        Args:
            data_loader (DataLoader): DataLoader for the training data.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in data_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(data_loader):.4f}")

    def save_model(self, filepath):
        """Save the trained model to a file.

        Args:
            filepath (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a model from a file.

        Args:
            filepath (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
        print(f"Model loaded from {filepath}")

def get_data_loaders(data_dir, batch_size=32):
    """Get training and validation data loaders.

    Args:
        data_dir (str): Directory containing the dataset.
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize for VGG/ResNet
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    data_directory = './data'  # Path to your dataset
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Get data loaders
    train_loader, val_loader = get_data_loaders(data_directory, batch_size)

    # Initialize the transfer learning model ```python
    model_name = 'resnet'  # Choose 'resnet' or 'vgg'
    num_classes = 10  # Adjust based on your dataset
    transfer_model = TransferLearningModel(model_name, num_classes)

    # Train the model
    transfer_model.train(train_loader, num_epochs, learning_rate)

    # Save the model
    transfer_model.save_model('./models/transfer_learning_model.pth')

    # Load the model
    transfer_model.load_model('./models/transfer_learning_model.pth')
