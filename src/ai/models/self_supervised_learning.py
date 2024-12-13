# src/ai/models/self_supervised_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """Initialize a simple feedforward neural network.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden units.
            output_size (int): Number of output classes.
        """
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        """Initialize the contrastive loss.

        Args:
            temperature (float): Temperature parameter for scaling.
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute the contrastive loss.

        Args:
            z_i (torch.Tensor): Encoded representation of the first sample.
            z_j (torch.Tensor): Encoded representation of the second sample.

        Returns:
            torch.Tensor: Computed contrastive loss.
        """
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # Compute similarity scores
        similarity = torch.matmul(z_i, z_j.T) / self.temperature
        labels = torch.arange(similarity.size(0)).to(similarity.device)

        # Compute loss
        loss = nn.CrossEntropyLoss()(similarity, labels)
        return loss

def get_data_loaders(batch_size=32):
    """Get data loaders for the MNIST dataset.

    Args:
        batch_size (int): Batch size for data loaders.

    Returns:
        DataLoader: DataLoader for the MNIST dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 64  # Dimensionality of the representation
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Get data loader
    data_loader = get_data_loaders(batch_size)

    # Initialize model and optimizer
    model = SimpleNN(input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = ContrastiveLoss(temperature=0.1)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for data, _ in data_loader:
            # Create positive pairs by augmenting the same data
            data1 = data.view(data.size(0), -1)  # Flatten the input
            data2 = data.view(data.size(0), -1)  # In a real scenario, apply different augmentations

            optimizer.zero_grad()

            # Forward pass
            z1 = model(data1)
            z2 = model(data2)

            # Compute loss
            loss = criterion(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(data_loader):.4f}")

    print("Training complete.")
