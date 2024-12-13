# src/ai/models/adversarial_models.py

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

def fgsm_attack(model, data, target, epsilon):
    """Generate adversarial examples using the FGSM method.

    Args:
        model (nn.Module): The neural network model.
        data (torch.Tensor): Input data.
        target (torch.Tensor): True labels.
        epsilon (float): Perturbation amount.

    Returns:
        torch.Tensor: Adversarial examples.
    """
    # Set requires_grad attribute of tensor to True
    data.requires_grad = True

    # Forward pass
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Collect the gradients of the data
    data_grad = data.grad.data

    # Create the adversarial example
    adversarial_data = data + epsilon * data_grad.sign()
    adversarial_data = torch.clamp(adversarial_data, 0, 1)  # Ensure values are in [0, 1]

    return adversarial_data

def train_with_adversarial_examples(model, train_loader, optimizer, num_epochs, epsilon):
    """Train the model with adversarial examples.

    Args:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.
        epsilon (float): Perturbation amount for adversarial examples.
    """
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            optimizer.zero_grad()

            # Generate adversarial examples
            adversarial_data = fgsm_attack(model, data, target, epsilon)

            # Forward pass with adversarial examples
            output = model(adversarial_data)
            loss = nn.CrossEntropyLoss()(output, target)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

def get_data_loaders(batch_size=32):
    """Get training and validation data loaders.

    Args:
        batch_size (int): Batch size for data loaders.

    Returns:
        tuple: Training and validation DataLoaders.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale images
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 10  # Number of output classes for MNIST
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    epsilon = 0.1  # Perturbation amount for FGSM

    # Get data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Initialize model, optimizer
    model = SimpleNN (input_size, hidden_size, output_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model with adversarial examples
    train_with_adversarial_examples(model, train_loader, optimizer, num_epochs, epsilon)

    # Evaluate the model on validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data.view(-1, input_size))
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')
