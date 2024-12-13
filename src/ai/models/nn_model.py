# src/ai/models/nn_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import GradScaler, autocast
import os

class AdvancedNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        """Initialize the advanced neural network architecture.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Number of output classes.
        """
        super(AdvancedNN, self).__init__()
        layers = []
        last_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))  # Batch normalization
            last_size = hidden_size
        
        layers.append(nn.Linear(last_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralNetworkTrainer:
    def __init__(self, model, criterion, optimizer, device):
        """Initialize the trainer for the neural network.

        Args:
            model (nn.Module): The neural network model.
            criterion (nn.Module): Loss function.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scaler = GradScaler()  # For mixed precision training

    def train(self, data_loader, num_epochs, scheduler=None):
        """Train the neural network model.

        Args:
            data_loader (DataLoader): DataLoader for training data.
            num_epochs (int): Number of epochs to train.
            scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler.
        """
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Mixed precision training
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item()

            if scheduler:
                scheduler.step()

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
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_sizes = [128, 64]  # Example hidden layer sizes
    output_size = 10  ```python
    # Number of output classes for MNIST
    num_epochs = 10
    batch_size = 32
    learning_rate = 0.001

    # Initialize data loaders
    train_loader, val_loader = get_data_loaders(batch_size)

    # Initialize model, criterion, optimizer, and scheduler
    model = AdvancedNN(input_size, hidden_sizes, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # Learning rate scheduler

    # Create trainer instance
    trainer = NeuralNetworkTrainer(model, criterion, optimizer, device)

    # Train the model
    trainer.train(train_loader, num_epochs, scheduler)

    # Save the trained model
    model_save_path = os.path.join('models', 'advanced_nn_model.pth')
    trainer.save_model(model_save_path)

    # Load the model (optional)
    trainer.load_model(model_save_path)
