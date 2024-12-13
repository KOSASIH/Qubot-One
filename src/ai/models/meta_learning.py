# src/ai/models/meta_learning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
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

def train_on_task(model, data_loader, num_epochs, learning_rate):
    """Train the model on a single task.

    Args:
        model (nn.Module): The model to train.
        data_loader (DataLoader): DataLoader for the task's data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
    """
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(num_epochs):
        for data, target in data_loader:
            optimizer.zero_grad()
            output = model(data.view(data.size(0), -1))  # Flatten the input
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

def meta_train(model, tasks, num_epochs, inner_lr, outer_lr):
    """Meta-train the model on multiple tasks.

    Args:
        model (nn.Module): The model to train.
        tasks (list): List of tasks (datasets).
        num_epochs (int): Number of epochs to train.
        inner_lr (float): Learning rate for inner loop (task-specific).
        outer_lr (float): Learning rate for outer loop (meta-update).
    """
    outer_optimizer = optim.Adam(model.parameters(), lr=outer_lr)

    for epoch in range(num_epochs):
        meta_loss = 0.0

        for task in tasks:
            # Split task into support and query sets
            support_size = int(0.8 * len(task))
            support_set, query_set = random_split(task, [support_size, len(task) - support_size])
            support_loader = DataLoader(support_set, batch_size=5, shuffle=True)
            query_loader = DataLoader(query_set, batch_size=5, shuffle=True)

            # Clone the model for inner training
            task_model = SimpleNN(input_size=28*28, hidden_size=128, output_size=10)
            task_model.load_state_dict(model.state_dict())

            # Inner loop: train on the support set
            train_on_task(task_model, support_loader, num_epochs=1, learning_rate=inner_lr)

            # Evaluate on the query set
            task_model.eval()
            query_loss = 0.0
            criterion = nn.CrossEntropyLoss()
            with torch.no_grad():
                for data, target in query_loader:
                    output = task_model(data.view(data.size(0), -1))
                    loss = criterion(output, target)
                    query_loss += loss.item()

            # Meta loss is the average loss on the query set
            meta_loss += query_loss / len(query_loader)

        # Update the meta-model
        outer_optimizer.zero_grad()
        meta_loss.backward()
        outer_optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Meta Loss: {meta_loss / len(tasks):.4f}")

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
    return dataset

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 10  # Number of output classes
    num_epochs = 10
    inner_lr = 0.01
    outer_lr = 0.001
    num_tasks = 5

    # Get the dataset
    dataset = get_data_loaders(batch_size)

    # Create tasks by splitting the dataset into multiple subsets
    tasks = [random_split(dataset, [len(dataset) // num_tasks] * num_tasks) for _ in range(num_tasks)]

    # Initialize model
    model = SimpleNN(input_size, hidden_size, output_size)

    # Meta-training loop
    meta_train(model, tasks, num_epochs, inner_lr, outer_lr)

    print("Meta-training complete.")
