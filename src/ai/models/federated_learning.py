# src/ai/models/federated_learning.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

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

class Client:
    def __init__(self, model, data_loader, device):
        """Initialize the client.

        Args:
            model (nn.Module): The model to train.
            data_loader (DataLoader): DataLoader for the client's data.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device

    def train(self, num_epochs, learning_rate):
        """Train the model on the client's local data.

        Args:
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
        """
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(num_epochs):
            for data, target in self.data_loader:
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = self.model(data.view(data.size(0), -1))  # Flatten the input
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

class Server:
    def __init__(self, model, device):
        """Initialize the server.

        Args:
            model (nn.Module): The global model.
            device (torch.device): Device to run the model on (CPU or GPU).
        """
        self.model = model.to(device)
        self.device = device

    def aggregate_models(self, client_models):
        """Aggregate the models from clients.

        Args:
            client_models (list): List of client models.

        Returns:
            nn.Module: The aggregated global model.
        """
        global_state_dict = self.model.state_dict()

        for key in global_state_dict.keys():
            global_state_dict[key] = torch.mean(
                torch.stack([client_model.state_dict()[key] for client_model in client_models]), dim=0
            )

        self.model.load_state_dict(global_state_dict)

def get_data_loaders(data, batch_size=32):
    """Get data loaders for clients.

    Args:
        data (list): List of datasets for each client.
        batch_size (int): Batch size for data loaders.

    Returns:
        list: List of DataLoaders for each client.
    """
    data_loaders = []
    for client_data in data:
        data_loader = DataLoader(client_data, batch_size=batch_size, shuffle=True)
        data_loaders.append(data_loader)
    return data_loaders

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    input_size = 28 * 28  # MNIST images are 28x28
    hidden_size = 128
    output_size = 10  # Number of output classes for MNIST
    num_epochs = 1
    learning_rate = 0.001
    num_clients = 5

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Split dataset into clients
    client_data = np.array_split(dataset, num_clients)
    data_loaders = get_data_loaders(client_data)

    # Initialize global model and server
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global_model = SimpleNN(input_size, hidden_size, output_size)
    server = Server(global_model, device)
