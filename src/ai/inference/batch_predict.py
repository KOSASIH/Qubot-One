# src/ai/inference/batch_predict.py

import torch
from torch.utils.data import DataLoader

def batch_predict(model, data_loader, device):
    """Make predictions on a batch of data.

    Args:
        model (nn.Module): The trained model.
        data_loader (DataLoader): DataLoader for the input data.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        list: List of predicted outputs for each input in the batch.
    """
    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # Disable gradient calculation
        for data in data_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability
            predictions.extend(predicted.cpu().numpy())  # Store predictions

    return predictions
