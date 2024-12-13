# src/ai/inference/predict.py

import torch

def predict(model, input_data, device):
    """Make a prediction using the trained model.

    Args:
        model (nn.Module): The trained model.
        input_data (torch.Tensor): Input data for prediction.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        torch.Tensor: Predicted output.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        input_data = input_data.to(device)
        output = model(input_data)
        _, predicted = torch.max(output.data, 1)  # Get the index of the max log-probability

    return predicted
