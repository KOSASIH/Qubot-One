# src/ai/training/evaluate.py

import torch

def evaluate_model(model, val_loader, device):
    """Evaluate the model.

    Args:
        model (nn.Module): The model to evaluate.
        val_loader (DataLoader): DataLoader for validation data.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        float: Accuracy of the model on the validation set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    return accuracy
