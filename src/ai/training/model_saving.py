# src/ai/training/model_saving.py

import torch

def save_model(model, file_path):
    """Save the model to a file.

    Args:
        model (nn.Module): The model to save.
        file_path (str): Path to save the model.
    """
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")
