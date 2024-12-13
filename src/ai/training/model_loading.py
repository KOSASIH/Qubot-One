# src/ai/training/model_loading.py

import torch

def load_model(model, file_path):
    """Load the model from a file.

    Args:
        model (nn.Module): The model to load the weights into.
        file_path (str): Path to the saved model.
    """
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")
