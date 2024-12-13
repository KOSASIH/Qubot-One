# src/ai/training/hyperparameter_tuning.py

from sklearn.model_selection import ParameterGrid

def tune_hyperparameters(model, train_loader, val_loader, param_grid, num_epochs, device):
    """Tune hyperparameters using grid search.

    Args:
        model (nn.Module): The model to tune.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        param_grid (dict): Dictionary with parameters names as keys and lists of parameter settings to try as values.
        num_epochs (int): Number of epochs to train for each parameter setting.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        dict: Best parameters and corresponding accuracy.
    """
    best_accuracy = 0
    best_params = {}

    for params in ParameterGrid(param_grid):
        print(f"Trying parameters: {params}")
        model_copy = model.__class__(*model.args, **model.kwargs)  # Create a new instance of the model
        model_copy.to(device)

        # Train the model with the current parameters
        train_model(model_copy, train_loader, num_epochs, params['learning_rate'], device)

        # Evaluate the model
        accuracy = evaluate_model(model_copy, val_loader, device)

        if accuracy > best_accuracy best_accuracy = accuracy
            best_params = params

    print(f"Best parameters: {best_params}, Best accuracy: {best_accuracy:.2f}%")
    return best_params, best_accuracy
