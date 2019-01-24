import torch

MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"
TRAINING_LOSSES_KEY = "training_losses"
VALIDATION_LOSSES_KEY = "validation_losses"
TRAINING_ACCURACIES_KEY = "training_accuracies"
VALIDATION_ACCURACIES_KEY = "validation_accuracies"

def save_model_checkpoint(model, optimizer, training_losses, validation_losses, training_accuracies, validation_accuracies, filepath):
    model_checkpoint = {
        MODEL_KEY: model.state_dict(),
        OPTIMIZER_KEY: optimizer.state_dict(), 
        TRAINING_LOSSES_KEY: training_losses, 
        VALIDATION_LOSSES_KEY: validation_losses,
        TRAINING_ACCURACIES_KEY: training_accuracies,
        VALIDATION_ACCURACIES_KEY: validation_accuracies
    }

    torch.save(model_checkpoint, filepath)

def load_model_checkpoint(model, optimizer, filepath):
    model_checkpoint = torch.load(filepath)
    model.load_state_dict(model_checkpoint[MODEL_KEY])
    optimizer.load_state_dict(model_checkpoint[OPTIMIZER_KEY])

    return model, optimizer, model_checkpoint[TRAINING_LOSSES_KEY], model_checkpoint[VALIDATION_LOSSES_KEY], model_checkpoint[TRAINING_ACCURACIES_KEY], model_checkpoint[VALIDATION_ACCURACIES_KEY]
