import torch

MODEL_KEY = "model"
OPTIMIZER_KEY = "optimizer"

def save_model_checkpoint(model, optimizer, filepath):
    model_checkpoint = {MODEL_KEY: model.state_dict(), OPTIMIZER_KEY: optimizer.state_dict()}

    torch.save(model_checkpoint, filepath)

def load_model_checkpoint(model, optimizer, filepath):
    model_checkpoint = torch.load(filepath)
    model.load_state_dict(model_checkpoint[MODEL_KEY])
    optimizer.load_state_dict(model_checkpoint[OPTIMIZER_KEY])

    return model, optimizer
