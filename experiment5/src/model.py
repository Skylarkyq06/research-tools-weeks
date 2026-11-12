"""小型全连接分类网络，MNIST 用。784 -> 128 -> 10。"""
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), parameters):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)
    return model


def build_initial_model(seed=42):
    torch.manual_seed(seed)
    return MLP()


def evaluate_model(model, test_loader, device="cpu"):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total, loss_sum / total
