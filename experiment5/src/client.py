"""Flower NumPyClient：本地 SGD 训练一轮，返回参数和样本数。"""
import numpy as np
import torch
import torch.nn as nn
from flwr.client import NumPyClient

from .data import build_splits, make_loader
from .model import MLP, get_parameters, set_parameters

_SPLIT_CACHE = {}


def _get_train_split(mode, seed=42):
    if mode not in _SPLIT_CACHE:
        train, _test, iid, skew = build_splits(seed=seed)
        _SPLIT_CACHE[mode] = (train, iid if mode == "iid" else skew)
    return _SPLIT_CACHE[mode]


def _partition_id_from_context(context):
    if hasattr(context, "node_config"):
        cfg = context.node_config
        for key in ("partition-id", "partition_id"):
            if key in cfg:
                return int(cfg[key])
    try:
        return int(context)
    except (TypeError, ValueError):
        return int(str(context).split("-")[-1])


class FlowerClient(NumPyClient):
    def __init__(self, partition_id, train_dataset, train_indices, seed=42):
        self.partition_id = int(partition_id)
        self.train_dataset = train_dataset
        self.train_indices = np.asarray(train_indices)
        self.seed = int(seed)
        self.model = MLP()
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        server_round = int(config.get("server_round", 0))
        loader = make_loader(
            self.train_dataset,
            self.train_indices,
            shuffle=True,
            seed=self.seed + server_round * 100 + self.partition_id,
        )
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=0.01, momentum=0.0, weight_decay=0.0
        )
        self.model.train()
        total_loss, total_n = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)
        return get_parameters(self.model), total_n, {"train_loss": total_loss / total_n}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loader = make_loader(self.train_dataset, self.train_indices, shuffle=False)
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, y in loader:
                logits = self.model(x)
                loss_sum += criterion(logits, y).item()
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        return loss_sum / total, total, {"local_accuracy": correct / total}


def make_client_fn(mode, seed=42):
    train_dataset, clients = _get_train_split(mode, seed)

    def client_fn(context):
        pid = _partition_id_from_context(context)
        return FlowerClient(pid, train_dataset, clients[pid], seed=seed).to_client()

    return client_fn
