"""对照实验：标签偏斜下 5 个客户端各自独立训练 10 遍（不交换参数）。"""
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import build_splits, make_loader
from .model import MLP, build_initial_model, evaluate_model, get_parameters, set_parameters

NUM_EPOCHS = 10
LR = 0.01


def train_local(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.0, weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    return model


def main(seed=42):
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    train, test, _iid, skew = build_splits(seed=seed)
    test_loader = make_loader(test, np.arange(len(test)), shuffle=False)

    # 与联邦实验相同的初始参数
    init_model = build_initial_model(seed=seed)
    init_params = get_parameters(init_model)

    rows = []
    for k, idx in enumerate(skew):
        model = MLP()
        set_parameters(model, init_params)
        loader = make_loader(train, idx, shuffle=True, seed=seed + k)
        train_local(model, loader, epochs=NUM_EPOCHS)
        acc, loss = evaluate_model(model, test_loader)
        rows.append({"client": k, "test_accuracy": acc, "test_loss": loss})
        print(f"client {k}: acc={acc:.4f}, loss={loss:.4f}")

    csv_path = results_dir / "baseline_local.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["client", "test_accuracy", "test_loss"])
        writer.writeheader()
        writer.writerows(rows)
    print("saved", csv_path)


if __name__ == "__main__":
    main()