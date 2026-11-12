"""MNIST 抽取 + IID / 标签偏斜两套划分，seed=42，两套使用相同 5000 样本。"""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
NUM_CLIENTS = 5
PER_CLASS_TOTAL = 500
BATCH_SIZE = 32

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def get_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
    ])


def load_mnist():
    train = datasets.MNIST(DATA_ROOT, train=True, download=True, transform=get_transform())
    test = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=get_transform())
    return train, test


def sample_500_per_class(train_dataset, seed=42):
    targets = np.array(train_dataset.targets)
    rng = np.random.default_rng(seed)
    picked = []
    for c in range(10):
        cls_idx = np.where(targets == c)[0]
        picked.append(rng.choice(cls_idx, size=PER_CLASS_TOTAL, replace=False))
    return np.sort(np.concatenate(picked))


def split_iid(indices, labels, seed=42):
    rng = np.random.default_rng(seed)
    clients = [[] for _ in range(NUM_CLIENTS)]
    for c in range(10):
        cls_idx = indices[labels[indices] == c]
        cls_idx = rng.permutation(cls_idx)
        for k in range(NUM_CLIENTS):
            clients[k].extend(cls_idx[k * 100:(k + 1) * 100])
    return [np.sort(np.array(ci)) for ci in clients]


def split_skew(indices, labels):
    clients = [[] for _ in range(NUM_CLIENTS)]
    for k in range(NUM_CLIENTS):
        for c in (2 * k, 2 * k + 1):
            clients[k].extend(indices[labels[indices] == c])
    return [np.sort(np.array(ci)) for ci in clients]


def build_splits(seed=42):
    train, test = load_mnist()
    labels = np.array(train.targets)
    base = sample_500_per_class(train, seed=seed)
    iid = split_iid(base, labels, seed=seed)
    skew = split_skew(base, labels)
    return train, test, iid, skew


def make_loader(dataset, indices, shuffle=False, seed=42):
    subset = Subset(dataset, indices.tolist())
    if shuffle:
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True, generator=gen)
    return DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
