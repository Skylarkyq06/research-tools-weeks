"""自定义 FedAvg：每轮聚合后评估全局模型，记录指标与耗时到 CSV。"""
import csv
import time
from pathlib import Path

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg

from .data import load_mnist, make_loader
from .model import MLP, build_initial_model, evaluate_model, get_parameters, set_parameters


def weighted_average(metrics):
    """按样本数加权聚合客户端 fit 指标（train_loss）。"""
    total = sum(num for num, _ in metrics)
    loss = sum(num * m["train_loss"] for num, m in metrics) / total
    return {"train_loss": loss}


def make_evaluate_fn(seed=42):
    _train, test = load_mnist()
    test_loader = make_loader(test, np.arange(len(test)), shuffle=False)

    def evaluate(server_round, parameters, config):
        model = MLP()
        set_parameters(model, parameters)
        acc, loss = evaluate_model(model, test_loader)
        return float(loss), {"accuracy": float(acc)}

    return evaluate


class LoggingFedAvg(FedAvg):
    def __init__(self, csv_path, evaluate_fn, num_rounds=10, seed=42):
        initial_model = build_initial_model(seed=seed)
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=5,
            min_evaluate_clients=5,
            min_available_clients=5,
            evaluate_fn=evaluate_fn,
            initial_parameters=ndarrays_to_parameters(get_parameters(initial_model)),
            fit_metrics_aggregation_fn=weighted_average,
        )
        self.csv_path = Path(csv_path)
        self.num_rounds = num_rounds
        self.evaluate_fn = evaluate_fn
        self._round_start = None
        self.round_times = []
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                ["round", "test_loss", "test_accuracy", "train_loss", "round_time_sec"]
            )

    def configure_fit(self, server_round, parameters, client_manager):
        self._round_start = time.perf_counter()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        train_loss = None if not metrics else metrics.get("train_loss")

        test_loss, test_acc = None, None
        if params is not None:
            ndarrays = parameters_to_ndarrays(params)
            test_loss, eval_metrics = self.evaluate_fn(server_round, ndarrays, {})
            test_acc = eval_metrics.get("accuracy")

        rt = None if self._round_start is None else time.perf_counter() - self._round_start
        if rt is not None:
            self.round_times.append(rt)
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([server_round, test_loss, test_acc, train_loss, rt])
        return params, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        # 评估已在 aggregate_fit 内完成并写 CSV，这里不再重复记录
        return None, {}