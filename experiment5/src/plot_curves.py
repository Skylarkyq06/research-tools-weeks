"""从 results/*.csv 绘制训练曲线，输出到 report/figures/。"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
RESULTS = BASE / "results"
FIGURES = BASE / "report" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(rows, key):
    return [float(r[key]) if r[key] not in ("", None) else None for r in rows]


def plot_metric(key, ylabel, fname):
    plt.figure(figsize=(6, 4))
    for mode, label in (("iid", "IID"), ("skew", "Label-skew")):
        rows = read_csv(RESULTS / f"{mode}_metrics.csv")
        rounds = [int(r["round"]) for r in rows]
        vals = to_float(rows, key)
        plt.plot(rounds, vals, marker="o", label=label)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 11))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = FIGURES / fname
    plt.savefig(out, dpi=200)
    plt.close()
    print("saved", out)


def main():
    plot_metric("test_accuracy", "Test accuracy", "fig_accuracy.png")
    plot_metric("test_loss", "Test loss (per-sample mean)", "fig_test_loss.png")
    plot_metric("train_loss", "Client train loss (weighted)", "fig_train_loss.png")


if __name__ == "__main__":
    main()