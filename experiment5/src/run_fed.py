"""运行 IID 或标签偏斜 FedAvg 实验。"""
import argparse
import json
import time
from pathlib import Path

from flwr.client import ClientApp
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

from .client import make_client_fn
from .strategy import LoggingFedAvg, make_evaluate_fn

NUM_CLIENTS = 5
NUM_ROUNDS = 10


def run(mode: str, seed: int = 42):
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{mode}_metrics.csv"

    client_app = ClientApp(client_fn=make_client_fn(mode, seed))

    def server_fn(context):
        strategy = LoggingFedAvg(
            csv_path=csv_path,
            evaluate_fn=make_evaluate_fn(seed),
            num_rounds=NUM_ROUNDS,
            seed=seed,
        )
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=NUM_ROUNDS),
        )

    server_app = ServerApp(server_fn=server_fn)

    t0 = time.perf_counter()
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_CLIENTS,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
    )
    wall = time.perf_counter() - t0

    timing = {
        "mode": mode,
        "wall_clock_sec_including_ray": wall,
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "seed": seed,
    }
    with open(results_dir / f"{mode}_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["iid", "skew"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.mode, args.seed)
