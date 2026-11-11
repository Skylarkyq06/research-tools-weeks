import csv
import random
import time
from pathlib import Path

from experiment3.millionaire.paillier_compare import secure_compare

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANGES = [15, 63, 255]
PAIRS_PER_RANGE = 10
RUNS_PER_PAIR = 3
KEY_BITS = 1024
SEED = 20261111


def run_benchmark():
    random.seed(SEED)
    raw_rows = []
    summary_rows = []

    for max_val in RANGES:
        pairs = [(random.randint(0, max_val), random.randint(0, max_val))
                 for _ in range(PAIRS_PER_RANGE)]

        t_totals, t_keys, t_protocols, byte_list = [], [], [], []

        for pair_id, (a, b) in enumerate(pairs, start=1):
            expected = 1 if a > b else 0
            for run_id in range(1, RUNS_PER_PAIR + 1):
                got, t_total, t_key, nbytes = secure_compare(a, b, KEY_BITS)
                t_protocol = t_total - t_key
                assert got == expected, (a, b, expected, got)

                raw_rows.append({
                    "range": f"0~{max_val}",
                    "pair_id": pair_id,
                    "run_id": run_id,
                    "a": a,
                    "b": b,
                    "expected": expected,
                    "got": got,
                    "t_total_s": f"{t_total:.6f}",
                    "t_keygen_s": f"{t_key:.6f}",
                    "t_protocol_s": f"{t_protocol:.6f}",
                    "bytes": nbytes,
                })
                t_totals.append(t_total)
                t_keys.append(t_key)
                t_protocols.append(t_protocol)
                byte_list.append(nbytes)

        n = len(t_totals)
        summary_rows.append({
            "range": f"0~{max_val}",
            "runs": n,
            "avg_total_s": f"{sum(t_totals)/n:.6f}",
            "avg_keygen_s": f"{sum(t_keys)/n:.6f}",
            "avg_protocol_s": f"{sum(t_protocols)/n:.6f}",
            "avg_bytes": f"{sum(byte_list)/n:.1f}",
        })

    raw_path = RESULTS_DIR / "paillier_benchmark_raw.csv"
    summary_path = RESULTS_DIR / "paillier_benchmark_summary.csv"

    with raw_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"原始数据: {raw_path}")
    print(f"汇总数据: {summary_path}\n")

    header = f"{'范围':<8}{'次数':<6}{'平均总耗时(s)':<16}{'其中密钥生成(s)':<18}{'协议耗时(s)':<14}{'平均字节数':<12}"
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(f"{row['range']:<8}{row['runs']:<6}{row['avg_total_s']:<16}"
              f"{row['avg_keygen_s']:<18}{row['avg_protocol_s']:<14}{row['avg_bytes']:<12}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_benchmark()
    print(f"\n总运行时间: {time.perf_counter() - t0:.1f} s")