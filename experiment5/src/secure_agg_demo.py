"""拓展任务：3 客户端整数向量求和的成对随机掩码安全聚合演示。

掩码构造（成对，三个客户端共享三对随机向量）：
  client 0: m0 = r01 + r02
  client 1: m1 = -r01 + r12
  client 2: m2 = -r02 - r12
  m0 + m1 + m2 = 0

上传 y_i = x_i + m_i。服务器求和：
  Σ y_i = Σ x_i + Σ m_i = Σ x_i + 0 = Σ x_i

缺少任一客户端时掩码无法抵消，服务器得不到正确和。
"""
import csv
from pathlib import Path

import numpy as np

NUM_CLIENTS = 3
VEC_LEN = 8
SEED = 42


def make_vectors(seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.integers(-50, 51, size=VEC_LEN) for _ in range(NUM_CLIENTS)]


def make_masks(seed=SEED):
    rng = np.random.default_rng(seed + 1)
    r01 = rng.integers(-100, 101, size=VEC_LEN)
    r02 = rng.integers(-100, 101, size=VEC_LEN)
    r12 = rng.integers(-100, 101, size=VEC_LEN)
    masks = [r01 + r02, -r01 + r12, -r02 - r12]
    return masks, (r01, r02, r12)


def main():
    results_dir = Path(__file__).resolve().parents[1] / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    vectors = make_vectors()
    masks, (r01, r02, r12) = make_masks()

    mask_sum = sum(masks)
    assert np.all(mask_sum == 0), "掩码和应恒为 0"
    true_sum = sum(vectors)

    print("=== 原始向量与真实总和 ===")
    for k, v in enumerate(vectors):
        print(f"  client {k}: {v.tolist()}")
    print(f"  true_sum: {true_sum.tolist()}")

    print("\n=== 成对随机掩码 ===")
    print(f"  r01 = {r01.tolist()}")
    print(f"  r02 = {r02.tolist()}")
    print(f"  r12 = {r12.tolist()}")
    print(f"  mask_sum = {mask_sum.tolist()}  (应全 0)")

    uploaded = [vectors[k] + masks[k] for k in range(NUM_CLIENTS)]
    print("\n=== 各客户端上传 y_i = x_i + m_i ===")
    for k, y in enumerate(uploaded):
        print(f"  client {k}: {y.tolist()}")

    # 场景 A：全部上传
    server_sum_a = sum(uploaded)
    ok_a = np.array_equal(server_sum_a, true_sum)
    print("\n[场景 A] 全部 3 个客户端上传")
    print(f"  服务器求和 = {server_sum_a.tolist()}")
    print(f"  真实总和   = {true_sum.tolist()}")
    print(f"  一致：{ok_a}")

    # 场景 B：客户端 2 掉线，只 0、1 上传
    server_sum_b = uploaded[0] + uploaded[1]
    partial_true = vectors[0] + vectors[1]
    residual = server_sum_b - partial_true
    ok_b = np.array_equal(server_sum_b, true_sum)
    print("\n[场景 B] 客户端 2 掉线，仅 0、1 上传")
    print(f"  服务器求和 = {server_sum_b.tolist()}")
    print(f"  x0 + x1   = {partial_true.tolist()}")
    print(f"  残留掩码   = {residual.tolist()}")
    print(f"  理论 r02+r12 = {(r02 + r12).tolist()}  (应与上面一致)")
    print(f"  是否等于真实总和：{ok_b}")

    csv_path = results_dir / "secure_agg.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "values"])
        w.writerow(["client0_vector", str(vectors[0].tolist())])
        w.writerow(["client1_vector", str(vectors[1].tolist())])
        w.writerow(["client2_vector", str(vectors[2].tolist())])
        w.writerow(["true_sum", str(true_sum.tolist())])
        w.writerow(["mask_sum", str(mask_sum.tolist())])
        w.writerow(["scenarioA_server_sum", str(server_sum_a.tolist())])
        w.writerow(["scenarioA_ok", str(ok_a)])
        w.writerow(["scenarioB_server_sum", str(server_sum_b.tolist())])
        w.writerow(["scenarioB_residual_mask", str(residual.tolist())])
        w.writerow(["scenarioB_ok", str(ok_b)])
    print(f"\nsaved {csv_path}")


if __name__ == "__main__":
    main()