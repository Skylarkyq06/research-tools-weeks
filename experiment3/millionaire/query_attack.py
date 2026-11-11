import csv
import random
from pathlib import Path

from experiment4.paillier.paillier import keygen, encrypt, decrypt

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KEY_BITS = 1024


def _get_n(pub):
    for name in ("n", "N", "modulus"):
        if hasattr(pub, name):
            return getattr(pub, name)
    raise AttributeError("无法确定公钥模数属性名")


def setup_key(key_bits=KEY_BITS):
    kp = keygen(key_bits)
    pub, priv = (kp[0], kp[1]) if isinstance(kp, tuple) else (kp, kp)
    return pub, priv, _get_n(pub)


def compare_with_key(a: int, b: int, pub, priv, n: int) -> int:
    """复用固定密钥的安全比较，返回 1 表示 a > b。"""
    ca = encrypt(pub, a)
    cb_neg = encrypt(pub, (-b) % n)
    c_diff = (ca * cb_neg) % (n ** 2)
    diff = decrypt(priv, c_diff)
    if diff == 0:
        return 0
    return 1 if diff <= n // 2 else 0


def binary_search_attack(secret_a, pub, priv, n, max_val=255):
    lo, hi = 0, max_val
    queries = 0
    history = []
    while lo < hi:
        mid = (lo + hi) // 2
        result = compare_with_key(secret_a, mid, pub, priv, n)
        queries += 1
        history.append({"b": mid, "result": result})
        if result == 1:
            lo = mid + 1
        else:
            hi = mid
    return lo, queries, history


def linear_attack(secret_a, pub, priv, n, max_val=255):
    """从 b=0 往上问，找第一个 a > b 不成立的 b，此时 a == b。"""
    queries = 0
    for b in range(max_val + 1):
        result = compare_with_key(secret_a, b, pub, priv, n)
        queries += 1
        if result == 0:
            return b, queries
    return max_val, queries


def run_attack_experiment():
    random.seed(20261111)

    print("生成一对固定 Paillier 密钥，供所有查询复用……")
    pub, priv, n = setup_key(KEY_BITS)
    print("密钥就绪。\n")

    # 二分：20 个样本
    binary_samples = [0, 1, 42, 127, 128, 200, 254, 255]
    binary_samples += [random.randint(0, 255) for _ in range(12)]

    binary_rows = []
    for a in binary_samples:
        inferred, q, history = binary_search_attack(a, pub, priv, n)
        binary_rows.append({
            "secret_a": a,
            "inferred": inferred,
            "queries": q,
            "correct": inferred == a,
            "query_history": ";".join(f"b={h['b']}->{h['result']}" for h in history),
        })

    # 线性：只取 5 个代表值，避免耗时过长
    linear_samples = [0, 1, 42, 128, 255]
    linear_rows = []
    for a in linear_samples:
        inferred, q = linear_attack(a, pub, priv, n)
        linear_rows.append({
            "secret_a": a,
            "inferred": inferred,
            "queries": q,
            "correct": inferred == a,
        })

    path = RESULTS_DIR / "query_attack.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "secret_a", "inferred", "queries", "correct", "detail"])
        for r in binary_rows:
            writer.writerow(["binary", r["secret_a"], r["inferred"], r["queries"],
                             r["correct"], r["query_history"]])
        for r in linear_rows:
            writer.writerow(["linear", r["secret_a"], r["inferred"], r["queries"],
                             r["correct"], ""])

    print(f"数据: {path}\n")

    print("二分攻击（20 个样本）：")
    print(f"{'a':<6}{'推断':<8}{'次数':<8}{'正确':<6}")
    for r in binary_rows:
        print(f"{r['secret_a']:<6}{r['inferred']:<8}{r['queries']:<8}{'是' if r['correct'] else '否':<6}")
    avg_bin = sum(r["queries"] for r in binary_rows) / len(binary_rows)
    print(f"平均查询次数: {avg_bin:.2f}（理论上界 ⌈log2(256)⌉ = 8）\n")

    print("线性攻击（5 个代表样本）：")
    print(f"{'a':<6}{'推断':<8}{'次数':<8}{'正确':<6}")
    for r in linear_rows:
        print(f"{r['secret_a']:<6}{r['inferred']:<8}{r['queries']:<8}{'是' if r['correct'] else '否':<6}")
    avg_lin = sum(r["queries"] for r in linear_rows) / len(linear_rows)
    print(f"平均查询次数: {avg_lin:.2f}（最坏 256）")
    print(f"二分相对线性的加速比: {avg_lin / avg_bin:.1f}×")


if __name__ == "__main__":
    run_attack_experiment()