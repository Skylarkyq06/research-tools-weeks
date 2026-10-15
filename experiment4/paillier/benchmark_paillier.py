"""Paillier 性能测量：512/1024/2048 bit，自行实现 vs phe。"""
import time
import statistics
import secrets
from paillier import keygen, encrypt, decrypt, add, mul_plain, ciphertext_bytes
from phe import paillier as phe_paillier

BIT_LENGTHS = [512, 1024, 2048]
KEY_REPEATS = 3
PAIRS_N = 20
ROUNDS = 5


def _median(xs):
    return statistics.median(xs)


def measure_own(bits):
    """自行实现：3 次密钥生成取中位数，5 轮操作取中位数。"""
    key_times = []
    pub = priv = None
    for _ in range(KEY_REPEATS):
        t0 = time.perf_counter()
        pub, priv = keygen(bits=bits)
        t1 = time.perf_counter()
        key_times.append((t1 - t0) * 1000)

    n = pub.n
    msgs = [secrets.randbelow(1001) for _ in range(PAIRS_N)]

    # 预热
    c_warm = encrypt(pub, 123)
    _ = decrypt(priv, c_warm)

    enc_rounds, add_rounds, dec_rounds = [], [], []
    for _ in range(ROUNDS):
        # 加密
        t0 = time.perf_counter()
        cs = [encrypt(pub, m) for m in msgs]
        t1 = time.perf_counter()
        enc_rounds.append((t1 - t0) * 1000 / PAIRS_N)

        # 同态加法：20 对
        t0 = time.perf_counter()
        for i in range(PAIRS_N - 1):
            _ = add(pub, cs[i], cs[i + 1])
        t1 = time.perf_counter()
        add_rounds.append((t1 - t0) * 1000 / (PAIRS_N - 1))

        # 解密
        t0 = time.perf_counter()
        for c in cs:
            _ = decrypt(priv, c)
        t1 = time.perf_counter()
        dec_rounds.append((t1 - t0) * 1000 / PAIRS_N)

    # 密文大小：取一个样本
    sample_c = cs[0]
    csize = ciphertext_bytes(sample_c)

    return {
        "bits": bits,
        "n_bits": n.bit_length(),
        "keygen_ms": _median(key_times),
        "encrypt_ms": _median(enc_rounds),
        "add_ms": _median(add_rounds),
        "decrypt_ms": _median(dec_rounds),
        "ciphertext_bytes": csize,
    }


def measure_phe(bits):
    """phe 库：同样口径。"""
    key_times = []
    pk = sk = None
    for _ in range(KEY_REPEATS):
        t0 = time.perf_counter()
        pk, sk = phe_paillier.generate_paillier_keypair(n_length=bits)
        t1 = time.perf_counter()
        key_times.append((t1 - t0) * 1000)

    msgs = [secrets.randbelow(1001) for _ in range(PAIRS_N)]

    # 预热
    c_warm = pk.encrypt(123)
    _ = sk.decrypt(c_warm)

    enc_rounds, add_rounds, dec_rounds = [], [], []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        cs = [pk.encrypt(m) for m in msgs]
        t1 = time.perf_counter()
        enc_rounds.append((t1 - t0) * 1000 / PAIRS_N)

        t0 = time.perf_counter()
        for i in range(PAIRS_N - 1):
            _ = cs[i] + cs[i + 1]
        t1 = time.perf_counter()
        add_rounds.append((t1 - t0) * 1000 / (PAIRS_N - 1))

        t0 = time.perf_counter()
        for c in cs:
            _ = sk.decrypt(c)
        t1 = time.perf_counter()
        dec_rounds.append((t1 - t0) * 1000 / PAIRS_N)

    sample_c = cs[0].ciphertext()
    csize = (sample_c.bit_length() + 7) // 8

    return {
        "bits": bits,
        "n_bits": pk.n.bit_length(),
        "keygen_ms": _median(key_times),
        "encrypt_ms": _median(enc_rounds),
        "add_ms": _median(add_rounds),
        "decrypt_ms": _median(dec_rounds),
        "ciphertext_bytes": csize,
    }


def main():
    header = (f"{'实现':<8}{'模数bit':<10}{'n.bit':<8}"
              f"{'keygen/ms':<14}{'加密/ms':<12}{'加法/ms':<12}"
              f"{'解密/ms':<12}{'密文/B':<10}")
    print("=" * 100)
    print("Paillier 性能测量：自行实现 vs phe")
    print("=" * 100)
    print(header)
    print("-" * 100)

    rows = []
    for bits in BIT_LENGTHS:
        r_own = measure_own(bits)
        r_phe = measure_phe(bits)
        for name, r in [("自行实现", r_own), ("phe", r_phe)]:
            print(f"{name:<8}{r['bits']:<10}{r['n_bits']:<8}"
                  f"{r['keygen_ms']:<14.2f}{r['encrypt_ms']:<12.3f}"
                  f"{r['add_ms']:<12.3f}{r['decrypt_ms']:<12.3f}"
                  f"{r['ciphertext_bytes']:<10}")
            rows.append((name, r))

    # 保存
    from pathlib import Path
    out = Path(__file__).parent.parent / "results" / "benchmark_paillier.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for name, r in rows:
            f.write(f"{name:<8}{r['bits']:<10}{r['n_bits']:<8}"
                    f"{r['keygen_ms']:<14.2f}{r['encrypt_ms']:<12.3f}"
                    f"{r['add_ms']:<12.3f}{r['decrypt_ms']:<12.3f}"
                    f"{r['ciphertext_bytes']:<10}\n")
    print(f"\n结果已保存到 {out}")


if __name__ == "__main__":
    main()