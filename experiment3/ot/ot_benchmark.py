import csv
import secrets
import time
from pathlib import Path

from experiment3.common.channel import Channel
from experiment3.ot.ot_rsa import rsa_keygen, ot_2_1, ot_4_1

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RSA_BITS = 3072
MSG_LEN = 16
RUNS = 10
LENGTHS = [16, 64, 256]


def run_ot_benchmark():
    print(f"生成 {RSA_BITS} 位 RSA 密钥（固定，供所有测试复用）……")
    t_key0 = time.perf_counter()
    n, e, d = rsa_keygen(RSA_BITS)
    t_key = time.perf_counter() - t_key0
    print(f"密钥生成耗时: {t_key:.4f} s，模数位数: {n.bit_length()}\n")

    # ---------- 1-out-of-2：c=0 / c=1 各 10 组 ----------
    ot_rows = []
    for c in (0, 1):
        for i in range(RUNS):
            m0 = secrets.token_bytes(MSG_LEN)
            m1 = secrets.token_bytes(MSG_LEN)
            ch = Channel()
            t0 = time.perf_counter()
            got = ot_2_1(m0, m1, c, n, e, d, ch)
            t = time.perf_counter() - t0
            expected = m0 if c == 0 else m1
            ot_rows.append({
                "c": c,
                "run_id": i + 1,
                "msg_len": MSG_LEN,
                "correct": got == expected,
                "time_s": f"{t:.6f}",
                "bytes": ch.total_bytes,
            })

    # ---------- 消息长度对比：16 / 64 / 256，各 10 次 ----------
    length_rows = []
    for L in LENGTHS:
        for i in range(RUNS):
            m0 = secrets.token_bytes(L)
            m1 = secrets.token_bytes(L)
            c = secrets.choice([0, 1])
            ch = Channel()
            t0 = time.perf_counter()
            got = ot_2_1(m0, m1, c, n, e, d, ch)
            t = time.perf_counter() - t0
            length_rows.append({
                "msg_len": L,
                "run_id": i + 1,
                "c": c,
                "correct": got == (m0 if c == 0 else m1),
                "time_s": f"{t:.6f}",
                "bytes": ch.total_bytes,
            })

    # ---------- 4-out-of-1：c=0..3 ----------
    four_rows = []
    for c in range(4):
        msgs = [secrets.token_bytes(MSG_LEN) for _ in range(4)]
        ch = Channel()
        t0 = time.perf_counter()
        got = ot_4_1(msgs, c, n, e, d, ch)
        t = time.perf_counter() - t0
        four_rows.append({
            "c": c,
            "msg_len": MSG_LEN,
            "correct": got == msgs[c],
            "time_s": f"{t:.6f}",
            "bytes": ch.total_bytes,
        })

    # ---------- 写 CSV ----------
    ot_path = RESULTS_DIR / "ot_2_1.csv"
    with ot_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ot_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ot_rows)

    length_path = RESULTS_DIR / "ot_length.csv"
    with length_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(length_rows[0].keys()))
        writer.writeheader()
        writer.writerows(length_rows)

    four_path = RESULTS_DIR / "ot_4_1.csv"
    with four_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(four_rows[0].keys()))
        writer.writeheader()
        writer.writerows(four_rows)

    # ---------- 汇总打印 ----------
    print(f"1-out-of-2 数据: {ot_path}")
    print(f"消息长度对比数据: {length_path}")
    print(f"4-out-of-1 数据: {four_path}\n")

    for c in (0, 1):
        rows_c = [r for r in ot_rows if r["c"] == c]
        avg_t = sum(float(r["time_s"]) for r in rows_c) / len(rows_c)
        avg_b = sum(r["bytes"] for r in rows_c) / len(rows_c)
        ok = sum(r["correct"] for r in rows_c)
        print(f"c={c}: 通过 {ok}/{len(rows_c)}，平均耗时 {avg_t:.4f} s，平均字节数 {avg_b:.1f}")

    print("\n消息长度对比：")
    print(f"{'长度':<8}{'通过':<8}{'平均耗时(s)':<16}{'平均字节数':<12}")
    for L in LENGTHS:
        rows_L = [r for r in length_rows if r["msg_len"] == L]
        avg_t = sum(float(r["time_s"]) for r in rows_L) / len(rows_L)
        avg_b = sum(r["bytes"] for r in rows_L) / len(rows_L)
        ok = sum(r["correct"] for r in rows_L)
        print(f"{L:<8}{ok}/{len(rows_L):<6}{avg_t:<16.4f}{avg_b:<12.1f}")

    print("\n4-out-of-1：")
    for r in four_rows:
        print(f"c={r['c']}: {'通过' if r['correct'] else '失败'}，耗时 {float(r['time_s']):.4f} s，字节数 {r['bytes']}")


if __name__ == "__main__":
    run_ot_benchmark()