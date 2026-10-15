"""BFV 性能测量：5 轮预热计时 + 密文/上下文/得分字节大小。"""
import json
import time
import statistics
from pathlib import Path
import tenseal as ts

from preprocess import load_all, DIM

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

CONFIGS = {
    "A": {"N": 8192, "t": 536903681},
    "B": {"N": 16384, "t": 536903681},
}

ROUNDS = 5
WARMUP = 1


def median(xs):
    return statistics.median(xs)


def run(name, N, t, gallery, queries):
    print(f"\n{'='*72}")
    print(f"配置 {name}: N={N}, t={t}")
    print(f"{'='*72}")

    # 1. 上下文与评估密钥生成（一次性，不取中位数）
    t0 = time.perf_counter()
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=N,
                     plain_modulus=t, n_threads=1)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    t1 = time.perf_counter()
    ctx_gen_ms = (t1 - t0) * 1000

    # 2. 上下文序列化与服务器恢复
    t0 = time.perf_counter()
    public_bytes = ctx.serialize(save_secret_key=False)
    t1 = time.perf_counter()
    ser_ctx_ms = (t1 - t0) * 1000
    ctx_bytes = len(public_bytes)

    t0 = time.perf_counter()
    server_ctx = ts.context_from(public_bytes)
    t1 = time.perf_counter()
    restore_ctx_ms = (t1 - t0) * 1000
    assert not server_ctx.has_secret_key()

    print(f"上下文生成 = {ctx_gen_ms:.1f} ms")
    print(f"上下文序列化 = {ser_ctx_ms:.1f} ms, 大小 = {ctx_bytes} B "
          f"({ctx_bytes/1024/1024:.2f} MiB)")
    print(f"服务器上下文恢复 = {restore_ctx_ms:.1f} ms")

    # 3. 图像加密（预热 + 5 轮）
    for _ in range(WARMUP):
        for gid, v in gallery.items():
            _ = ts.bfv_vector(ctx, v.tolist()).serialize()
        for qid, v in queries.items():
            _ = ts.bfv_vector(ctx, v.tolist()).serialize()

    enc_rounds = []
    img_cipher_bytes = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for gid, v in gallery.items():
            c = ts.bfv_vector(ctx, v.tolist()).serialize()
            img_cipher_bytes.append(len(c))
        for qid, v in queries.items():
            c = ts.bfv_vector(ctx, v.tolist()).serialize()
            img_cipher_bytes.append(len(c))
        t1 = time.perf_counter()
        enc_rounds.append((t1 - t0) * 1000 / 9)
    enc_ms = median(enc_rounds)

    # 4. 单对求值：反序列化 + 密文运算 + 得分序列化
    qid = "Q01"
    gid = "G01"
    enc_q = ts.bfv_vector(ctx, queries[qid].tolist()).serialize()
    enc_g = ts.bfv_vector(ctx, gallery[gid].tolist()).serialize()

    # 预热
    for _ in range(WARMUP):
        cq = ts.bfv_vector_from(server_ctx, enc_q)
        cg = ts.bfv_vector_from(server_ctx, enc_g)
        diff = cq - cg
        sq = diff * diff
        score = sq.sum()
        _ = score.serialize()

    deser_rounds, eval_rounds, ser_score_rounds = [], [], []
    score_bytes_list = []
    for _ in range(ROUNDS):
        # 反序列化
        t0 = time.perf_counter()
        cq = ts.bfv_vector_from(server_ctx, enc_q)
        cg = ts.bfv_vector_from(server_ctx, enc_g)
        t1 = time.perf_counter()
        deser_rounds.append((t1 - t0) * 1000)

        # 核心求值：相减、平方、槽位求和
        t0 = time.perf_counter()
        diff = cq - cg
        sq = diff * diff
        score = sq.sum()
        t1 = time.perf_counter()
        eval_rounds.append((t1 - t0) * 1000)

        # 得分序列化
        t0 = time.perf_counter()
        sb = score.serialize()
        t1 = time.perf_counter()
        ser_score_rounds.append((t1 - t0) * 1000)
        score_bytes_list.append(len(sb))

    deser_ms = median(deser_rounds)
    eval_ms = median(eval_rounds)
    ser_score_ms = median(ser_score_rounds)
    img_cipher_bytes_med = int(median(img_cipher_bytes))
    score_bytes_med = int(median(score_bytes_list))

    print(f"图像加密（单张，5 轮中位）= {enc_ms:.3f} ms")
    print(f"图像密文大小（中位）= {img_cipher_bytes_med} B")
    print(f"反序列化（单对，5 轮中位）= {deser_ms:.3f} ms")
    print(f"密文求值（单对，5 轮中位）= {eval_ms:.3f} ms")
    print(f"得分序列化（5 轮中位）= {ser_score_ms:.3f} ms")
    print(f"得分密文大小 = {score_bytes_med} B")

    # 5. 单查询遍历（含加解密 + 6 张图库）
    traverse_rounds = []
    for _ in range(WARMUP):
        for g2 in gallery:
            cq = ts.bfv_vector_from(server_ctx, enc_q)
            cg = ts.bfv_vector_from(server_ctx,
                                    ts.bfv_vector(ctx, gallery[g2].tolist()).serialize())
            _ = (cq - cg) * (cq - cg)
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for g2 in gallery:
            cg = ts.bfv_vector_from(server_ctx,
                                    ts.bfv_vector(ctx, gallery[g2].tolist()).serialize())
            diff = cq - cg
            sq = diff * diff
            _ = sq.sum().serialize()
        t1 = time.perf_counter()
        traverse_rounds.append((t1 - t0) * 1000)
    traverse_ms = median(traverse_rounds)
    print(f"单查询遍历（6 图库，5 轮中位）= {traverse_ms:.1f} ms")

    return {
        "config": name, "N": N, "t": t,
        "ctx_gen_ms": ctx_gen_ms,
        "ctx_ser_ms": ser_ctx_ms,
        "ctx_restore_ms": restore_ctx_ms,
        "ctx_bytes": ctx_bytes,
        "enc_ms_per_img": enc_ms,
        "img_cipher_bytes": img_cipher_bytes_med,
        "deser_ms": deser_ms,
        "eval_ms": eval_ms,
        "score_ser_ms": ser_score_ms,
        "score_bytes": score_bytes_med,
        "traverse_ms": traverse_ms,
    }


def main():
    gallery, queries, _ = load_all()
    results = []
    for name, cfg in CONFIGS.items():
        r = run(name, cfg["N"], cfg["t"], gallery, queries)
        results.append(r)

    out = RESULTS / "bfv_timing.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()