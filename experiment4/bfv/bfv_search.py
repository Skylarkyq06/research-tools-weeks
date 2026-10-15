"""BFV 密文图像检索：客户端/服务器角色分离。"""
import json
import time
from pathlib import Path
import numpy as np
import tenseal as ts

from preprocess import load_all, sse, DIM

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

CONFIGS = {
    "A": {"N": 8192, "t": 536903681},
    "B": {"N": 16384, "t": 536903681},
}


def make_context(N, t):
    """客户端上下文：含私钥 + Galois + relin。"""
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=N,
                     plain_modulus=t, n_threads=1)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return ctx


def make_server_context(ctx):
    """服务器上下文：不含私钥。"""
    public_bytes = ctx.serialize(save_secret_key=False)
    server_ctx = ts.context_from(public_bytes)
    assert not server_ctx.has_secret_key(), "服务器上下文不得含私钥"
    return server_ctx


def encrypted_sse(server_ctx, x_bytes, y_bytes):
    """服务器侧：密文相减、逐元素平方、槽位求和、序列化。"""
    if server_ctx.has_secret_key():
        raise ValueError("服务器上下文不得含私钥")
    cx = ts.bfv_vector_from(server_ctx, x_bytes)
    cy = ts.bfv_vector_from(server_ctx, y_bytes)
    diff = cx - cy
    sq = diff * diff
    total = sq.sum()
    return total.serialize()


def run_config(name, N, t, gallery, queries, checks):
    print(f"\n{'='*72}")
    print(f"配置 {name}: N={N}, t={t}")
    print(f"{'='*72}")

    # 1. 上下文生成与评估密钥
    t0 = time.perf_counter()
    ctx = make_context(N, t)
    t1 = time.perf_counter()
    ctx_gen_ms = (t1 - t0) * 1000

    q_bits = ctx.seal_context().data.key_context_data().total_coeff_modulus_bit_count()
    print(f"系数模数总位数 = {q_bits}, 上下文生成 {ctx_gen_ms:.1f} ms")

    # 2. 上下文序列化与服务器恢复
    t0 = time.perf_counter()
    server_ctx = make_server_context(ctx)
    t1 = time.perf_counter()
    ctx_restore_ms = (t1 - t0) * 1000
    print(f"上下文序列化+恢复 {ctx_restore_ms:.1f} ms, 服务器无私钥 = {not server_ctx.has_secret_key()}")

    # 3. 客户端加密图库和查询
    t0 = time.perf_counter()
    enc_gallery = {}
    for gid, v in gallery.items():
        enc_gallery[gid] = ts.bfv_vector(ctx, v.tolist()).serialize()
    enc_queries = {}
    for qid, v in queries.items():
        enc_queries[qid] = ts.bfv_vector(ctx, v.tolist()).serialize()
    t1 = time.perf_counter()
    enc_ms = (t1 - t0) * 1000
    print(f"图像加密（6+3）{enc_ms:.1f} ms")

    # 4. 图像加解密一致性
    dec_g = ts.bfv_vector_from(ctx, enc_gallery["G01"]).decrypt()
    ok_pix = (np.array(dec_g[:DIM], dtype=np.int64) == gallery["G01"]).all()
    print(f"G01 加解密 4096 像素一致 = {ok_pix}")

    # 5. 检索：每查询遍历 6 图库
    all_correct = True
    all_ranking_match = True
    max_err = 0
    per_query_ms = {}

    for qid in queries:
        t0 = time.perf_counter()
        scores = []
        for gid in gallery:
            score_bytes = encrypted_sse(server_ctx, enc_queries[qid], enc_gallery[gid])
            score = ts.bfv_vector_from(ctx, score_bytes).decrypt()[0]
            scores.append((gid, score))
        t1 = time.perf_counter()
        per_query_ms[qid] = (t1 - t0) * 1000

        # 排名
        scores.sort(key=lambda r: (r[1], r[0]))
        # 明文对照
        plain = sorted([(gid, sse(queries[qid], gallery[gid])) for gid in gallery],
                       key=lambda r: (r[1], r[0]))
        for (gid_c, s_c), (gid_p, s_p) in zip(scores, plain):
            err = abs(s_c - s_p)
            max_err = max(max_err, err)
            if err != 0:
                all_correct = False
        if [r[0] for r in scores] != [r[0] for r in plain]:
            all_ranking_match = False

        print(f"  {qid}: 遍历 {per_query_ms[qid]:.1f} ms, "
              f"rank1={scores[0][0]} SSE={scores[0][1]} "
              f"(明文 {plain[0][1]})")

    # 6. 极值检查
    black = checks["black"].tolist()
    white = checks["white"].tolist()
    c_b = ts.bfv_vector(ctx, black).serialize()
    c_w = ts.bfv_vector(ctx, white).serialize()
    sb = encrypted_sse(server_ctx, c_b, c_w)
    s_bw = ts.bfv_vector_from(ctx, sb).decrypt()[0]
    print(f"black vs white 密文 SSE = {s_bw} (期望 266342400)")

    return {
        "config": name, "N": N, "t": t, "q_bits": q_bits,
        "ctx_gen_ms": ctx_gen_ms, "ctx_restore_ms": ctx_restore_ms,
        "enc_ms": enc_ms, "per_query_ms": per_query_ms,
        "max_err": max_err, "all_correct": all_correct,
        "ranking_match": all_ranking_match, "black_white": int(s_bw),
        "server_no_secret": not server_ctx.has_secret_key(),
    }


def main():
    gallery, queries, checks = load_all()
    all_results = []
    for name, cfg in CONFIGS.items():
        r = run_config(name, cfg["N"], cfg["t"], gallery, queries, checks)
        all_results.append(r)

    out = RESULTS / "bfv_results.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()