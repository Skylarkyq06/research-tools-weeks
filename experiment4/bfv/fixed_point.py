"""定点编码对照：K=16/64/255，以及 K=512 范围分析。"""
import json
from pathlib import Path
import numpy as np
import tenseal as ts

from preprocess import load_all, sse, DIM

RESULTS = Path(__file__).parent.parent / "results"

N = 8192
T = 536903681


def encode(x, K):
    """z = x/255, u = rint(K*z)，返回 int64 向量。"""
    return np.rint(K * (x.astype(np.float64) / 255.0)).astype(np.int64)


def encrypted_sse(server_ctx, ctx, x, y):
    cx = ts.bfv_vector_from(server_ctx, x)
    cy = ts.bfv_vector_from(server_ctx, y)
    diff = cx - cy
    sq = diff * diff
    return sq.sum().serialize()


def main():
    gallery, queries, checks = load_all()
    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=N,
                     plain_modulus=T, n_threads=1)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    server_ctx = ts.context_from(ctx.serialize(save_secret_key=False))

    results = {}

    # K=512 范围推导（不做密文）
    true_sse = sse(checks["black"], checks["white"])
    K512_sse = DIM * (512 ** 2)
    print("=" * 72)
    print("K=512 范围分析")
    print("=" * 72)
    print(f"K=512 黑白极值 SSE 上界 = d*K^2 = {K512_sse}")
    print(f"有符号范围 t/2 = {T/2:.1f}")
    print(f"满足恢复条件 = {K512_sse < T/2}")
    results["K512_range"] = {
        "sse_upper": K512_sse, "t_half": T / 2,
        "satisfies": bool(K512_sse < T / 2),
    }

    # K=16/64/255 密文对照
    print("\n" + "=" * 72)
    print("定点编码对照")
    print("=" * 72)
    for K in [16, 64, 255]:
        # 计算明文基准
        base = {}
        for qid in queries:
            base[qid] = sorted(
                [(gid, sse(queries[qid], gallery[gid])) for gid in gallery],
                key=lambda r: (r[1], r[0])
            )

        # 加密后的向量
        enc_gallery = {}
        for gid, v in gallery.items():
            u = encode(v, K)
            enc_gallery[gid] = ts.bfv_vector(ctx, u.tolist()).serialize()
        enc_queries = {}
        for qid, v in queries.items():
            u = encode(v, K)
            enc_queries[qid] = ts.bfv_vector(ctx, u.tolist()).serialize()

        max_int_err = 0
        max_mse_dev = 0.0
        ranking_ok = 0
        total_pairs = 0

        for qid in queries:
            scores = []
            for gid in gallery:
                sb = encrypted_sse(server_ctx, ctx, enc_queries[qid], enc_gallery[gid])
                s = ts.bfv_vector_from(ctx, sb).decrypt()[0]
                scores.append((gid, int(s)))
            scores.sort(key=lambda r: (r[1], r[0]))

            # 整数对照
            for gid, s in scores:
                u = encode(queries[qid], K)
                v = encode(gallery[gid], K)
                d_int = int(np.sum((u - v) ** 2))
                max_int_err = max(max_int_err, abs(s - d_int))
                # 归一化 MSE
                mse_k = s / (DIM * K * K)
                mse_base = base[qid][0][1] / (DIM * 255 * 255)  # rank1 基准
                # 用真实配对比较
                idx = [i for i, (g, _) in enumerate(base[qid]) if g == gid][0]
                mse_ref = base[qid][idx][1] / (DIM * 255 * 255)
                max_mse_dev = max(max_mse_dev, abs(mse_k - mse_ref))
                total_pairs += 1
            if [r[0] for r in scores] == [r[0] for r in base[qid]]:
                ranking_ok += 1

        print(f"\nK={K}:")
        print(f"  整数误差 max = {max_int_err}")
        print(f"  归一化 MSE 最大偏差 = {max_mse_dev:.6f}")
        print(f"  排名一致 = {ranking_ok}/3")

        results[f"K{K}"] = {
            "max_int_err": max_int_err,
            "max_mse_dev": max_mse_dev,
            "ranking_match": f"{ranking_ok}/3",
        }

    out = RESULTS / "fixed_point.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()