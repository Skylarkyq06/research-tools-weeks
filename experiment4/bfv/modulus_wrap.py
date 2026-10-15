"""模数边界对照：t=65537，黑白极值输入，展示明文回绕。"""
import json
from pathlib import Path
import tenseal as ts
import numpy as np

from preprocess import load_all, sse, DIM

RESULTS = Path(__file__).parent.parent / "results"

N = 8192
T = 65537  # 故意选小模数


def signed_repr(r, t):
    """有符号代表值：0 <= r <= (t-1)/2 取 r，否则 r - t。"""
    if 0 <= r <= (t - 1) // 2:
        return r
    return r - t


def main():
    gallery, queries, checks = load_all()
    black = checks["black"].tolist()
    white = checks["white"].tolist()

    # 真实距离
    true_sse = sse(checks["black"], checks["white"])
    r_mod = true_sse % T
    signed = signed_repr(r_mod, T)

    print("=" * 72)
    print(f"模数边界对照: N={N}, t={T}")
    print("=" * 72)
    print(f"真实 SSE = {true_sse}")
    print(f"真实 SSE mod t = {r_mod}")
    print(f"有符号代表值 = {signed}")

    ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=N,
                     plain_modulus=T, n_threads=1)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()

    public_bytes = ctx.serialize(save_secret_key=False)
    server_ctx = ts.context_from(public_bytes)
    assert not server_ctx.has_secret_key()

    c_b = ts.bfv_vector(ctx, black).serialize()
    c_w = ts.bfv_vector(ctx, white).serialize()

    cx = ts.bfv_vector_from(server_ctx, c_b)
    cy = ts.bfv_vector_from(server_ctx, c_w)
    diff = cx - cy
    sq = diff * diff
    total = sq.sum()
    score_bytes = total.serialize()
    # 客户端用带私钥的 ctx 解密
    decrypted = ts.bfv_vector_from(ctx, score_bytes).decrypt()[0]

    print(f"密文解密结果 = {decrypted}")
    print(f"与有符号代表值一致 = {decrypted == signed}")
    print(f"与真实距离一致 = {decrypted == true_sse}")

    out = RESULTS / "modulus_wrap.json"
    out.write_text(json.dumps({
        "N": N, "t": T, "true_sse": true_sse,
        "r_mod_t": r_mod, "signed_value": signed,
        "decrypted": int(decrypted),
        "match_signed": decrypted == signed,
        "match_true": decrypted == true_sse,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()