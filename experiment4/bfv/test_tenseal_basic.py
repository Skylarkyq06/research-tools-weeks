"""TenSEAL BFV 基础接口测试。"""
import tenseal as ts

N = 8192
T = 536903681

ctx = ts.context(ts.SCHEME_TYPE.BFV, poly_modulus_degree=N, plain_modulus=T, n_threads=1)
ctx.generate_galois_keys()
ctx.generate_relin_keys()

q_bits = ctx.seal_context().data.key_context_data().total_coeff_modulus_bit_count()
print(f"N={N}, t={T}, 系数模数总位数={q_bits}")

# 小向量测试
a = [1, 2, 3, 4]
b = [4, 3, 2, 1]
ct_a = ts.bfv_vector(ctx, a)
ct_b = ts.bfv_vector(ctx, b)

diff = ct_a - ct_b
sq = diff * diff
total = sq.sum()
print(f"a-b = {diff.decrypt()}")
print(f"(a-b)^2 = {sq.decrypt()}")
print(f"SSE = {total.decrypt()[0]}  (期望 20)")

# 4096 维测试
import numpy as np
x = np.arange(4096, dtype=np.int64) % 256
y = (np.arange(4096, dtype=np.int64) + 10) % 256
ct_x = ts.bfv_vector(ctx, x.tolist())
ct_y = ts.bfv_vector(ctx, y.tolist())
diff = ct_x - ct_y
sq = diff * diff
s = sq.sum()
d = s.decrypt()[0]
exp = int(np.sum((x - y) ** 2))
print(f"4096 维 SSE = {d}, 期望 = {exp}, 通过 = {d == exp}")