"""四维手算验证：a=(1,2,3,4), b=(4,3,2,1)。"""
import numpy as np

a = np.array([1, 2, 3, 4], dtype=np.int64)
b = np.array([4, 3, 2, 1], dtype=np.int64)

diff = a - b
sq = diff * diff
sse = int(sq.sum())

print("a =", a.tolist())
print("b =", b.tolist())
print("a - b =", diff.tolist())
print("(a - b)^2 =", sq.tolist())
print("SSE =", sse)
print("期望 SSE = 10")
assert sse == 20, f"手算错误: {sse}"
print("手算正确")