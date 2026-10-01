"""Shamir 参数与开销分析。"""
import time
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from shamir import split_image, recover_image, save_share, P

BASE = Path(__file__).parent.parent
IMG = BASE / 'code' / 'shamir_test_rgb.png'
SHARES_DIR = BASE / 'shares'
RESULTS_DIR = BASE / 'results'

img = np.array(Image.open(IMG))

configs = [
    (5, 3, [1, 2, 3, 4, 5]),
    (7, 5, [1, 2, 3, 4, 5, 6, 7]),
    (4, 2, [1, 2, 3, 4]),
]

lines = []
for n, k, xs in configs:
    # 分片
    t0 = time.perf_counter()
    shares = split_image(img, n=n, k=k, xs=xs)
    t1 = time.perf_counter()
    split_ms = (t1 - t0) * 1000

    # 保存并算大小
    for i in range(n):
        save_share(SHARES_DIR / f'bench_n{n}k{k}_share_{i+1:02d}.npz',
                   xs[i], shares[i], P, n, k, 'RGB')
    sizes = [(SHARES_DIR / f'bench_n{n}k{k}_share_{i+1:02d}.npz').stat().st_size
             for i in range(n)]

    # 恢复前 k 份
    t0 = time.perf_counter()
    rec = recover_image(shares[:k], xs=xs[:k], k=k)
    t1 = time.perf_counter()
    recover_ms = (t1 - t0) * 1000

    err = int(np.abs(rec.astype(np.int16) - img.astype(np.int16)).max())

    lines.append(
        f'n={n}, k={k}: 分片 {split_ms:.2f} ms, 恢复 {recover_ms:.2f} ms, '
        f'单份 {sizes[0]} B, 总 {sum(sizes)} B, 最大误差 {err}'
    )

out = RESULTS_DIR / 'benchmark.txt'
out.write_text('\n'.join(lines), encoding='utf-8')
print('\n'.join(lines))