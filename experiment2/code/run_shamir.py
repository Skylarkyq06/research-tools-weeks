"""实验二 Shamir 基准实验驱动。"""
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
SHARES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

img = np.array(Image.open(IMG))
print(f'原图: shape={img.shape}, dtype={img.dtype}')

n, k = 5, 3
xs = [1, 2, 3, 4, 5]

t0 = time.perf_counter()
shares = split_image(img, n=n, k=k, xs=xs)
t1 = time.perf_counter()
print(f'分片耗时: {(t1-t0)*1000:.2f} ms')

for i in range(n):
    save_share(SHARES_DIR / f'share_{i+1:02d}.npz', xs[i], shares[i], P, n, k, 'RGB')
print('已保存 5 份 Share')

# 恢复 1,2,3
t0 = time.perf_counter()
rec123 = recover_image([shares[0], shares[1], shares[2]], xs=[1,2,3], k=k)
t1 = time.perf_counter()
print(f'恢复 1,2,3 耗时: {(t1-t0)*1000:.2f} ms')
Image.fromarray(rec123).save(RESULTS_DIR / 'recovered_123.png')
err123 = int(np.abs(rec123.astype(np.int16) - img.astype(np.int16)).max())
print(f'1,2,3 恢复最大绝对误差: {err123}')

# 恢复 1,4,5
t0 = time.perf_counter()
rec145 = recover_image([shares[0], shares[3], shares[4]], xs=[1,4,5], k=k)
t1 = time.perf_counter()
print(f'恢复 1,4,5 耗时: {(t1-t0)*1000:.2f} ms')
Image.fromarray(rec145).save(RESULTS_DIR / 'recovered_145.png')
err145 = int(np.abs(rec145.astype(np.int16) - img.astype(np.int16)).max())
print(f'1,4,5 恢复最大绝对误差: {err145}')

# 不足 k 份，拒绝恢复
try:
    recover_image([shares[0], shares[1]], xs=[1,2], k=k)
    print('错误: 应该拒绝但没拒绝')
except ValueError as e:
    print(f'Share 不足时正确拒绝: {e}')