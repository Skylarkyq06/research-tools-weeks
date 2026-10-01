"""CRT 进阶方案驱动。"""
import time
import sys
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from crt_scheme import split_image_crt, recover_image_crt, M_LIST

BASE = Path(__file__).parent.parent
IMG = BASE / 'code' / 'shamir_test_rgb.png'
RESULTS_DIR = BASE / 'results'

img = np.array(Image.open(IMG))
n, k = 5, 3

t0 = time.perf_counter()
shares = split_image_crt(img, n=n, k=k)
t1 = time.perf_counter()
print(f'CRT 分片耗时: {(t1-t0)*1000:.2f} ms')

t0 = time.perf_counter()
rec = recover_image_crt([shares[0], shares[1], shares[2]], xs=[0, 1, 2], k=k)
t1 = time.perf_counter()
print(f'CRT 恢复 1,2,3 耗时: {(t1-t0)*1000:.2f} ms')
Image.fromarray(rec).save(RESULTS_DIR / 'recovered_crt_123.png')
err = int(np.abs(rec.astype(np.int16) - img.astype(np.int16)).max())
print(f'CRT 恢复最大绝对误差: {err}')

try:
    recover_image_crt([shares[0], shares[1]], xs=[0, 1], k=k)
    print('错误: 应该拒绝但没拒绝')
except ValueError as e:
    print(f'CRT Share 不足时正确拒绝: {e}')