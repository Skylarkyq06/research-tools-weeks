"""生成报告插图：原图、Share 预览、恢复对比。"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(Path(__file__).parent))
from shamir import load_share

BASE = Path(__file__).parent.parent
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

img = np.array(Image.open(BASE / 'code' / 'shamir_test_rgb.png'))
rec123 = np.array(Image.open(BASE / 'results' / 'recovered_123.png'))
rec145 = np.array(Image.open(BASE / 'results' / 'recovered_145.png'))
rec_crt = np.array(Image.open(BASE / 'results' / 'recovered_crt_123.png'))

# 图1：原图 + 两组恢复
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, data, title in zip(axes,
                           [img, rec123, rec145],
                           ['原图', 'Share 1,2,3 恢复', 'Share 1,4,5 恢复']):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG / 'recover_compare.pdf')
plt.close(fig)

# 图2：Share 预览（灰度化便于观察）
share_paths = sorted((BASE / 'shares').glob('share_*.npz'))
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for ax, p in zip(axes, share_paths):
    s = load_share(p)
    arr = s['share'].astype(np.uint16)
    # 归一化到 0-255 以便显示
    vis = (arr * 255 // 256).astype(np.uint8)
    ax.imshow(vis)
    ax.set_title(f"Share x={s['x']}")
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG / 'share_preview.pdf')
plt.close(fig)

# 图3：CRT vs Shamir 恢复对比
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, data, title in zip(axes,
                           [img, rec123, rec_crt],
                           ['原图', 'Shamir 恢复', 'CRT 恢复']):
    ax.imshow(data)
    ax.set_title(title)
    ax.axis('off')
fig.tight_layout()
fig.savefig(FIG / 'crt_vs_shamir.pdf')
plt.close(fig)

print('已生成:')
for f in sorted(FIG.glob('*.pdf')):
    print(' ', f.name)