"""图像预处理：统一转为 L 模式，缩放 64x64，按行展开为 4096 维向量。"""
from pathlib import Path
import numpy as np
from PIL import Image

DATA = Path(__file__).parent.parent / "data"
GALLERY = DATA / "gallery"
QUERIES = DATA / "queries"
CHECKS = DATA / "checks"

SIZE = (64, 64)
DIM = SIZE[0] * SIZE[1]


def preprocess(path):
    """读取图像 -> L 模式 -> 缩放到 64x64 -> 行优先展平为 int64 向量。"""
    img = Image.open(path).convert("L").resize(SIZE, Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    return arr.reshape(-1).astype(np.int64)


def load_all():
    gallery = {}
    for i in range(1, 7):
        gid = f"G{i:02d}"
        gallery[gid] = preprocess(GALLERY / f"{gid}.png")
    queries = {}
    for qid, name in [("Q01", "Q01_exact"), ("Q02", "Q02_perturbed"), ("Q03", "Q03_unseen")]:
        queries[qid] = preprocess(QUERIES / f"{name}.png")
    checks = {}
    for cid in ["black", "white"]:
        checks[cid] = preprocess(CHECKS / f"{cid}.png")
    return gallery, queries, checks


def sse(x, y):
    """平方差之和：先转 int64 再相减，避免 uint8 溢出。"""
    d = x.astype(np.int64) - y.astype(np.int64)
    return int(np.sum(d * d))


if __name__ == "__main__":
    gallery, queries, checks = load_all()
    print(f"维度 = {DIM}")
    for gid, v in gallery.items():
        print(f"  {gid}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
    for qid, v in queries.items():
        print(f"  {qid}: shape={v.shape}, min={v.min()}, max={v.max()}")
    for cid, v in checks.items():
        print(f"  check {cid}: min={v.min()}, max={v.max()}")
    # 黑-白距离
    black, white = checks["black"], checks["white"]
    print(f"\nblack vs white SSE = {sse(black, white)}")
    print(f"期望 266342400")