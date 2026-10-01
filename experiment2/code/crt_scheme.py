"""Asmuth-Bloom CRT 门限秘密共享（进阶方案）。
参数：n=5, k=3, m_0=257, m_1..m_5 两两互素。
满足 m_1*m_2*m_3 > m_0 * m_4 * m_5。
"""
import secrets
import numpy as np

M0 = 257
M_LIST = [521, 523, 541, 547, 557]


def _check():
    n, k = len(M_LIST), 3
    p1 = 1
    for i in range(k):
        p1 *= M_LIST[i]
    p2 = M0
    for i in range(n - k + 1, n):
        p2 *= M_LIST[i]
    assert p1 > p2, f"Asmuth-Bloom 条件不满足: {p1} <= {p2}"


_check()


def crt(residues, moduli):
    M = 1
    for m in moduli:
        M *= m
    x = 0
    for r, m in zip(residues, moduli):
        Mi = M // m
        inv = pow(Mi, -1, m)
        x = (x + r * Mi * inv) % M
    return x


def split_image_crt(img_array, n=5, k=3):
    _check()
    if img_array.dtype != np.uint8:
        raise ValueError("图像类型须为 uint8")

    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1
    flat = img_array.reshape(-1, channels) if img_array.ndim == 3 else img_array.reshape(-1, 1)

    prod_k = 1
    for i in range(k):
        prod_k *= M_LIST[i]

    shares = np.zeros((n, flat.shape[0], channels), dtype=np.uint16)
    for idx in range(flat.shape[0]):
        for c in range(channels):
            s = int(flat[idx, c])
            max_r = (prod_k - 1 - s) // M0
            r = secrets.randbelow(max_r + 1)
            y = s + M0 * r
            for i in range(n):
                shares[i, idx, c] = y % M_LIST[i]

    if img_array.ndim == 3:
        shares = shares.reshape(n, h, w, channels)
    else:
        shares = shares.reshape(n, h, w)
    return shares


def recover_image_crt(share_arrays, xs, k=3):
    m = len(share_arrays)
    if m < k:
        raise ValueError(f"Share 数量不足，需要至少 {k} 份，实际 {m} 份")

    shares_k = share_arrays[:k]
    moduli_k = [M_LIST[x] for x in xs[:k]]

    shape = shares_k[0].shape
    h, w = shape[:2]
    channels = shape[2] if len(shape) == 3 else 1
    flat = np.stack([a.reshape(-1, channels) if len(shape) == 3 else a.reshape(-1, 1)
                     for a in shares_k], axis=0)

    recovered = np.zeros((flat.shape[1], channels), dtype=np.uint8)
    for idx in range(flat.shape[1]):
        for c in range(channels):
            residues = [int(flat[i, idx, c]) for i in range(k)]
            y = crt(residues, moduli_k)
            recovered[idx, c] = y % M0

    if len(shape) == 3:
        recovered = recovered.reshape(h, w, channels)
    else:
        recovered = recovered.reshape(h, w)
    return recovered