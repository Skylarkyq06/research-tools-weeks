"""Shamir 门限秘密共享（GF(257)），图像逐像素逐通道分片与恢复。"""
import secrets
import numpy as np

P = 257


def mod_inv(a, p=P):
    a = a % p
    if a == 0:
        raise ZeroDivisionError("0 在模 p 下没有逆元")
    return pow(a, p - 2, p)  # p 是素数，用费马小定理


def poly_eval(coeffs, x, p=P):
    result = 0
    for c in reversed(coeffs):
        result = (result * x + c) % p
    return result


def lagrange_at_zero(xs, ys, p=P):
    k = len(xs)
    if k != len(ys):
        raise ValueError("横坐标与 Share 数量不一致")
    if len(set(xs)) != k:
        raise ValueError("横坐标必须互不相同")
    secret = 0
    for i in range(k):
        num, den = 1, 1
        for j in range(k):
            if i == j:
                continue
            num = (num * (-xs[j])) % p
            den = (den * (xs[i] - xs[j])) % p
        secret = (secret + ys[i] * num * mod_inv(den, p)) % p
    return secret


def split_image(img_array, n, k, xs, p=P):
    if not (2 <= k <= n <= p - 1):
        raise ValueError(f"参数不满足 2 <= k <= n <= p-1: k={k}, n={n}")
    if len(xs) != n:
        raise ValueError(f"横坐标数量应为 {n}，实际 {len(xs)}")
    if len(set(xs)) != n:
        raise ValueError("横坐标必须互不相同")
    if any(x <= 0 or x >= p for x in xs):
        raise ValueError(f"横坐标须在 1 到 {p-1}")
    if img_array.dtype != np.uint8:
        raise ValueError(f"图像类型须为 uint8，实际 {img_array.dtype}")

    h, w = img_array.shape[:2]
    channels = img_array.shape[2] if img_array.ndim == 3 else 1
    flat = img_array.reshape(-1, channels) if img_array.ndim == 3 else img_array.reshape(-1, 1)

    shares = np.zeros((n, flat.shape[0], channels), dtype=np.uint16)

    for idx in range(flat.shape[0]):
        for c in range(channels):
            s = int(flat[idx, c])
            coeffs = [s] + [secrets.randbelow(p) for _ in range(k - 1)]
            for i, x in enumerate(xs):
                shares[i, idx, c] = poly_eval(coeffs, x, p)

    if img_array.ndim == 3:
        shares = shares.reshape(n, h, w, channels)
    else:
        shares = shares.reshape(n, h, w)
    return shares


def recover_image(share_arrays, xs, k, p=P):
    m = len(share_arrays)
    if m < k:
        raise ValueError(f"Share 数量不足，需要至少 {k} 份，实际 {m} 份")
    if len(xs) != m:
        raise ValueError("横坐标数量与 Share 数量不一致")
    if len(set(xs)) != m:
        raise ValueError("横坐标必须互不相同")

    xs_k = xs[:k]
    shares_k = share_arrays[:k]

    shape = shares_k[0].shape
    for arr in shares_k[1:]:
        if arr.shape != shape:
            raise ValueError("Share 数组尺寸不一致")

    h, w = shape[:2]
    channels = shape[2] if len(shape) == 3 else 1
    flat = np.stack([a.reshape(-1, channels) if len(shape) == 3 else a.reshape(-1, 1)
                     for a in shares_k], axis=0)

    recovered = np.zeros((flat.shape[1], channels), dtype=np.uint8)
    for idx in range(flat.shape[1]):
        for c in range(channels):
            ys = [int(flat[i, idx, c]) for i in range(k)]
            recovered[idx, c] = lagrange_at_zero(xs_k, ys, p)

    if len(shape) == 3:
        recovered = recovered.reshape(h, w, channels)
    else:
        recovered = recovered.reshape(h, w)
    return recovered


def save_share(path, x, share_array, p, n, k, mode):
    np.savez(path, x=x, share=share_array, p=p, n=n, k=k, mode=mode)


def load_share(path):
    data = np.load(path, allow_pickle=False)
    return {
        'x': int(data['x']),
        'share': data['share'],
        'p': int(data['p']),
        'n': int(data['n']),
        'k': int(data['k']),
        'mode': str(data['mode']),
    }