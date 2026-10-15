"""Paillier 部分同态加密的独立实现。
接口：keygen / encrypt / decrypt / add / mul_plain
素数生成使用 Miller-Rabin 素性检测，不使用第三方密码学库。
"""
import secrets
import math
import time
from dataclasses import dataclass

_SMALL_PRIMES = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67,
    71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
    317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
    419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499,
    503, 509, 521, 523, 541,
]


@dataclass
class PaillierPublicKey:
    n: int
    g: int
    n_sq: int


@dataclass
class PaillierPrivateKey:
    lam: int
    mu: int
    n: int
    n_sq: int


def _is_probable_prime(n, rounds=30):
    """Miller-Rabin 素性检测：先用小底快速筛选，再随机 rounds 轮。"""
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    def _miller(a):
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                return True
        return False

    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if not _miller(a):
            return False
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        if not _miller(a):
            return False
    return True


def _gen_prime(bits):
    """生成 bits 位素数，最高位和最低位强制为 1。"""
    while True:
        cand = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(cand):
            return cand


def keygen(bits=1024):
    """生成 Paillier 密钥对，返回 (pub, priv)。"""
    if bits < 16:
        raise ValueError("密钥位数过小")
    half = bits // 2
    while True:
        p = _gen_prime(half)
        q = _gen_prime(half)
        if p == q:
            continue
        n = p * q
        if n.bit_length() != bits:
            continue
        if math.gcd(n, (p - 1) * (q - 1)) != 1:
            continue
        break
    g = n + 1
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
    n_sq = n * n
    u = pow(g, lam, n_sq)
    l_val = (u - 1) // n
    mu = pow(l_val, -1, n)
    pub = PaillierPublicKey(n=n, g=g, n_sq=n_sq)
    priv = PaillierPrivateKey(lam=lam, mu=mu, n=n, n_sq=n_sq)
    return pub, priv


def encrypt(pub: PaillierPublicKey, m: int) -> int:
    """加密明文 m，要求 0 <= m < n。"""
    if not isinstance(m, int):
        raise TypeError("明文必须为整数")
    if m < 0 or m >= pub.n:
        raise ValueError(f"明文须在 [0, {pub.n}) 范围内，收到 {m}")
    while True:
        r = secrets.randbelow(pub.n - 1) + 1
        if math.gcd(r, pub.n) == 1:
            break
    c = (pow(pub.g, m, pub.n_sq) * pow(r, pub.n, pub.n_sq)) % pub.n_sq
    return c


def decrypt(priv: PaillierPrivateKey, c: int) -> int:
    u = pow(c, priv.lam, priv.n_sq)
    l_val = (u - 1) // priv.n
    return (l_val * priv.mu) % priv.n


def add(pub: PaillierPublicKey, c1: int, c2: int) -> int:
    return (c1 * c2) % pub.n_sq


def mul_plain(pub: PaillierPublicKey, c: int, k: int) -> int:
    if k < 0:
        raise ValueError("标量须为非负整数")
    if k == 0:
        return encrypt(pub, 0)  # c^0 = 1 不代表有效密文，返回加密的 0
    return pow(c, k, pub.n_sq)


def ciphertext_bytes(c: int) -> int:
    return (c.bit_length() + 7) // 8