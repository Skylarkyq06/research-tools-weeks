import secrets
import time
from sympy import randprime, mod_inverse
from experiment3.common.channel import Channel


def rsa_keygen(bits=3072):
    half = bits // 2
    p = randprime(2 ** (half - 1), 2 ** half)
    q = randprime(2 ** (half - 1), 2 ** half)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = 65537
    d = mod_inverse(e, phi)
    return n, e, d


def msg_to_int(msg: bytes) -> int:
    return int.from_bytes(msg, "big")


def int_to_msg(x: int, length: int) -> bytes:
    return x.to_bytes(length, "big")


def _nb(n: int) -> int:
    return (n.bit_length() + 7) // 8


def ot_2_1(m0: bytes, m1: bytes, c: int, n: int, e: int, d: int, ch: Channel) -> bytes:
    if c not in (0, 1):
        raise ValueError(f"选择位 c 必须为 0 或 1，收到 {c!r}")
    if len(m0) != len(m1):
        raise ValueError("两条消息长度必须相同")
    if msg_to_int(m0) >= n or msg_to_int(m1) >= n:
        raise ValueError("消息过长，需增大 RSA 模数")

    L = len(m0)
    NB = _nb(n)

    x0 = secrets.randbelow(n - 1) + 1
    x1 = secrets.randbelow(n - 1) + 1
    ch.send("S->R", n.to_bytes(NB, "big") + e.to_bytes(4, "big")
            + x0.to_bytes(NB, "big") + x1.to_bytes(NB, "big"))

    k = secrets.randbelow(n - 1) + 1
    xc = x0 if c == 0 else x1
    v = (xc + pow(k, e, n)) % n
    ch.send("R->S", v.to_bytes(NB, "big"))

    k0 = pow((v - x0) % n, d, n)
    k1 = pow((v - x1) % n, d, n)
    M0 = (msg_to_int(m0) + k0) % n
    M1 = (msg_to_int(m1) + k1) % n
    ch.send("S->R", M0.to_bytes(NB, "big") + M1.to_bytes(NB, "big"))

    Mc = M0 if c == 0 else M1
    return int_to_msg((Mc - k) % n, L)


def ot_4_1(msgs, c: int, n: int, e: int, d: int, ch: Channel) -> bytes:
    if c not in (0, 1, 2, 3):
        raise ValueError(f"选择位 c 必须为 0~3，收到 {c!r}")
    if len(msgs) != 4:
        raise ValueError("需要 4 条消息")
    L = len(msgs[0])
    if any(len(m) != L for m in msgs):
        raise ValueError("4 条消息长度必须相同")
    if any(msg_to_int(m) >= n for m in msgs):
        raise ValueError("消息过长，需增大 RSA 模数")

    NB = _nb(n)
    xs = [secrets.randbelow(n - 1) + 1 for _ in range(4)]
    ch.send("S->R", n.to_bytes(NB, "big") + e.to_bytes(4, "big")
            + b"".join(x.to_bytes(NB, "big") for x in xs))

    k = secrets.randbelow(n - 1) + 1
    v = (xs[c] + pow(k, e, n)) % n
    ch.send("R->S", v.to_bytes(NB, "big"))

    Ms = []
    for i in range(4):
        ki = pow((v - xs[i]) % n, d, n)
        Ms.append((msg_to_int(msgs[i]) + ki) % n)
    ch.send("S->R", b"".join(M.to_bytes(NB, "big") for M in Ms))

    return int_to_msg((Ms[c] - k) % n, L)


if __name__ == "__main__":
    print("生成 3072 位 RSA 密钥……")
    t0 = time.perf_counter()
    n, e, d = rsa_keygen(3072)
    print(f"密钥生成耗时: {time.perf_counter() - t0:.2f} s，模数位数: {n.bit_length()}")

    for c in (0, 1):
        print(f"\n1-out-of-2，c={c}：")
        ok = 0
        for i in range(10):
            m0 = secrets.token_bytes(16)
            m1 = secrets.token_bytes(16)
            ch = Channel()
            got = ot_2_1(m0, m1, c, n, e, d, ch)
            exp = m0 if c == 0 else m1
            passed = got == exp
            ok += passed
            print(f"  {i+1:2d}: {'通过' if passed else '失败'}  字节数={ch.total_bytes}")
        print(f"  通过 {ok}/10")

    m = secrets.token_bytes(16)
    for c in (0, 1):
        ch = Channel()
        got = ot_2_1(m, m, c, n, e, d, ch)
        print(f"相同消息 c={c}: {'通过' if got == m else '失败'}")

    for bad in (-1, 2, 3):
        try:
            ot_2_1(b"a" * 16, b"b" * 16, bad, n, e, d, Channel())
            print(f"非法 c={bad} 未被拒绝")
        except ValueError as ex:
            print(f"非法 c={bad} -> ValueError: {ex}")

    print("\n消息长度对比（固定密钥，重新生成随机数）：")
    for L in (16, 64, 256):
        times, bytes_list = [], []
        for _ in range(10):
            m0 = secrets.token_bytes(L)
            m1 = secrets.token_bytes(L)
            ch = Channel()
            t = time.perf_counter()
            ot_2_1(m0, m1, secrets.choice([0, 1]), n, e, d, ch)
            times.append(time.perf_counter() - t)
            bytes_list.append(ch.total_bytes)
        print(f"  L={L:3d}: 平均耗时={sum(times)/10:.4f} s, 平均字节数={sum(bytes_list)/10:.1f}")

    print("\n4-out-of-1：")
    for c in range(4):
        msgs = [secrets.token_bytes(16) for _ in range(4)]
        ch = Channel()
        got = ot_4_1(msgs, c, n, e, d, ch)
        print(f"  c={c}: {'通过' if got == msgs[c] else '失败'}  字节数={ch.total_bytes}")