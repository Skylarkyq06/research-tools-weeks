import time
import random
from experiment3.common.channel import Channel
from experiment4.paillier.paillier import keygen, encrypt, decrypt


def _get_attr(obj, names):
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    raise AttributeError(f"对象 {type(obj).__name__} 缺少属性 {names}")


def _get_n(pub):
    return _get_attr(pub, ("n", "N", "modulus"))


def _nb(n: int) -> int:
    return (n.bit_length() + 7) // 8


def validate_wealth(value, name="wealth", max_val=255):
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} 必须为整数，收到 {type(value).__name__}: {value!r}")
    if value < 0:
        raise ValueError(f"{name} 不能为负数：{value}")
    if value > max_val:
        raise ValueError(f"{name} 超出当前范围（0~{max_val}）：{value}")
    return True


def secure_compare(a: int, b: int, key_bits: int = 1024):
    validate_wealth(a, "Alice 的财富 a")
    validate_wealth(b, "Bob 的财富 b")

    ch = Channel()
    t0 = time.perf_counter()

    # 1. Alice 生成密钥对
    kp = keygen(key_bits)
    if isinstance(kp, tuple):
        pub, priv = kp[0], kp[1]
    else:
        pub, priv = kp, kp
    t_key = time.perf_counter() - t0
    n = _get_n(pub)
    NB = _nb(n)

    # 2. Alice -> Bob：公钥（公共参数）+ E(a)
    g = getattr(pub, "g", None) or getattr(pub, "G", None) or (n + 1)
    pub_bytes = n.to_bytes(NB, "big") + (g % (n * n)).to_bytes(2 * NB, "big")
    ca = encrypt(pub, a)
    ca_bytes = ca.to_bytes(2 * NB, "big")
    ch.send("A->B", pub_bytes + ca_bytes)

    # 3. Bob -> Alice：E(a-b) = E(a) * E(-b)
    cb_neg = encrypt(pub, (-b) % n)
    c_diff = (ca * cb_neg) % (n ** 2)
    ch.send("B->A", c_diff.to_bytes(2 * NB, "big"))

    # 4. Alice 解密并判定
    diff = decrypt(priv, c_diff)
    t_total = time.perf_counter() - t0

    if diff == 0:
        result = 0
    elif diff <= n // 2:
        result = 1
    else:
        result = 0

    return result, t_total, t_key, ch.total_bytes


if __name__ == "__main__":
    cases = [
        (0, 0, 0), (255, 255, 0), (128, 127, 1), (127, 128, 0),
        (1, 1, 0), (42, 42, 0), (254, 255, 0), (255, 254, 1),
        (0, 255, 0), (255, 0, 1),
    ]
    print("固定用例：")
    for a, b, exp in cases:
        got, *_ = secure_compare(a, b)
        print(f"a={a:3d} b={b:3d} 预期={exp} 实际={got} {'通过' if got == exp else '失败'}")

    print("\n非法输入：")
    for bad in [-1, 256, 3.5, "100", True]:
        try:
            secure_compare(bad, 100)
            print(f"{bad!r} 未被拒绝")
        except (ValueError, TypeError) as ex:
            print(f"{bad!r} -> {type(ex).__name__}: {ex}")

    random.seed(20261111)
    print("\n20 对随机输入：")
    for i in range(20):
        a = random.randint(0, 255)
        b = random.randint(0, 255)
        got, *_ = secure_compare(a, b)
        exp = 1 if a > b else 0
        print(f"{i+1:2d}: a={a:3d} b={b:3d} 预期={exp} 实际={got} {'通过' if got == exp else '失败'}")