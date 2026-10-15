"""Paillier 功能测试：基本加解密、模边界、随机性、输入校验。"""
import secrets
import time
from paillier import keygen, encrypt, decrypt, add, mul_plain


def main():
    print("=" * 72)
    print("Paillier 功能测试")
    print("=" * 72)

    # 小参数调试
    print("\n[1] 小参数调试 (16 bit)")
    pub, priv = keygen(bits=16)
    n = pub.n
    print(f"  n.bit_length() = {n.bit_length()}, n = {n}")
    for m in [0, 1, n - 1]:
        c = encrypt(pub, m)
        d = decrypt(priv, c)
        print(f"  m={m} -> 解密 = {d}, 通过 = {d == m}")

    # 正式测试
    print("\n[2] 正式测试 (1024 bit)")
    t0 = time.perf_counter()
    pub, priv = keygen(bits=1024)
    t1 = time.perf_counter()
    n = pub.n
    print(f"  n.bit_length() = {n.bit_length()}, 生成耗时 {1000*(t1-t0):.1f} ms")

    results = []

    # 基本加解密
    for m in [0, 1, n - 1]:
        c = encrypt(pub, m)
        d = decrypt(priv, c)
        results.append(("基本加解密", f"m={m}", m, d, d == m))

    # 普通加法
    m1, m2 = 12, 23
    c1, c2 = encrypt(pub, m1), encrypt(pub, m2)
    d = decrypt(priv, add(pub, c1, c2))
    results.append(("普通加法", f"m1={m1},m2={m2}", (m1 + m2) % n, d, d == (m1 + m2) % n))

    # 模边界加法
    m1, m2 = n - 1, 10
    c1, c2 = encrypt(pub, m1), encrypt(pub, m2)
    d = decrypt(priv, add(pub, c1, c2))
    exp = (m1 + m2) % n
    results.append(("模边界加法", "m1=n-1,m2=10", exp, d, d == exp))

    # 模回绕加法
    m1, m2 = n - 2, n - 3
    c1, c2 = encrypt(pub, m1), encrypt(pub, m2)
    d = decrypt(priv, add(pub, c1, c2))
    exp = (m1 + m2) % n
    results.append(("模回绕加法", "m1=n-2,m2=n-3", exp, d, d == exp))

    # 标量乘法
    m = 12
    for k in [0, 1, 3]:
        c = encrypt(pub, m)
        d = decrypt(priv, mul_plain(pub, c, k))
        exp = (m * k) % n
        results.append(("标量乘法", f"m={m},k={k}", exp, d, d == exp))

    # 随机性
    m = 12345
    ciphertexts = [encrypt(pub, m) for _ in range(10)]
    all_distinct = len(set(ciphertexts)) == 10
    all_decrypt_ok = all(decrypt(priv, c) == m for c in ciphertexts)
    results.append(("随机性", "同一明文加密10次",
                    "密文全不同 + 解密正确",
                    f"distinct={all_distinct}, decrypt_ok={all_decrypt_ok}",
                    all_distinct and all_decrypt_ok))

    # 输入校验
    try:
        encrypt(pub, -1)
        results.append(("输入校验 m=-1", "拒绝", "拒绝", "未拒绝", False))
    except ValueError as e:
        results.append(("输入校验 m=-1", "拒绝", "拒绝", f"已拒绝", True))
    try:
        encrypt(pub, n)
        results.append(("输入校验 m=n", "拒绝", "拒绝", "未拒绝", False))
    except ValueError as e:
        results.append(("输入校验 m=n", "拒绝", "拒绝", f"已拒绝", True))

    # 20 组随机明文
    print("\n[3] 20 组随机明文验证")
    passed = 0
    for _ in range(20):
        m = secrets.randbelow(n)
        c = encrypt(pub, m)
        d = decrypt(priv, c)
        if d == m:
            passed += 1
    print(f"  通过 {passed}/20")

    # 汇总
    print("\n[4] 测试汇总")
    print(f"{'操作':<16}{'输入':<26}{'期望':<22}{'实际':<22}{'通过'}")
    print("-" * 90)
    for op, inp, exp, act, ok in results:
        e = str(exp)[:20]
        a = str(act)[:20]
        print(f"{op:<16}{inp:<26}{e:<22}{a:<22}{ok}")


if __name__ == "__main__":
    main()