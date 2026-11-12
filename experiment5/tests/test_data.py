"""验证：两套划分样本数、类别分布、样本互不重复、使用相同 5000 样本。"""
import numpy as np

from experiment5.src.data import build_splits, sample_500_per_class


def class_counts(indices, labels):
    return {c: int((labels[indices] == c).sum()) for c in range(10)}


def main():
    train, test, iid, skew = build_splits(seed=42)
    labels = np.array(train.targets)
    base = sample_500_per_class(train, seed=42)

    assert len(base) == 5000 and len(set(base.tolist())) == 5000
    print("base: 5000 个唯一训练索引，每类", class_counts(base, labels))

    for name, clients in (("IID", iid), ("Skew", skew)):
        all_idx = np.concatenate(clients)
        assert len(all_idx) == 5000 and len(set(all_idx.tolist())) == 5000
        assert set(all_idx.tolist()) == set(base.tolist()), f"{name} 与 base 样本集不一致"
        print(f"\n[{name}]")
        for k, ci in enumerate(clients):
            assert len(ci) == 1000, f"{name} 客户端 {k} 样本数 {len(ci)} != 1000"
            print(f"  client {k}: n={len(ci)}, 类别分布={class_counts(ci, labels)}")

    print("\ntest set size:", len(test))
    print("全部断言通过。")


if __name__ == "__main__":
    main()
