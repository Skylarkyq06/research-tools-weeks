"""明文基准：3 个查询对 6 张图库的 SSE 距离与排名。"""
import json
from pathlib import Path
import numpy as np
from preprocess import load_all, sse, DIM

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


def main():
    gallery, queries, checks = load_all()

    report = {}
    print("=" * 72)
    print("明文基准距离表")
    print("=" * 72)

    for qid, qv in queries.items():
        print(f"\n{qid}:")
        rows = []
        for gid, gv in gallery.items():
            d = sse(qv, gv)
            mse = d / DIM
            rows.append((gid, d, mse))
        rows.sort(key=lambda r: (r[1], r[0]))
        for rank, (gid, d, mse) in enumerate(rows, 1):
            print(f"  rank {rank}: {gid}  SSE={d}  MSE={mse:.4f}")
        report[qid] = [
            {"rank": i + 1, "gid": gid, "sse": d, "mse": mse}
            for i, (gid, d, mse) in enumerate(rows)
        ]

    # 极值
    black, white = checks["black"], checks["white"]
    sse_bw = sse(black, white)
    print(f"\nblack vs white SSE = {sse_bw}")
    report["check_black_white"] = sse_bw

    # 保存
    out = RESULTS / "plaintext_baseline.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已保存: {out}")


if __name__ == "__main__":
    main()