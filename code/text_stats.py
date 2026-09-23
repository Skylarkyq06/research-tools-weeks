"""统计文本文件中的词频，并输出出现次数最多的前 10 个单词。

用法:
    python code/text_stats.py <文本文件路径>

单词的定义:以字母或数字开头和结尾,中间允许撇号或连字符
(例如 don't、state-of-the-art),统一按小写计数。
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

TOP_N = 10

WORD_RE = re.compile(r"[a-z0-9]+(?:['\u2019-][a-z0-9]+)*")

# 优先 UTF-8(含 BOM),失败后回退到 GBK,兼容中文 Windows 上保存的文件。
ENCODINGS = ("utf-8-sig", "gbk")


def read_text(path: Path) -> str:
    """读取文本文件,自动尝试常见编码。"""
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def count_words(text: str) -> Counter[str]:
    """把文本切成小写单词并统计出现次数。"""
    return Counter(WORD_RE.findall(text.lower()))


def top_words(counter: Counter[str], limit: int = TOP_N) -> list[tuple[str, int]]:
    """取前 limit 个高频词:先按次数降序,次数相同按单词升序。"""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("用法: python code/text_stats.py <文本文件>", file=sys.stderr)
        return 2

    path = Path(argv[1])
    if not path.is_file():
        print(f"找不到文件: {path}", file=sys.stderr)
        return 1

    counter = count_words(read_text(path))
    if not counter:
        print(f"{path}: 没有统计到任何单词。")
        return 0

    items = top_words(counter)
    print(f"文件: {path}")
    print(f"总词数: {sum(counter.values())}    不同单词数: {len(counter)}")
    print(f"出现次数最多的前 {len(items)} 个单词:")
    for rank, (word, count) in enumerate(items, start=1):
        print(f"{rank:>2}. {word:<24} {count:>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
