"""读取词频统计结果, 画出前 10 个高频词的水平柱状图并保存为矢量 PDF。

用法:
    python code/plot_word_freq.py [统计结果文件] [输出 PDF]

默认输入 ``result/text_stats_output.txt``, 默认输出 ``figures/word_freq.pdf``。
输出使用 matplotlib 的 PDF 后端 (纯矢量, 不是截图), 并嵌入 TrueType 字体,
方便直接插入 LaTeX。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无需图形界面, 服务器/脚本环境下也能运行

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator

DEFAULT_STATS = Path("result/text_stats_output.txt")
DEFAULT_OUTPUT = Path("figures/word_freq.pdf")

TOP_N = 10

# 输入文件优先按 UTF-8(含 BOM) 读取, 失败后回退到 GBK, 兼容中文 Windows 保存的文件
ENCODINGS = ("utf-8-sig", "gbk")

# 中文字体候选, 按优先级排列(Windows 上 SimHei / Microsoft YaHei 通常都可用)
CJK_FONTS = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
)

# 形如 " 1. hello                        3"
ENTRY_RE = re.compile(r"^\s*\d+\.\s+(?P<word>.+?)\s+(?P<count>\d+)\s*$")


def read_text(path: Path) -> str:
    """读取文本文件, 自动尝试常见编码。"""
    for encoding in ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def parse_entries(text: str, limit: int = TOP_N) -> list[tuple[str, int]]:
    """从统计输出中解析 "序号. 单词 次数" 行, 取出现次数最高的前 limit 个。"""
    entries: list[tuple[str, int]] = []
    for line in text.splitlines():
        match = ENTRY_RE.match(line)
        if match:
            entries.append((match.group("word").strip(), int(match.group("count"))))
    # 输出文件里已经排好序, 这里再排一次以防顺序变化: 次数降序, 同次数按单词升序
    entries.sort(key=lambda item: (-item[1], item[0]))
    return entries[:limit]


def setup_chinese_font() -> str | None:
    """挑选一个系统中可用的中文字体并写入 rcParams, 返回字体名。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_FONTS if name in available), None)
    if chosen is None:
        for name in CJK_FONTS:
            try:
                font_manager.findfont(name, fallback_to_default=False)
            except ValueError:
                continue
            chosen = name
            break

    # 无论是否找到中文字体都把候选写进配置, 由 matplotlib 自行回退
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = (
        [chosen, *CJK_FONTS] if chosen else list(CJK_FONTS)
    )
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号正常显示
    matplotlib.rcParams["pdf.fonttype"] = 42  # TrueType 嵌入, LaTeX 里不会缺字
    matplotlib.rcParams["ps.fonttype"] = 42
    return chosen


def plot_barh(items: list[tuple[str, int]], output: Path) -> None:
    """画水平柱状图并保存为矢量 PDF。"""
    # barh 从下往上画, 反转后出现次数最多的词在最上方
    words = [word for word, _ in reversed(items)]
    counts = [count for _, count in reversed(items)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(words, counts, color="#4C72B0", height=0.7)

    ax.set_xlabel("出现次数")
    ax.set_ylabel("单词")
    ax.set_title(f"词频最高的前 {len(items)} 个单词")
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_xlim(0, max(counts) * 1.15)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    # 明确指定 PDF 后端, 保证输出是矢量文件
    fig.savefig(output, format="pdf")  # 页面尺寸保持 figsize, 不做 tight 裁剪
    plt.close(fig)


def main(argv: list[str]) -> int:
    if len(argv) > 3:
        print(
            "用法: python code/plot_word_freq.py [统计结果文件] [输出 PDF]",
            file=sys.stderr,
        )
        return 2

    stats_path = Path(argv[1]) if len(argv) >= 2 else DEFAULT_STATS
    output_path = Path(argv[2]) if len(argv) >= 3 else DEFAULT_OUTPUT

    if not stats_path.is_file():
        print(f"找不到统计结果文件: {stats_path}", file=sys.stderr)
        return 1

    items = parse_entries(read_text(stats_path))
    if not items:
        print(f"{stats_path}: 没有解析出任何词频数据。", file=sys.stderr)
        return 1

    font = setup_chinese_font()
    plot_barh(items, output_path)

    if font is None:
        print("警告: 未找到常见中文字体, 图中中文可能显示为方块。")
    else:
        print(f"使用中文字体: {font}")
    print(f"已从 {stats_path} 读取 {len(items)} 个高频词。")
    print(f"矢量 PDF 已保存到: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
