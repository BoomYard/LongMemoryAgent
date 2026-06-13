#!/usr/bin/env python3
"""
自动遍历三因子占比组合，依次修改 retriever.py 权重并运行 run_test.py。

用法：
  python test/sweep_weights.py
"""

import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RETRIEVER_PATH = os.path.join(PROJECT_ROOT, "memory_agent", "memory", "retriever.py")
RUN_TEST_PATH = os.path.join(PROJECT_ROOT, "memory_agent", "test", "run_test.py")
SUMMARY_MD = os.path.join(PROJECT_ROOT, "文档", "结果汇总", "result_summary.md")

# 三因子占比测试列表：(importance, bm25, relevance)  百分比
WEIGHT_CONFIGS = [
    (10, 20, 70),
    (10, 30, 60),
    (5,  25, 70),
    (5,  35, 60),
    (5,  40, 55),
    (0,  25, 75),
    (0,  40, 60),
]

# 标记行（用于定位替换位置）
COMMENT_MARKER = "# 加权求和："
SCORE_LINE_MARKER = "score = ("


def modify_weights(importance_pct: int, bm25_pct: int, relevance_pct: int):
    """逐行修改 retriever.py 中的三因子权重，避免正则替换多行文本导致格式损坏。"""
    with open(RETRIEVER_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    imp = importance_pct / 100
    bm = bm25_pct / 100
    rel = relevance_pct / 100

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 替换注释行
        if COMMENT_MARKER in line and "BM25" in line:
            new_lines.append(
                f"        # 加权求和：BM25 {bm25_pct}% + Importance {importance_pct}% + Relevance {relevance_pct}%\n"
            )
            i += 1
            continue

        # 替换 score = ( ... 多行块
        if SCORE_LINE_MARKER in line and "norm_bm25" in line:
            new_lines.append(f"            score = ({bm:.2f} * norm_bm25[i] +\n")
            new_lines.append(f"                     {imp:.2f} * norm_importance[i] +\n")
            new_lines.append(f"                     {rel:.2f} * norm_relevance[i])\n")
            # 跳过原来的 score 多行块（直到遇到包含 ) 的行）
            i += 1
            while i < len(lines) and ")" not in lines[i]:
                i += 1
            # 跳过包含 ) 的那一行
            if i < len(lines):
                i += 1
            continue

        new_lines.append(line)
        i += 1

    with open(RETRIEVER_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"[sweep] 权重已修改 -> importance={importance_pct}%, bm25={bm25_pct}%, relevance={relevance_pct}%")


def run_test():
    """运行 run_test.py -run -change 6，等待完成。"""
    print(f"[sweep] 开始运行 run_test.py -run -change 6 ...")
    result = subprocess.run(
        [sys.executable, RUN_TEST_PATH, "-run", "-change", "6"],
        cwd=PROJECT_ROOT,
    )
    if result.returncode != 0:
        print(f"[sweep] run_test.py 返回非零退出码: {result.returncode}")
    else:
        print(f"[sweep] run_test.py 运行完成")


def main():
    print(f"[sweep] 共 {len(WEIGHT_CONFIGS)} 组权重配置待测试\n")

    for idx, (imp, bm, rel) in enumerate(WEIGHT_CONFIGS):
        print(f"\n{'='*60}")
        print(f"[sweep] === 第 {idx+1}/{len(WEIGHT_CONFIGS)} 组: importance={imp}%, bm25={bm}%, relevance={rel}% ===")
        print(f"{'='*60}\n")

        modify_weights(imp, bm, rel)
        run_test()

        print(f"\n[sweep] 第 {idx+1} 组测试完成")

    # 恢复默认权重 (5, 15, 80)
    print(f"\n[sweep] 全部测试完成，恢复默认权重 (5%, 15%, 80%)...")
    modify_weights(5, 15, 80)
    print(f"[sweep] 完成！请查看 {SUMMARY_MD}")


if __name__ == "__main__":
    main()
