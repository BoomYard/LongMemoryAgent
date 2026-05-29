"""
[规划对应] 评测入口

本文件是运行评测的入口脚本，封装了调用助教提供的评测工具包的逻辑：
- Step 1：调用 run_generation.py 生成预测结果
- Step 2：调用 run_judge.py 用 LLM-as-Judge 打分
"""

import subprocess
import sys
import os

EVAL_KIT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit")


def run_generation(eval_set: str, agent: str, output: str, limit_conversations: int = None):
    cmd = [
        sys.executable, os.path.join(EVAL_KIT_DIR, "run_generation.py"),
        "--eval_set", eval_set,
        "--agent", agent,
        "--output", output,
    ]
    if limit_conversations:
        cmd.extend(["--limit_conversations", str(limit_conversations)])
    subprocess.run(cmd, check=True)


def run_judge(predictions: str, output: str, num_workers: int = 4):
    cmd = [
        sys.executable, os.path.join(EVAL_KIT_DIR, "run_judge.py"),
        "--predictions", predictions,
        "--output", output,
        "--num_workers", str(num_workers),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_set", default="eval_set.json")
    parser.add_argument("--agent", default="memory_agent.agent.controller:MemoryAgent")
    parser.add_argument("--output", default="predictions.json")
    parser.add_argument("--results", default="results.json")
    parser.add_argument("--limit_conversations", type=int, default=None)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_judge", action="store_true")
    args = parser.parse_args()

    if not args.skip_generation:
        print("[Step 1] Running generation...")
        run_generation(args.eval_set, args.agent, args.output, args.limit_conversations)

    if not args.skip_judge:
        print("[Step 2] Running judge...")
        run_judge(args.output, args.results)
