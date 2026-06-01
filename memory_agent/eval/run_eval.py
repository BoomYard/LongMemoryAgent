"""

本文件封装了评测流程。注意：生成（Generation）和评测（Judge）使用不同的 LLM：
- Generation：从 .env 读取 LLM_BASE_URL（通常指向本地 vLLM）
- Judge：必须用云端 API（DeepSeek / DashScope），通过 --judge_base_url 和 --judge_model 指定

所有子进程通过 PYTHONPATH 注入项目根目录，保证 memory_agent 模块在任何位置都能 import。
子进程的 cwd 继承当前 Shell，相对路径按用户的工作目录解析，符合直觉。
"""

import subprocess
import sys
import os

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
EVAL_KIT_DIR = os.path.join(PROJECT_ROOT, "eval_kit", "eval_kit")


def _subprocess_env():
    """返回注入项目根到 PYTHONPATH 后的环境变量，确保子进程能 import memory_agent。"""
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + pythonpath if pythonpath else "")
    return env


def run_generation(eval_set: str, agent: str, output: str, limit_conversations: int = None):
    cmd = [
        sys.executable, os.path.join(EVAL_KIT_DIR, "run_generation.py"),
        "--eval_set", eval_set,
        "--agent", agent,
        "--output", output,
    ]
    if limit_conversations:
        cmd.extend(["--limit_conversations", str(limit_conversations)])
    subprocess.run(cmd, check=True, env=_subprocess_env())


def run_judge(predictions: str, output: str, judge_base_url: str = None,
              judge_model: str = None, num_workers: int = 4):
    cmd = [
        sys.executable, os.path.join(EVAL_KIT_DIR, "run_judge.py"),
        "--predictions", predictions,
        "--output", output,
        "--num_workers", str(num_workers),
    ]
    if judge_base_url:
        cmd.extend(["--judge_base_url", judge_base_url])
    if judge_model:
        cmd.extend(["--judge_model", judge_model])
    subprocess.run(cmd, check=True, env=_subprocess_env())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="长期记忆 Agent 评测入口（Generation + Judge）")
    parser.add_argument("--eval_set", default="eval_set.json",
                        help="评测集 JSON 路径")
    parser.add_argument("--agent", default="memory_agent.agent.controller:MemoryAgent",
                        help="Agent 模块路径:类名")
    parser.add_argument("--output", default="predictions.json",
                        help="预测结果输出路径")
    parser.add_argument("--results", default="results.json",
                        help="Judge 评测结果输出路径")
    parser.add_argument("--limit_conversations", type=int, default=None,
                        help="只跑前 N 段对话（快速调试用）")
    parser.add_argument("--skip_generation", action="store_true",
                        help="已有 predictions 时跳过生成，只跑 Judge")
    parser.add_argument("--skip_judge", action="store_true",
                        help="跳过 Judge，只生成 predictions")
    parser.add_argument("--judge_base_url", default=None,
                        help="Judge 的 API 地址（覆盖 .env），如 https://api.deepseek.com/v1")
    parser.add_argument("--judge_model", default=None,
                        help="Judge 的模型名（覆盖 .env），如 deepseek-v4-flash")
    parser.add_argument("--judge_api_key", default=None,
                        help="Judge 的 API Key（覆盖 .env），如 sk-xxx")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Judge 并发数")
    args = parser.parse_args()

    if args.judge_api_key:
        os.environ["LLM_API_KEY"] = args.judge_api_key

    if not args.skip_generation:
        print("[Step 1] Running generation...")
        run_generation(args.eval_set, args.agent, args.output, args.limit_conversations)

    if not args.skip_judge:
        print("[Step 2] Running judge...")
        run_judge(args.output, args.results, args.judge_base_url, args.judge_model, args.num_workers)
