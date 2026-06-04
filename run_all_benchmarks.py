#!/usr/bin/env python3
"""
=============================================================================
  run_all_benchmarks.py  --  全量自动化评测脚本
=============================================================================

依次对以下四套系统跑完整的 Generation + Judge：
  1. NoMemory     — 零记忆（下界）
  2. FullContext  — 整段对话截断到模型上限
  3. VanillaRAG   — 逐轮切片 + 向量检索
  4. MemoryAgent  — 你自己的记忆系统

特性：
  - 每套系统先跑 Generation 再跑 Judge，中间自动存档
  - Generation 失败自动重启 vLLM 并以 --resume 接续（不丢已答数据）
  - Judge 失败自动重试（最多 5 次，指数退避）
  - 所有输出集中到 benchmark_results/ 目录
  - 运行日志写入 run_all_benchmarks.log
  - PID 文件防止重复启动

使用方式：
  conda deactivate
  source .venv/bin/activate
  nohup python run_all_benchmarks.py > /dev/null 2>&1 &

  查看进度的 4 种方法（任选其一）：
    tail -f run_all_benchmarks.log
    watch -n 10 'ls -lh benchmark_results/'
    python -c "import json; d=json.load(open('benchmark_results/predictions_memory.json')); print(len(d))"
    ps aux | grep run_all

  （关掉 SSH 后进程继续运行，数据安全落盘）
=============================================================================
"""

import subprocess
import os
import sys
import json
import time
import shutil
import argparse
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path


# ========================================================================
# 路径配置（依赖固定路径的都在这里，方便维护）
# ========================================================================
PROJECT_ROOT    = Path("/home/LongMemoryAgent")
VENV_PYTHON     = str(PROJECT_ROOT / ".venv" / "bin" / "python")
EVAL_KIT_DIR    = str(PROJECT_ROOT / "eval_kit" / "eval_kit")
EVAL_SET_PATH   = os.path.join(EVAL_KIT_DIR, "eval_set.json")
OUTPUT_DIR      = str(PROJECT_ROOT / "benchmark_results")
LOG_PATH        = str(PROJECT_ROOT / "run_all_benchmarks.log")
PID_PATH        = str(PROJECT_ROOT / "run_all_benchmarks.pid")
VLLM_LOG_PATH   = str(PROJECT_ROOT / "vllm_benchmark.log")


# ========================================================================
# 评测配置
# ========================================================================
BENCHMARKS = [
    {
        "name":        "NoMemory",
        "agent":       "agent_template:NoMemoryAgent",
        "predictions": "predictions_nomem.json",
        "results":     "results_nomem.json",
    },
    {
        "name":        "FullContext",
        "agent":       "agent_template:FullContextAgent",
        "predictions": "predictions_fullctx.json",
        "results":     "results_fullctx.json",
    },
    {
        "name":        "VanillaRAG",
        "agent":       "agent_template:VanillaRAGAgent",
        "predictions": "predictions_rag.json",
        "results":     "results_rag.json",
    },
    {
        "name":        "MemoryAgent",
        "agent":       "memory_agent.agent.controller:MemoryAgent",
        "predictions": "predictions_memory.json",
        "results":     "results_memory.json",
    },
]

MAX_GEN_RETRIES       = 3     # 每个 benchmark 的 Generation 最多重试次数
MAX_JUDGE_RETRIES     = 5     # Judge 最多重试次数
VLLM_START_TIMEOUT    = 120   # 等待 vLLM 启动的最长秒数
VLLM_HEALTH_RETRIES   = 3     # 每次健康检查失败后重试次数
VLLM_HEALTH_INTERVAL  = 10    # 健康检查间隔（秒）
GEN_TIMEOUT           = 14400 # 单个 benchmark Generation 超时（4 小时）
JUDGE_TIMEOUT         = 7200  # 单个 benchmark Judge 超时（2 小时）


# ========================================================================
# 日志
# ========================================================================
_log_lock = None  # placeholder，防止多线程写同一文件

def log(msg: str) -> None:
    """同时输出到终端和日志文件，保证 flush。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass  # 写日志失败不影响主流程


# ========================================================================
# .env 读取
# ========================================================================
def load_dotenv_config() -> dict:
    """从项目根目录的 .env 读取所有键值对（不依赖 python-dotenv）。"""
    config = {}
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return config
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            config[key.strip()] = val.strip().strip('"').strip("'")
    return config


# ========================================================================
# 子进程执行
# ========================================================================
def run_subprocess(
    cmd: list,
    env: dict = None,
    cwd: str = None,
    timeout: int = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """
    执行子进程。
    实时把 stdout/stderr 写入日志（每行一个 log entry），避免长时间 silence。
    """
    cmd_str = " ".join(cmd)
    log(f"  CMD: {cmd_str}")

    child = subprocess.Popen(
        cmd,
        env=env or os.environ,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    collected: list[str] = []
    start = time.time()

    try:
        for line in child.stdout:
            line = line.rstrip("\n")
            if line:
                log(f"    | {line}")
            collected.append(line)
            # 心跳日志：每 5 分钟报告仍在运行
            if time.time() - start > 300:
                log(f"    | ... still running ({int(time.time()-start)}s elapsed) ...")
                start = time.time()

        child.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
        log(f"  TIMEOUT after {timeout}s")
        raise
    except Exception:
        child.kill()
        child.wait()
        raise

    stdout = "\n".join(collected)

    if check and child.returncode != 0:
        raise subprocess.CalledProcessError(
            child.returncode, cmd, output=stdout, stderr=None
        )

    return subprocess.CompletedProcess(
        args=cmd, returncode=child.returncode, stdout=stdout, stderr=""
    )


# ========================================================================
# vLLM 管理
# ========================================================================
def _http_get(url: str, timeout: int = 5) -> bool:
    """简单 HTTP GET，只关心 200 状态码。"""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def check_vllm_health() -> bool:
    """向 vLLM 发送 /health 请求，多次重试以防短暂抖动。"""
    for attempt in range(1, VLLM_HEALTH_RETRIES + 1):
        if _http_get("http://localhost:8000/health", timeout=5):
            return True
        if attempt < VLLM_HEALTH_RETRIES:
            time.sleep(3)
    return False


def ensure_vllm_running() -> bool:
    """检测 vLLM，未运行则尝试启动并等待。"""
    if check_vllm_health():
        log("vLLM: 已在线")
        return True

    log("vLLM: 未响应，尝试启动 ...")

    vllm_cmd = [
        VENV_PYTHON, "-m", "vllm.entrypoints.cli.main", "serve",
        "Qwen/Qwen2.5-3B-Instruct-AWQ",
        "--port", "8000",
        "--max-model-len", "8192",
        "--gpu-memory-utilization", "0.75",
    ]

    # 环境变量不含 conda base 路径
    vllm_env = os.environ.copy()
    vllm_env["HF_ENDPOINT"] = "https://hf-mirror.com"
    vllm_env["PYTHONPATH"] = str(PROJECT_ROOT)

    with open(VLLM_LOG_PATH, "a", encoding="utf-8") as vllm_log:
        vllm_log.write(f"\n--- vLLM started at {datetime.now()} ---\n")
        proc = subprocess.Popen(
            vllm_cmd, stdout=vllm_log, stderr=subprocess.STDOUT,
            env=vllm_env, cwd=str(PROJECT_ROOT),
        )

    log(f"vLLM: PID={proc.pid}，等待就绪（最长 {VLLM_START_TIMEOUT}s）...")

    waited = 0
    while waited < VLLM_START_TIMEOUT:
        time.sleep(VLLM_HEALTH_INTERVAL)
        waited += VLLM_HEALTH_INTERVAL

        if proc.poll() is not None:
            log(f"vLLM: 进程意外退出，exitcode={proc.returncode}")
            return False

        if check_vllm_health():
            log(f"vLLM: 就绪（用时约 {waited}s）")
            return True

        log(f"vLLM: 等待中 ... {waited}s / {VLLM_START_TIMEOUT}s")

    log("vLLM: 启动超时")
    return False


# ========================================================================
# eval_set 准备
# ========================================================================
def regenerate_eval_set() -> bool:
    """如果 eval_set.json 不存在或条目过少（<40 段对话），重新生成。"""
    need = False
    if not os.path.exists(EVAL_SET_PATH):
        need = True
    else:
        try:
            with open(EVAL_SET_PATH, encoding="utf-8") as f:
                data = json.load(f)
            if len(data) < 40:
                log(f"eval_set.json 只有 {len(data)} 段对话，需要重新生成（目标 ≥40）")
                need = True
        except (json.JSONDecodeError, OSError):
            need = True

    if not need:
        log(f"eval_set.json 就绪 ({len(data)} 段对话)")
        return True

    log("正在生成 eval_set.json（per_category=40）...")
    try:
        run_subprocess(
            [
                VENV_PYTHON,
                os.path.join(EVAL_KIT_DIR, "prepare_eval_set.py"),
                "--output", EVAL_SET_PATH,
                "--per_category", "40",
                "--seed", "42",
            ],
            cwd=EVAL_KIT_DIR,
            timeout=600,
            check=True,
        )
        log("eval_set.json 生成完成")
        return True
    except Exception as e:
        log(f"生成 eval_set.json 失败: {e}")
        return False


# ========================================================================
# 单个 benchmark 的执行
# ========================================================================
def run_generation_for(bm: dict, gen_env: dict) -> bool:
    """对一个 benchmark 运行 Generation（失败时可重试，自动 --resume）。"""
    pred_path = os.path.join(OUTPUT_DIR, bm["predictions"])

    cmd = [
        VENV_PYTHON,
        os.path.join(EVAL_KIT_DIR, "run_generation.py"),
        "--eval_set", EVAL_SET_PATH,
        "--agent", bm["agent"],
        "--output", pred_path,
    ]
    if os.path.exists(pred_path):
        cmd.append("--resume")
        log(f"  [resume] 已有部分预测，跳过已完成的 qa_id")

    run_subprocess(cmd, env=gen_env, cwd=EVAL_KIT_DIR, timeout=GEN_TIMEOUT, check=True)
    return True


def run_judge_for(bm: dict, judge_env: dict) -> bool:
    """对一个 benchmark 运行 Judge（失败时可重试）。"""
    pred_path = os.path.join(OUTPUT_DIR, bm["predictions"])
    res_path  = os.path.join(OUTPUT_DIR, bm["results"])

    cmd = [
        VENV_PYTHON,
        os.path.join(EVAL_KIT_DIR, "run_judge.py"),
        "--predictions", pred_path,
        "--output", res_path,
        "--judge_base_url", judge_env["LLM_BASE_URL"],
        "--judge_model", judge_env["LLM_MODEL"],
        "--num_workers", "4",
    ]

    run_subprocess(cmd, env=judge_env, cwd=EVAL_KIT_DIR, timeout=JUDGE_TIMEOUT, check=True)
    return True


def execute_one_benchmark(bm: dict, gen_env: dict, judge_env: dict, idx: int, total: int) -> dict:
    """执行单个 benchmark 的完整流程，返回结果字典。"""
    result = {
        "name":        bm["name"],
        "gen_ok":      False,
        "judge_ok":    False,
        "gen_retries": 0,
        "judge_retries": 0,
        "duration":    0.0,
        "predictions": bm["predictions"],
        "results":     bm["results"],
    }

    log("")
    log("=" * 70)
    log(f"  [{idx}/{total}] {bm['name']}")
    log("=" * 70)

    t0 = time.time()

    # ---------- Step 1: Generation ----------
    log(f"[{bm['name']}] === Step 1: Generation ===")
    gen_ok = False
    gen_attempt = 0
    for gen_attempt in range(1, MAX_GEN_RETRIES + 1):
        log(f"[{bm['name']}] Generation 第 {gen_attempt}/{MAX_GEN_RETRIES} 次尝试")

        if not check_vllm_health():
            log(f"[{bm['name']}] vLLM 掉线，重启中 ...")
            if not ensure_vllm_running():
                log(f"[{bm['name']}] vLLM 启动失败，稍后重试")
                time.sleep(30)
                continue

        try:
            run_generation_for(bm, gen_env)
            gen_ok = True
            break
        except subprocess.TimeoutExpired:
            log(f"[{bm['name']}] Generation 超时（>{GEN_TIMEOUT}s）")
        except subprocess.CalledProcessError as e:
            log(f"[{bm['name']}] Generation 返回非零: {e.returncode}")
        except Exception as e:
            log(f"[{bm['name']}] Generation 异常: {type(e).__name__}: {e}")

        if gen_attempt < MAX_GEN_RETRIES:
            wait = 60 * gen_attempt
            log(f"[{bm['name']}] 等待 {wait}s 后重试 ...")
            time.sleep(wait)

    result["gen_retries"] = gen_attempt
    result["gen_ok"] = gen_ok
    if not gen_ok:
        log(f"[{bm['name']}] Generation 失败（{MAX_GEN_RETRIES} 次尝试均未成功）")
        result["duration"] = time.time() - t0
        return result

    log(f"[{bm['name']}] Generation 完成 ✓")

    # ---------- Step 2: Judge ----------
    log(f"[{bm['name']}] === Step 2: Judge ===")
    judge_ok = False
    judge_attempt = 0
    for judge_attempt in range(1, MAX_JUDGE_RETRIES + 1):
        log(f"[{bm['name']}] Judge 第 {judge_attempt}/{MAX_JUDGE_RETRIES} 次尝试")

        try:
            run_judge_for(bm, judge_env)
            judge_ok = True
            break
        except subprocess.TimeoutExpired:
            log(f"[{bm['name']}] Judge 超时（>{JUDGE_TIMEOUT}s）")
        except subprocess.CalledProcessError as e:
            log(f"[{bm['name']}] Judge 返回非零: {e.returncode}")
        except Exception as e:
            log(f"[{bm['name']}] Judge 异常: {type(e).__name__}: {e}")

        if judge_attempt < MAX_JUDGE_RETRIES:
            wait = 30 * judge_attempt
            log(f"[{bm['name']}] 等待 {wait}s 后重试 ...")
            time.sleep(wait)

    result["judge_retries"] = judge_attempt
    result["judge_ok"] = judge_ok
    result["duration"] = time.time() - t0

    if judge_ok:
        log(f"[{bm['name']}] Judge 完成 ✓")
    else:
        log(f"[{bm['name']}] Judge 失败（{MAX_JUDGE_RETRIES} 次尝试均未成功）")

    return result


# ========================================================================
# 汇总报告
# ========================================================================
def write_summary(results: list[dict]) -> None:
    """将四个 benchmark 的结果写入 SUMMARY.txt 并打印到日志。"""
    lines = [
        "",
        "=" * 70,
        "              全 量 评 测 汇 总",
        f"              完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
    ]

    for r in results:
        status = "✓ PASS" if (r["gen_ok"] and r["judge_ok"]) else "✗ FAIL"
        lines.append(f"  [{status}] {r['name']}")
        lines.append(f"    Generation: {'OK' if r['gen_ok'] else 'FAILED'}  (重试 {r['gen_retries']} 次)")
        lines.append(f"    Judge:      {'OK' if r['judge_ok'] else 'FAILED'}  (重试 {r['judge_retries']} 次)")
        lines.append(f"    耗时: {r['duration']:.0f}s ({r['duration']/60:.1f} min)")

        if r["gen_ok"] and r["judge_ok"]:
            results_path = os.path.join(OUTPUT_DIR, r["results"])
            if os.path.exists(results_path):
                try:
                    with open(results_path, encoding="utf-8") as f:
                        data = json.load(f)
                    o = data.get("overall", {})
                    lines.append(f"    Score: {o.get('score', 'N/A')}")
                    lines.append(f"    F1:    {o.get('f1', 'N/A')}")
                    lines.append(f"    EM:    {o.get('em', 'N/A')}")
                except Exception:
                    lines.append("    (无法读取详细分数)")
        lines.append("")

    # 表格对比
    lines.append("-" * 70)
    header = f"  {'Benchmark':<16} {'Score':>8} {'F1':>8} {'EM':>8} {'Status':>8}"
    lines.append(header)
    lines.append("-" * 70)
    for r in results:
        name = r["name"]
        status = "OK" if (r["gen_ok"] and r["judge_ok"]) else "FAIL"
        score_str = "N/A"
        f1_str = "N/A"
        em_str = "N/A"
        if r["gen_ok"] and r["judge_ok"]:
            results_path = os.path.join(OUTPUT_DIR, r["results"])
            if os.path.exists(results_path):
                try:
                    with open(results_path, encoding="utf-8") as f:
                        data = json.load(f)
                    o = data.get("overall", {})
                    score_str = f"{o.get('score', 0):.4f}"
                    f1_str    = f"{o.get('f1', 0):.4f}"
                    em_str    = f"{o.get('em', 0):.4f}"
                except Exception:
                    pass
        lines.append(f"  {name:<16} {score_str:>8} {f1_str:>8} {em_str:>8} {status:>8}")
    lines.append("-" * 70)

    lines.append("")
    lines.append(f"  所有输出文件位于: {OUTPUT_DIR}/")
    for r in results:
        lines.append(f"    {r['predictions']}  /  {r['results']}")
    lines.append("")

    summary_text = "\n".join(lines)
    log(summary_text)

    summary_path = os.path.join(OUTPUT_DIR, "SUMMARY.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log(f"汇总已保存至 {summary_path}")


# ========================================================================
# 入口
# ========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="全量自动化评测：依次运行 NoMemory / FullContext / VanillaRAG / MemoryAgent"
    )
    parser.add_argument(
        "--skip", nargs="*",
        choices=["nomem", "fullctx", "rag", "memory"],
        help="跳过指定 benchmark（例如 --skip nomem rag）",
    )
    parser.add_argument(
        "--only", nargs="*",
        choices=["nomem", "fullctx", "rag", "memory"],
        help="只运行指定 benchmark（例如 --only memory）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="只打印将要执行的步骤，不实际运行",
    )
    args = parser.parse_args()

    # --- PID 文件：防止重复启动 ---
    if os.path.exists(PID_PATH):
        try:
            with open(PID_PATH) as f:
                old_pid = int(f.read().strip())
            # 检查旧进程是否还在
            os.kill(old_pid, 0)
            log(f"检测到已有评测进程 (PID={old_pid}) 正在运行，退出")
            sys.exit(1)
        except (OSError, ValueError):
            # 旧进程已不存在，移除残留 PID 文件
            os.unlink(PID_PATH)

    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))

    # --- 初始化 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 70)
    log("  全量自动化评测脚本 启动")
    log(f"  PID: {os.getpid()}")
    log(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Python: {VENV_PYTHON}")
    log(f"  输出目录: {OUTPUT_DIR}")
    log("=" * 70)

    # --- 加载配置 ---
    dotenv = load_dotenv_config()

    # Generation 用本地 vLLM
    gen_env = os.environ.copy()
    gen_env["LLM_BASE_URL"] = dotenv.get("LLM_BASE_URL", "http://localhost:8000/v1")
    gen_env["LLM_API_KEY"]  = dotenv.get("LLM_API_KEY", "EMPTY")
    gen_env["LLM_MODEL"]    = dotenv.get("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
    gen_env["PYTHONPATH"]   = str(PROJECT_ROOT)
    gen_env["HF_ENDPOINT"]  = "https://hf-mirror.com"

    # Judge 用云端 API（从 .env 的 AGENT_* 读取）
    judge_env = os.environ.copy()
    judge_env["LLM_BASE_URL"] = dotenv.get("AGENT_URL", "")
    judge_env["LLM_API_KEY"]  = dotenv.get("AGENT_API_KEY", "")
    judge_env["LLM_MODEL"]    = dotenv.get("AGENT_MODEL", "")
    judge_env["PYTHONPATH"]   = str(PROJECT_ROOT)

    if not judge_env["LLM_BASE_URL"] or not judge_env["LLM_API_KEY"]:
        log("警告: .env 中未找到 AGENT_URL / AGENT_API_KEY，Judge 可能无法正常工作")

    log(f"Generation: {gen_env['LLM_BASE_URL']}  model={gen_env['LLM_MODEL']}")
    log(f"Judge:      {judge_env['LLM_BASE_URL']}  model={judge_env['LLM_MODEL']}")

    # --- 过滤 benchmark ---
    name_map = {
        "nomem":   "NoMemory",
        "fullctx": "FullContext",
        "rag":     "VanillaRAG",
        "memory":  "MemoryAgent",
    }
    skip_set = {name_map[n] for n in (args.skip or [])}
    only_set = {name_map[n] for n in (args.only or [])} if args.only else None

    to_run = []
    for bm in BENCHMARKS:
        if bm["name"] in skip_set:
            continue
        if only_set is not None and bm["name"] not in only_set:
            continue
        to_run.append(bm)

    if not to_run:
        log("没有需要执行的 benchmark（全部被 --skip 或 --only 过滤）")
        os.unlink(PID_PATH)
        sys.exit(0)

    log(f"待执行: {[b['name'] for b in to_run]}")

    # --- 准备 eval_set ---
    if not regenerate_eval_set():
        log("FATAL: 无法生成 eval_set.json")
        os.unlink(PID_PATH)
        sys.exit(1)

    if args.dry_run:
        log("--dry-run 模式，仅打印计划，不实际执行")
        for bm in to_run:
            log(f"  {bm['name']}:  {bm['agent']} -> {bm['predictions']} -> {bm['results']}")
        os.unlink(PID_PATH)
        sys.exit(0)

    # --- 启动 vLLM ---
    if not ensure_vllm_running():
        log("FATAL: vLLM 启动失败，无法继续")
        os.unlink(PID_PATH)
        sys.exit(1)

    # --- 逐个跑 ---
    all_results = []
    for i, bm in enumerate(to_run, start=1):
        try:
            res = execute_one_benchmark(bm, gen_env, judge_env, i, len(to_run))
            all_results.append(res)
        except KeyboardInterrupt:
            log("\n收到中断信号，保存已完成结果后退出 ...")
            all_results.append({
                "name": bm["name"],
                "gen_ok": False, "judge_ok": False,
                "gen_retries": 0, "judge_retries": 0,
                "duration": 0,
                "predictions": bm["predictions"],
                "results": bm["results"],
            })
            break
        except Exception as e:
            log(f"[{bm['name']}] 未捕获的异常: {type(e).__name__}: {e}")
            all_results.append({
                "name": bm["name"],
                "gen_ok": False, "judge_ok": False,
                "gen_retries": 0, "judge_retries": 0,
                "duration": 0,
                "predictions": bm["predictions"],
                "results": bm["results"],
            })

    # --- 汇总 ---
    write_summary(all_results)

    # --- 清理 ---
    try:
        os.unlink(PID_PATH)
    except OSError:
        pass

    log("脚本退出。")


if __name__ == "__main__":
    main()
