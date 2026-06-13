#!/usr/bin/env python3
"""
快速测试脚本 —— 避免每次重新 ingest 数据。

用法：
  python test/run_test.py -ingest              # 只导入数据并持久化
  python test/run_test.py -run                 # 全量测试（有缓存则跳过 ingest）
  python test/run_test.py -run -change 5       # 并发 answer 数改为 5
  python test/run_test.py -run -change 1       # 串行 answer（观察日志用）
"""

import argparse
import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 路径 ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

EVAL_SET_PATH = os.path.join(PROJECT_ROOT, "eval_kit", "eval_kit", "eval_set.json")
CHROMA_DB_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DB_ROOT = os.path.join(TEST_DIR, "saved_chroma_db")       # 每 conversation 一个子目录
PERSONA_FILE = os.path.join(TEST_DIR, "persona_summaries.json")  # {sample_id: summary}
PREDICTIONS_FILE = os.path.join(TEST_DIR, "predictions.json")
JUDGE_RESULTS_FILE = os.path.join(TEST_DIR, "judge_results.json")
RESULTS_MD_FILE = os.path.join(TEST_DIR, "results.md")
SUMMARY_DIR = os.path.join(PROJECT_ROOT, "文档", "结果汇总")
SUMMARY_MD = os.path.join(SUMMARY_DIR, "result_summary.md")

# 临时禁用 SentenceTransformer 在首次 import 时打印的进度条
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

print_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════
#  从 ChromaDB 直接加载已有 collection（不删除、不重建）
# ═══════════════════════════════════════════════════════════════
def _load_store_from_disk(persist_dir: str):
    """打开一个已存在的 ChromaDB collection，返回 MemoryStore。"""
    import chromadb
    from memory_agent.memory.store import MemoryStore, COLLECTION_NAME

    store = MemoryStore.__new__(MemoryStore)
    store.client = chromadb.PersistentClient(path=persist_dir)
    store.collection = store.client.get_collection(COLLECTION_NAME)
    store._next_id = store.collection.count()
    return store


# ═══════════════════════════════════════════════════════════════
#  ingest 核心：每段对话一个独立 Agent，状态完全隔离
# ═══════════════════════════════════════════════════════════════
def ingest_all():
    import shutil
    from memory_agent.agent.controller import MemoryAgent

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    total = len(eval_set)

    # 清空旧的缓存
    if os.path.exists(SAVED_DB_ROOT):
        shutil.rmtree(SAVED_DB_ROOT)
    os.makedirs(SAVED_DB_ROOT, exist_ok=True)

    all_personas = {}

    for i, sample in enumerate(eval_set):
        sample_id = sample["sample_id"]
        sessions_count = len(sample["conversation"]["sessions"])
        print(f"\n[ingest] ({i+1}/{total}) {sample_id}：正在导入 "
              f"{sessions_count} 个 session ...")

        # 每段对话 new 一个新的 Agent，状态完全隔离
        agent = MemoryAgent()
        agent.ingest(sample["conversation"])

        # ── 持久化该 conversation 的 ChromaDB ──
        conv_db_dir = os.path.join(SAVED_DB_ROOT, sample_id)
        if os.path.exists(conv_db_dir):
            shutil.rmtree(conv_db_dir)
        if os.path.exists(CHROMA_DB_DIR):
            shutil.copytree(CHROMA_DB_DIR, conv_db_dir)

        # ── 收集 persona ──
        all_personas[sample_id] = agent.persona_summaries
        print(f"[ingest] {sample_id} 完成，persona: {len(agent.persona_summaries)} 字符")

    # 持久化 persona summaries
    with open(PERSONA_FILE, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, ensure_ascii=False, indent=2)
    print(f"\n[ingest] persona_summaries 已保存到 {PERSONA_FILE}")

    print(f"[ingest] 全部完成，共 {total} 段对话。")


ANSWER_PROMPT = """
You are a fact extractor. Your task is to find the exact answer from the Memory Logs.
[Persona Profiles]
{persona_context}

[Memory Logs]
{memory_context}

Question: {question}

RULES:
1. ONLY use the information directly written in the memories.
2. If the question asks for a specific fact (e.g., date, name, item), find the sentence that exactly answers it and copy that fact.
3. Output ONLY the answer, no extra words.

Answer:"""


FILTER_PROMPT = """
Your task is to keep only the memories that help answer the question. Delete all others.

Memories:
{memory_context}

Question: {question}

Rules:
- If a memory contains information directly related to the question, keep it.
- If it does not, delete it.
- Output only the kept memories, exactly as they appear. Do not add anything. Do not explain.
"""


# ═══════════════════════════════════════════════════════════════
#  run：有缓存就走缓存，没缓存先 ingest
# ═══════════════════════════════════════════════════════════════
def run_all(concurrency: int):
    from memory_agent.memory.writer import MemoryWriter
    from memory_agent.memory.retriever import MemoryRetriever
    from llm_client import LLMClient

    # ── 检查缓存 ──
    cached = os.path.exists(SAVED_DB_ROOT) and os.path.exists(PERSONA_FILE)

    if not cached:
        print("[run] 未找到数据库缓存，开始导入数据...")
        ingest_all()

    # 加载 persona 字典
    with open(PERSONA_FILE, "r", encoding="utf-8") as f:
        all_personas = json.load(f)

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        eval_set = json.load(f)

    # ── 共享组件（所有 conversation 复用）──
    writer = MemoryWriter()
    llm = LLMClient()

    # ── 为每个 conversation 构建独立组件（各自指向独立的 DB 目录）──
    print("[run] 正在加载各 conversation 的数据库...")
    conv_map = {}  # sample_id -> dict(store, retriever, persona)

    for sample in eval_set:
        sample_id = sample["sample_id"]
        conv_db_dir = os.path.join(SAVED_DB_ROOT, sample_id)

        if not os.path.exists(conv_db_dir):
            print(f"  {sample_id}: 未找到数据库，跳过")
            continue

        # 直接从该 conversation 的独立目录加载 DB，不走 MemoryAgent
        store = _load_store_from_disk(conv_db_dir)
        retriever = MemoryRetriever(
            store=store,
            embed_model=writer.embed_model,
            writer=writer,
        )

        conv_map[sample_id] = {
            "store": store,
            "retriever": retriever,
            "persona": all_personas.get(sample_id, ""),
        }
        print(f"  {sample_id}: {store.collection.count()} 条记忆")

    # ── 构建所有 QA 列表 ──
    all_qa = []
    for sample in eval_set:
        sample_id = sample["sample_id"]
        for qa in sample["qa_list"]:
            all_qa.append({
                **qa,
                "sample_id": sample_id,
            })

    total_qa = len(all_qa)
    predictions = [None] * total_qa
    qa_done = [0]

    def _answer_one(idx: int):
        qa = all_qa[idx]
        sample_id = qa["sample_id"]
        ctx = conv_map[sample_id]
        t0 = time.time()
        try:
            # 手动执行 answer 逻辑（与 controller.answer 一致）
            query_embedding = writer.embed_text(qa["question"])
            results = ctx["retriever"].retrieve(
                qa["question"], query_embedding, top_k=40
            )

            if not results:
                pred = "unknown"
            else:
                # ── 格式化：仅去掉开头的 [Conversation Date: xxx] ──
                context_lines = []
                for mem, _ in results:
                    text = mem.text_description
                    # 去掉开头的日期标签 "[Conversation Date: ...]"
                    cleaned = re.sub(r'^\[Conversation Date: [^\]]+\]\s*', '', text)
                    context_lines.append(f"- {cleaned}")
                context = "\n".join(context_lines)

                # ── 第一阶段：精简记忆单元 ──
                filter_prompt = FILTER_PROMPT.format(
                    memory_context=context,
                    question=qa["question"],
                )
                filtered_context = llm.generate(filter_prompt, max_tokens=512).strip()

                # ── 第二阶段：用精简后的记忆回答问题 ──
                prompt = ANSWER_PROMPT.format(
                    memory_context=filtered_context,
                    question=qa["question"],
                    persona_context=ctx["persona"],
                )
                pred = llm.generate(prompt, max_tokens=64).strip()
            err = None
        except Exception as e:
            pred = ""
            err = f"answer_failed: {e}"
        latency = time.time() - t0

        result = {
            "qa_id": qa["qa_id"],
            "question": qa["question"],
            "reference": qa["answer"],
            "category": qa["category"],
            "category_name": qa["category_name"],
            "prediction": str(pred).strip(),
            "error": err,
            "latency_sec": round(latency, 3),
        }
        with print_lock:
            predictions[idx] = result
            qa_done[0] += 1
            if qa_done[0] % 10 == 0 or qa_done[0] == total_qa:
                print(f"[run] 已回答 {qa_done[0]}/{total_qa} 题")

    print(f"\n[run] 开始并发回答 {total_qa} 题（并发数={concurrency}）...")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(_answer_one, i) for i in range(total_qa)]
        for fut in as_completed(futures):
            fut.result()

    # 剔除 None
    predictions = [p for p in predictions if p is not None]

    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"[run] 预测结果已保存 -> {PREDICTIONS_FILE}")

    # ── 运行 Judge ──
    print("[run] 正在运行 Judge 评测...")
    run_judge_subprocess()

    # ── 写 MD ──
    write_results_md()

    # ── 归档结果 ──
    archive_results()

    print(f"[run] 全部完成！结果见 {RESULTS_MD_FILE}")


# ═══════════════════════════════════════════════════════════════
#  Judge 子进程
# ═══════════════════════════════════════════════════════════════
def run_judge_subprocess():
    import subprocess

    eval_kit_dir = os.path.join(PROJECT_ROOT, "eval_kit", "eval_kit")
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = PROJECT_ROOT + (os.pathsep + pythonpath if pythonpath else "")

    # 强制 Judge 使用云端 deepseek-v4-flash，不受本地 vllm 环境影响
    env["LLM_MODEL"] = "deepseek-v4-flash"
    env["LLM_BASE_URL"] = "https://api.deepseek.com/v1"
    env["LLM_API_KEY"] = "sk-97a50281d2b442ab84408c6c9d73be41"

    cmd = [
        sys.executable, os.path.join(eval_kit_dir, "run_judge.py"),
        "--predictions", PREDICTIONS_FILE,
        "--output", JUDGE_RESULTS_FILE,
        "--num_workers", "4",
    ]
    subprocess.run(cmd, check=True, env=env)
    print(f"[run] Judge 结果已保存 -> {JUDGE_RESULTS_FILE}")


# ═══════════════════════════════════════════════════════════════
#  结果写入 MD
# ═══════════════════════════════════════════════════════════════
def write_results_md():
    with open(JUDGE_RESULTS_FILE, encoding="utf-8") as f:
        results = json.load(f)

    overall = results["overall"]
    by_cat = results["by_category"]
    model = results.get("judge_model", "unknown")

    lines = []
    lines.append(f"# 评测结果")
    lines.append(f"")
    lines.append(f"- **生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Judge 模型**：{model}")
    lines.append(f"- **总题数**：{overall['n']}")
    lines.append(f"")
    lines.append(f"## 总体指标")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|----|")
    lines.append(f"| 得分 (Score) | {overall['score']:.3f} |")
    lines.append(f"| F1 | {overall['f1']:.3f} |")
    lines.append(f"| EM | {overall['em']:.3f} |")
    lines.append(f"| 平均回答耗时 | {overall.get('avg_latency_sec', 0)}s |")
    lines.append(f"")
    lines.append(f"## 分类别结果")
    lines.append(f"")
    lines.append(f"| 类别 | 题数 | 得分 | F1 | EM | 正确 | 部分 | 错误 |")
    lines.append(f"|------|------|------|----|----|------|------|------|")
    for name in sorted(by_cat.keys()):
        d = by_cat[name]
        lines.append(
            f"| {name} | {d['n']} | {d['score']:.3f} | {d['f1']:.3f} | "
            f"{d['em']:.3f} | {d.get('correct', 0)} | "
            f"{d.get('partial', 0)} | {d.get('wrong', 0)} |"
        )
    lines.append(f"| **总体** | {overall['n']} | {overall['score']:.3f} | "
                 f"{overall['f1']:.3f} | {overall['em']:.3f} | | | |")

    with open(RESULTS_MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[run] 结果 MD 已写入 -> {RESULTS_MD_FILE}")


# ═══════════════════════════════════════════════════════════════
#  归档结果到 文档/结果汇总
# ═══════════════════════════════════════════════════════════════
def archive_results():
    import shutil

    os.makedirs(SUMMARY_DIR, exist_ok=True)

    # ── 1. 确定当前测试编号 ──
    test_num = 5
    if os.path.exists(SUMMARY_DIR):
        existing = [f for f in os.listdir(SUMMARY_DIR) if f.startswith("test") and f.endswith("_answer.json")]
        nums = []
        for f in existing:
            try:
                num_str = f[len("test"):-len("_answer.json")]
                nums.append(int(num_str))
            except ValueError:
                pass
        if nums:
            test_num = max(nums) + 1

    # ── 2. 保存 predictions.json -> testX_answer.json ──
    answer_json = os.path.join(SUMMARY_DIR, f"test{test_num}_answer.json")
    shutil.copy2(PREDICTIONS_FILE, answer_json)
    print(f"[run] 预测结果已归档 -> {answer_json}")

    # ── 3. 生成符合格式的汇总内容追加到 result_summary.md ──
    with open(JUDGE_RESULTS_FILE, encoding="utf-8") as f:
        jr = json.load(f)

    overall = jr["overall"]
    by_cat = jr["by_category"]
    model = jr.get("judge_model", "unknown")

    lines = []
    lines.append(f"## test{test_num}")
    lines.append(f"")
    lines.append(f"- **生成时间**：{time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- **Judge 模型**：{model}")
    lines.append(f"- **总题数**：{overall['n']}")
    lines.append(f"")
    lines.append(f"平均回答耗时:  {overall.get('avg_latency_sec', 0)}s ")
    lines.append(f"")
    lines.append(f"| 类别 | 题数 | 得分 | F1 | EM | 正确 | 部分 | 错误 |")
    lines.append(f"|------|------|------|----|----|------|------|------|")
    for name in sorted(by_cat.keys()):
        d = by_cat[name]
        lines.append(
            f"| {name} | {d['n']} | {d['score']:.3f} | {d['f1']:.3f} | "
            f"{d['em']:.3f} | {d.get('correct', 0)} | "
            f"{d.get('partial', 0)} | {d.get('wrong', 0)} |"
        )
    lines.append(f"| **总体** | {overall['n']} | {overall['score']:.3f} | "
                 f"{overall['f1']:.3f} | {overall['em']:.3f} | | | |")
    lines.append("")

    with open(SUMMARY_MD, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[run] 评测结果已追加 -> {SUMMARY_MD}")


# ═══════════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="快速测试脚本（缓存 ingest 数据，循环调参不用重新导入）")
    parser.add_argument("-ingest", action="store_true",
                        help="仅导入数据并持久化 DB")
    parser.add_argument("-run", action="store_true",
                        help="全量测试（有缓存则跳过 ingest）")
    parser.add_argument("-change", type=int, default=3,
                        help="修改 answer 并发数（默认 3）")
    args = parser.parse_args()

    if not args.ingest and not args.run:
        print("请指定 -ingest 或 -run 命令。用 -h 查看帮助。")
        sys.exit(1)

    if args.ingest:
        ingest_all()
    elif args.run:
        run_all(concurrency=args.change)
