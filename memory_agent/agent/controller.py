"""
本文件实现了 Agent 的控制逻辑：
ingest 方法：接收对话 → 调用 add方法（formatting → 评分 → 向量化 → 存库）
answer 方法：接收问题 → 检索 → 拼接上下文 → LLM 生成答案
调用逻辑：ingest 时自动触发记忆提取与反思检查
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))

from llm_client import LLMClient
from memory_agent.memory.store import MemoryStore
from memory_agent.memory.writer import MemoryWriter
from memory_agent.memory.retriever import MemoryRetriever
from memory_agent.memory.updater import MemoryUpdater



# 所有影响性能的可调参数集中于此，实验时一键修改
class Settings:
    # ── 检索参数 ──
    """answer 阶段检索多少条相关记忆用于生成答案"""
    retrieval_top_k: int = 30

    # ── 反思触发参数 ──
    #修改为450
    reflection_threshold: int = 45000
    """累计新增记忆的 importance_score 超过此值触发一次反思"""

    # ── 反思流程参数 ──
    #修改为90
    reflection_memory_limit: int = 90
    """反思时取最近多少条记忆作为分析素材"""

    reflection_question_count: int = 3
    """反思时生成几个高层次问题"""

    reflection_insight_per_q: int = 3
    """每个问题生成几条高层次洞察（3问题 × 3洞察 = 共9条反思记忆）"""

    reflection_retrieval_top_k: int = 50
    """反思检索时为每个问题检索多少条相关记忆"""

    # ── 反思 LLM 生成参数 ──
    reflection_max_tokens: int = 256
    """反思时 LLM 单次生成的最大 token 数"""

    reflection_temperature: float = 0
    """反思时 LLM 的温度参数，越大输出越随机"""


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



class MemoryAgent:
    def __init__(self, top_k: int = None, settings: Settings = None):
        if settings is None:
            settings = Settings()
        if top_k is not None:
            settings.retrieval_top_k = top_k
        self.settings = settings

        self.llm = LLMClient()
        self.store = MemoryStore()
        self.writer = MemoryWriter()
        self.persona_summaries = ""
        self.retriever = MemoryRetriever(
            store=self.store,
            embed_model=self.writer.embed_model,
            writer=self.writer,
        )
        self.updater = MemoryUpdater(
            store=self.store,
            retriever=self.retriever,
            writer=self.writer,
            reflection_threshold=settings.reflection_threshold,
            reflection_memory_limit=settings.reflection_memory_limit,
            reflection_question_count=settings.reflection_question_count,
            reflection_insight_per_q=settings.reflection_insight_per_q,
            reflection_retrieval_top_k=settings.reflection_retrieval_top_k,
            reflection_max_tokens=settings.reflection_max_tokens,
            reflection_temperature=settings.reflection_temperature,
        )


    def ingest(self, conversation: dict) -> None:
        # 从对话中提取记忆
        raw_memories = self.writer.extract_memories(conversation)
        if not raw_memories:
            return
        # 向量化工具
        texts = [m["text"] for m in raw_memories]
        embeddings = self.writer.embed_batch(texts)
        # 并发评分 → 存库 → 累加 sum_score
        scores = self.writer.score_importance_batch(texts, category="observation", max_workers=500)
        total_memories = len(raw_memories)
        for i, mem in enumerate(raw_memories):
            importance = scores[i]
            self.store.add(
                text=mem["text"],
                memory_type=mem["type"],
                importance_score=importance,
                embedding=embeddings[i],
                timestamp=mem["last_access_timestamp"],
            )
            # 维护 sum_score，超过 150 会自动触发 reflect
            self.updater.add_importance(importance)
            if (i + 1) % 30 == 0 or (i + 1) == total_memories:
                print(f"  已评分并存入 {i+1}/{total_memories} 条记忆, 当前累计重要性分数: {self.updater._accumulated_importance:.1f}")

        # 基于当前 conversation 提取到的记忆单元生成人物总结
        if texts:
            all_facts = "\n".join(texts)
            self.persona_summaries = self.writer.summarize_personas(all_facts)

 
    def answer(self, question: str) -> str:
        # 对 question 做向量化
        query_embedding = self.writer.embed_text(question)
        # 调用 retrieval 进行三因子检索
        results = self.retriever.retrieve(question, query_embedding, top_k=self.settings.retrieval_top_k)

        if not results:
            return "unknown"

        # ── 格式化：仅去掉开头的 [Conversation Date: xxx] ──
        context_lines = []
        for mem, _ in results:
            text = mem.text_description
            cleaned = re.sub(r'^\[Conversation Date: [^\]]+\]\s*', '', text)
            context_lines.append(f"- {cleaned}")
        context = "\n".join(context_lines)

        # ── 第一阶段：精简记忆单元 ──
        filter_prompt = FILTER_PROMPT.format(
            memory_context=context,
            question=question,
        )
        filtered_context = self.llm.generate(filter_prompt, max_tokens=512).strip()

        # ── 第二阶段：用精简后的记忆回答问题 ──
        prompt = ANSWER_PROMPT.format(
            memory_context=filtered_context,
            question=question,
            persona_context=self.persona_summaries,
        )
        return self.llm.generate(prompt, max_tokens=64).strip()
