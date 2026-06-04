"""
本文件实现了 Agent 的控制逻辑：
ingest 方法：接收对话 → 调用 add方法（formatting → 评分 → 向量化 → 存库）
answer 方法：接收问题 → 检索 → 拼接上下文 → LLM 生成答案
调用逻辑：ingest 时自动触发记忆提取与反思检查
"""

import sys
import os
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
    retrieval_top_k: int = 50

    # ── 反思触发参数 ──
    reflection_threshold: int = 100
    """累计新增记忆的 importance_score 超过此值触发一次反思"""

    # ── 反思流程参数 ──
    reflection_memory_limit: int = 60
    """反思时取最近多少条记忆作为分析素材"""

    reflection_question_count: int = 3
    """反思时生成几个高层次问题"""

    reflection_insight_per_q: int = 3
    """每个问题生成几条高层次洞察（3问题 × 3洞察 = 共9条反思记忆）"""

    reflection_retrieval_top_k: int = 30
    """反思检索时为每个问题检索多少条相关记忆"""

    # ── 反思 LLM 生成参数 ──
    reflection_max_tokens: int = 256
    """反思时 LLM 单次生成的最大 token 数"""

    reflection_temperature: float = 0
    """反思时 LLM 的温度参数，越大输出越随机"""


ANSWER_PROMPT = """You are an assistant with access to memories from a past conversation between two people.
Answer the user's question using only information from the retrieved memories below.
Keep the answer short (a phrase or one sentence).
If the memories do not contain the answer, reply 'unknown'.
Instructions:
1. Do not expect the answer to be explicitly written in the facts. You MUST make logical deductions.
2. If the question asks about a specific preference (e.g., a music genre, a travel destination), infer the answer based on the person's hobbies, past actions, and general persona.
3. If the question asks about future plans (e.g., moving), consider their current major life events (e.g., adopting, new job).
4. Only output "unknown" if the memories provide absolutely NO clues or related concepts whatsoever.
5. Provide a brief rationale before your final short answer.
=== Character Profiles ===
{persona_summaries}
=== Retrieved memories ===
{context}
=== Question ===
{question}
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

        # 在所有记忆提取并存储完成后，基于所有记忆生成人物总结
        all_memories = self.store.get_all()
        if all_memories:
            all_facts = "\n".join(m.text_description for m in all_memories)
            self.persona_summaries = self.writer.summarize_personas(all_facts)

 
    def answer(self, question: str) -> str:
        # 对 question 做向量化
        query_embedding = self.writer.embed_text(question)
        # 调用 retrieval 进行双因子检索
        results = self.retriever.retrieve(question, query_embedding, top_k=self.settings.retrieval_top_k)

        if not results:
            return "unknown"
        context = "\n".join(f"- {mem.text_description}" for mem, _ in results)
        prompt = ANSWER_PROMPT.format(context=context, question=question, persona_summaries=self.persona_summaries)
        # 喂给 LLM 生成答案
        return self.llm.generate(prompt, max_tokens=256).strip()
