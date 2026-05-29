"""
[规划对应] Agent Controller —— 主流程编排：ingest + answer + add_memory 流程

本文件实现了 Agent 的核心控制逻辑，对应 develop.md 中的：
- ingest 方法：接收对话 → 调用 add_memory（formatting → 评分 → 向量化 → 存库）
- answer 方法：接收问题 → 检索 → 拼接上下文 → LLM 生成答案
- 调用逻辑：ingest 时自动触发记忆提取与反思检查
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))

from llm_client import LLMClient
from memory_agent.memory.store import MemoryStore
from memory_agent.memory.writer import MemoryWriter
from memory_agent.memory.retriever import MemoryRetriever
from memory_agent.memory.updater import MemoryUpdater



ANSWER_PROMPT = """You are an assistant with access to memories from a past conversation between two people.
Answer the user's question using only information from the retrieved memories below.
Keep the answer short (a phrase or one sentence).
If the memories do not contain the answer, reply 'unknown'.

=== Retrieved memories ===
{context}
=== Question ===
{question}
"""


class MemoryAgent:
    def __init__(self, top_k: int = 5):
        self.llm = LLMClient()
        self.store = MemoryStore()
        self.writer = MemoryWriter()
        self.retriever = MemoryRetriever(store=self.store, embed_model=self.writer.embed_model)
        self.updater = MemoryUpdater(store=self.store, retriever=self.retriever, writer=self.writer)
        self.top_k = top_k


    def ingest(self, conversation: dict) -> None:
        # 从对话中提取记忆
        raw_memories = self.writer.extract_memories(conversation)
        if not raw_memories:
            return
        # 向量化工具
        texts = [m["text"] for m in raw_memories]
        embeddings = self.writer.embed_batch(texts)
        retriever.latest_time = self.writer.latest_time
        # 逐条评分 → 存库 → 累加 sum_score
        for i, mem in enumerate(raw_memories):
            importance = self.writer.score_importance(mem["text"])
            self.store.add(
                text=mem["text"],
                memory_type=mem["type"],
                importance_score=importance,
                embedding=embeddings[i],
                timestamp=mem["last_access_timestamp"],
            )
            # 维护 sum_score，超过 150 会自动触发 reflect
            self.updater.add_importance(importance)

 
    def answer(self, question: str) -> str:
        # 对 question 做向量化
        query_embedding = self.writer.embed_text(question)
        # 调用 retrieval 进行三因子检索
        results = self.retriever.retrieve(question, query_embedding, top_k=self.top_k)

        if not results:
            return "unknown"
        context = "\n".join(f"- {mem.text_description}" for mem, _ in results)
        prompt = ANSWER_PROMPT.format(context=context, question=question)
        # 喂给 LLM 生成答案
        return self.llm.generate(prompt, max_tokens=64).strip()
