"""
[规划对应] 三重检索 retrieval 方法

本文件实现了基于三因子打分的记忆检索，对应 develop.md 中的 retrieval 方法：
- 维度 A：Recency（近期性）—— 指数衰减函数
- 维度 B：Importance（重要性）—— LLM 打分
- 维度 C：Relevance（相关性）—— 余弦相似度
- Min-Max 归一化后加权求和，排序取 top_k
"""

import time
from typing import Optional
import numpy as np
from memory_agent.memory.store import MemoryStore, MemoryUnit


# [规划对应] 维度 A：Recency 衰减系数
# 论文采用指数衰减函数：decay_factor^Δt，其中 Δt 是距上次访问的小时数
DECAY_FACTOR = 0.995


class MemoryRetriever:
    def __init__(self, store: MemoryStore, embed_model=None):
        self.store = store
        self.embed_model = embed_model

    # ── 维度 A：Recency（近期性 / 衰减度）──────────────────────────
    # [规划对应] 越近发生或越近被访问的记忆，得分越高
    # 公式：0.995 ^ Δt，Δt = (当前时间 - last_access_timestamp) / 3600（转为小时）
    def _recency_score(self, mem: MemoryUnit, current_time: float) -> float:
        delta_t_hours = (current_time - mem.last_access_timestamp) / 3600.0
        return DECAY_FACTOR ** delta_t_hours

    # ── 维度 B：Importance（重要性）─────────────────────────────────
    # [规划对应] 读取记忆存入时 mark_agent 已打好的 importance_score 字段（1-10）
    def _importance_score(self, mem: MemoryUnit) -> float:
        return float(mem.importance_score)

    # ── 维度 C：Relevance（相关性）─────────────────────────────────
    # [规划对应] 计算 query embedding 与记忆 embedding 的余弦相似度
    # 由于 embedding 已做 normalize，使用 dot product 等价于 cosine similarity
    def _relevance_score(self, query_embedding: np.ndarray, mem: MemoryUnit) -> float:
        if mem.embedding is None or query_embedding is None:
            return 0.0
        sim = float(np.dot(mem.embedding, query_embedding))
        return max(0.0, sim)

    # ── Min-Max 归一化 ───────────────────────────────────────────
    # [规划对应] 三个维度的值域不同（Relevance 0-1, Importance 1-10），
    # 用 Min-Max Scaling 缩放到 [0, 1]，加 1e-5 防止除零
    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        return [(v - min_v) / (max_v - min_v + 1e-5) for v in values]

    # ── 综合检索 ─────────────────────────────────────────────────
    # [规划对应] retrieval 方法主入口
    # 1. 对所有记忆分别计算 Recency、Importance、Relevance 三个分数
    # 2. Min-Max 归一化到 [0,1]
    # 3. Final_Score = 1.0×Recency + 1.0×Importance + 1.0×Relevance
    # 4. 降序排序取 top_k，同时更新被检索记忆的 last_access_timestamp
    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 5,
                 current_time: Optional[float] = None) -> list[tuple[MemoryUnit, float]]:
        if current_time is None:
            current_time = time.time()

        memories = self.store.get_all()
        if not memories:
            return []

        recency_scores = [self._recency_score(m, current_time) for m in memories]
        importance_scores = [self._importance_score(m) for m in memories]
        relevance_scores = [self._relevance_score(query_embedding, m) for m in memories]

        norm_recency = self._min_max_normalize(recency_scores)
        norm_importance = self._min_max_normalize(importance_scores)
        norm_relevance = self._min_max_normalize(relevance_scores)

        final_scores = []
        for i, mem in enumerate(memories):
            score = norm_recency[i] + norm_importance[i] + norm_relevance[i]
            final_scores.append((mem, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for mem, score in final_scores[:top_k]:
            self.store.update_access_time(mem.id, current_time)
            results.append((mem, score))

        return results
