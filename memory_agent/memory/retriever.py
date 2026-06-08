"""
本文件实现了基于三因子打分的记忆检索
- 维度 A：BM25（关键词匹配）—— 权重 15%
- 维度 B：Importance（重要性）—— LLM 打分，权重 5%
- 维度 C：Relevance（语义相关性）—— 余弦相似度，权重 80%
- Min-Max 归一化后加权求和，排序取 top_k
"""

import numpy as np
from memory_agent.memory.store import MemoryStore, MemoryUnit
from rank_bm25 import BM25Okapi


class MemoryRetriever:
    def __init__(self, store: MemoryStore, embed_model=None, writer=None):
        self.store = store
        self.embed_model = embed_model
        self.writer = writer

    # ── 三个打分维度 ──

    def _importance_score(self, mem: MemoryUnit) -> float:
        return float(mem.importance_score)

    def _relevance_score(self, query_embedding: np.ndarray, mem: MemoryUnit) -> float:
        """余弦相似度（dot product，embedding 已 normalize）"""
        if mem.embedding is None or query_embedding is None:
            return 0.0
        sim = float(np.dot(mem.embedding, query_embedding))
        return max(0.0, sim)

    def _build_bm25(self, memories: list[MemoryUnit]) -> BM25Okapi:
        """基于所有记忆的 text_description 构建 BM25 索引"""
        tokenized_corpus = [
            m.text_description.lower().split() for m in memories
        ]
        return BM25Okapi(tokenized_corpus)

    def _bm25_scores(self, bm25: BM25Okapi, query: str) -> list[float]:
        """计算 query 与每条记忆的 BM25 分数"""
        tokenized_query = query.lower().split()
        return bm25.get_scores(tokenized_query).tolist()

    # ── 归一化 ──

    @staticmethod
    def _min_max_normalize(values: list[float]) -> list[float]:
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        return [(v - min_v) / (max_v - min_v + 1e-5) for v in values]

    # ── 三因子综合检索 ──

    def retrieve(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[MemoryUnit, float]]:
        memories = self.store.get_all()
        if not memories:
            return []

        # 计算三个维度的原始分数
        bm25_raw = self._bm25_scores(self._build_bm25(memories), query)
        importance_raw = [self._importance_score(m) for m in memories]
        relevance_raw = [self._relevance_score(query_embedding, m) for m in memories]

        # Min-Max 归一化到 [0, 1]
        norm_bm25 = self._min_max_normalize(bm25_raw)
        norm_importance = self._min_max_normalize(importance_raw)
        norm_relevance = self._min_max_normalize(relevance_raw)

        # 加权求和：BM25 15% + Importance 5% + Relevance 80%
        final_scores = []
        for i, mem in enumerate(memories):
            score = (0.15 * norm_bm25[i] +
                     0.05 * norm_importance[i] +
                     0.80 * norm_relevance[i])
            final_scores.append((mem, score))

        final_scores.sort(key=lambda x: x[1], reverse=True)
        return final_scores[:top_k]
