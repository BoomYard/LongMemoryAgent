"""
[规划对应] formatting 工具 + mark_agent 评分 + 向量化工具

本文件实现了记忆的提取、评分和向量化，对应 develop.md 中的三个子工具：
- formatting 工具：从原始对话中提取记忆，格式化为结构化字段
- mark_agent：调用 LLM 对记忆进行重要性评分（1-10分）
- 向量化工具：将文本转为 embedding 向量，供检索使用
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer
import numpy as np


# 调用 LLM 对记忆片段进行重要性评分（1-10分）
# 1 = 纯日常琐事（刷牙、叠被子），10 = 极其深刻的经历（分手、大学录取）
# 评分结果存入 MemoryUnit.importance_score 字段
IMPORTANCE_PROMPT = """On a scale of 1 to 10, where 1 is a purely mundane event (e.g., brushing teeth, making bed) and 10 is an extremely pivotal event (e.g., breakup, college acceptance), rate the likely poignancy/importance of the following memory.

Memory: {memory_text}
Just return the rating as an integer between 1 and 10.Don't contain any other text.
"""


class MemoryWriter:
    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 使用 sentence-transformers 加载 embedding 模型
        self.embed_model = SentenceTransformer(embed_model_name)
        self.mark_agent = LLMClient()
        

    # 输入：完整的多会话对话 conversation dict
    # 输出：list[dict]，每个 dict 包含 {text, timestamp, type}
    def extract_memories(self, conversation: dict) -> list[dict]:
        all_memories = []
        for sess in conversation["sessions"]:
            date_time_str = sess["date_time"]
            # 将字符串时间解析为 Unix 时间戳
            date_ts = _parse_date_time(date_time_str)
            # 合并一个session的文本为一个字符串
            line = ""
            for turn in sess["turns"]:
                line += turn["speaker"]+" say:"+turn["text"]+" "
        
            if line:
                all_memories.append({
                    "text": line,
                    "timestamp": date_ts,
                    "type": "observation",
                })
        return all_memories

    # 调用 LLM 对记忆文本进行重要性评分，返回 1-10 的整数
    # 解析失败时默认返回 5 分
    def score_importance(self, memory_text: str) -> int:
        prompt = IMPORTANCE_PROMPT.format(memory_text=memory_text)
        raw = self.mark_agent.generate(prompt, max_tokens=8, temperature=0.0).strip()
        try:
            # 从原始输出中提取整数评分
            score = int("".join(c for c in raw if c.isdigit()))
            return max(1, min(10, score))
        except ValueError:
            return 5

 
    # 使用 bge-small-zh-v1.5，输出为 float32 numpy 数组
    # 向量化工具：将单条文本转为归一化后的 embedding 向量，用于向量化query
    def embed_text(self, text: str) -> np.ndarray:
        vec = self.embed_model.encode([text], normalize_embeddings=True)
        return vec[0].astype(np.float32)

    # 向量化工具：一次编码多条文本，提高效率，用于向量化memory
    def embed_batch(self, texts: list[str]) -> np.ndarray:
        vecs = self.embed_model.encode(texts, normalize_embeddings=True)
        return np.array(vecs, dtype=np.float32)
