"""
本文件提供四个子工具，用于记忆的提取、评分、向量化和更新记录最新时间戳
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer
import numpy as np


# 时间解析 
# 解析为 Unix 时间戳，作为 Recency 计算的基准
_DATE_FORMAT = "%I:%M %p on %d %B, %Y"

def _parse_date_time(date_str: str) -> float:
    return datetime.strptime(date_str, _DATE_FORMAT).timestamp()


# 调用 LLM 对记忆片段进行重要性评分（1-10分）
# 评分结果存入 MemoryUnit.importance_score 字段
IMPORTANCE_PROMPT = """Rate the importance of the following memory for understanding a person's long-term profile and life trajectory on a scale of 1 to 10.

Scoring Guidelines:
- 1-3: Mundane, transient details or casual remarks (e.g., everyday routines, passing thoughts).
- 4-6: Informative facts, general preferences, or ongoing activities (e.g., hobbies, current job, weekend plans).
- 7-8: Significant personal traits, deep emotions, strong relationships, or clear future goals (e.g., planning to adopt, deep support from friends).
- 9-10: Pivotal life events, core identity aspects, or major achievements (e.g., transitioning gender, getting married, college acceptance).

Memory: {memory_text}

Return ONLY an integer between 1 and 10. Do not include any other text.
"""


#调用llm对记忆文本进行整理

class MemoryWriter:
    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 使用 sentence-transformers 加载 embedding 模型
        self.embed_model = SentenceTransformer(embed_model_name)
        self.mark_agent = LLMClient()
        self.latest_time = None

    # 输入：完整的多会话对话 conversation dict
    # 输出：list[dict]，每个 dict 包含 {text, timestamp, type}
    def extract_memories(self, conversation: dict) -> list[dict]:
        sessions = conversation["sessions"]
        num_sessions = len(sessions)

        date_ts_list = [_parse_date_time(s["date_time"]) for s in sessions]
        date_str_list = [s["date_time"] for s in sessions]
        self.latest_time = max(date_ts_list)

        lines = []
        for sess in sessions:
            line = "".join(f"{turn['speaker']} say:{turn['text']} " for turn in sess["turns"])
            lines.append(line)

        executor_results = [None] * num_sessions

        def _process_one(idx: int):
            llm = LLMClient()
            prompt = (f"""You are an expert dialogue memory extractor. Your task is to extract comprehensive, atomic memory units from the following conversation between speakers.
                        Context:
                        - Current Session Time: {date_str_list[idx]}
                        Extraction Rules:
                        1. Coreference Resolution (CRITICAL): NEVER use pronouns (I, you, he, she, it, they). Replace all pronouns with the explicit names of the speakers or specific entities. 
                        (e.g., Instead of "She likes painting", write "Melanie likes painting".)
                        2. Atomic Facts: Each extracted statement must contain ONLY ONE core idea, making it suitable for vector retrieval.
                        3. Comprehensiveness: You must extract information across the following dimensions:
                        - Personal Background & Identity (e.g., LGBTQ status, job, family members)
                        - Events & Experiences (What did they do? Include time anchors based on the Current Session Time)
                        - Opinions, Preferences & Sentiments (e.g., likes, dislikes, how they feel about something)
                        - Future Plans & Intentions (e.g., careers, trips, adoptions)
                        4. Temporal Anchoring: If a relative time is mentioned (e.g., "yesterday", "last year"), translate it to an absolute or roughly absolute time context based on the Current Session Time.
                        5. Conciseness: Keep statements concise but fully context-independent.

                        Output Format:
                        Output ONLY a raw list of strings, one statement per line. Do not use numbering, bullet points, or extra explanations.

                        Conversation:\n{lines[idx]}\n""")
            response = llm.generate(prompt, max_tokens=2048, temperature=0).strip()
            memories = []
            if response:
                for stmt in response.split("\n"):
                    stmt = stmt.strip()
                    if not stmt:
                        continue
                    if stmt[0].isdigit() and (len(stmt) < 3 or stmt[1] in ".、)）"):
                        stmt = stmt[2:].strip()
                    if len(stmt) < 5:
                        continue
                    memories.append({
                        "text": stmt,
                        "last_access_timestamp": date_ts_list[idx],
                        "creation_timestamp": date_ts_list[idx],
                        "type": "observation"
                    })
            return idx, memories

        with ThreadPoolExecutor(max_workers=num_sessions) as ex:
            futures = {ex.submit(_process_one, i): i for i in range(num_sessions)}
            for fut in as_completed(futures):
                idx, memories = fut.result()
                executor_results[idx] = memories

        all_memories = []
        for memories in executor_results:
            if memories:
                all_memories.extend(memories)
        return all_memories

    # 调用 LLM 对记忆文本进行重要性评分，返回 1-10 的整数
    # 解析失败时默认返回 5 分
    def score_importance(self, memory_text: str) -> int:
        prompt = IMPORTANCE_PROMPT.format(memory_text=memory_text)
        raw = self.mark_agent.generate(prompt, max_tokens=20, temperature=0.1).strip()
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
    
    # 更新 latest_time
    def update_latest_time(self, timestamp: float):
        self.latest_time = timestamp
    