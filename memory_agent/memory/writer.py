"""
本文件提供四个子工具，用于记忆的提取、评分、向量化和更新记录最新时间戳
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))
from llm_client import LLMClient
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


# 基础大语言模型配置
llm = ChatOpenAI(
    api_key=os.getenv("AGENT_API_KEY"),
    base_url=os.getenv("AGENT_URL"),
    model=os.getenv("AGENT_MODEL"),
    temperature=0
)

parser = StrOutputParser()
chain = llm | parser

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
Attention:If the category is Reflection, the importance score should be higher than 7.
Memory: {memory_text}
Category: {category}
Return ONLY an integer between 1 and 10. Do not include any other text.
"""


# 调用llm总结人物脉络
Summary_PROMPT = """You are an expert psychological profiler and behavioral analyst. Your task is to analyze a comprehensive list of extracted memory facts from conversations and construct a dense, predictive Persona Profile for each main character.

These profiles will be used by an AI to answer implicit, multi-hop questions (e.g., deducing if someone would like a specific music genre based on their hobbies, or if they would visit a national park based on their outdoor activities).

Please synthesize the facts and extract deep insights covering the following dimensions for each character:
1. Core Identity & Life Stage (marital/family status, major ongoing life events)
2. Career & Ambitions (current job, future goals, professional motivations)
3. Lifestyle & Preferences (abstract specific activities into broader concepts, e.g., playing violin -> affinity for classical music; camping -> loves the outdoors)
4. Values & Beliefs (religiousness, core motivations, social stances like LGBTQ+ advocacy)
5. Social Circle & Nicknames (key relationships, explicitly note any aliases or nicknames used)

CRITICAL FORMATTING CONSTRAINTS:
1. You MUST return the output EXACTLY in the format: `Name: Description`
2. Each character's profile MUST be on a single continuous line. DO NOT use line breaks (\n) within a character's description.
3. Separate different characters with a single line break (\n).
4. Length limit: Keep the description for each character highly dense and concise, strictly under 400 words (approx. 500 tokens) per character.
5. Output ONLY the requested format. No markdown formatting (like **Name**), no introductory text, and no concluding remarks.

Example Output:
Caroline: A transgender woman and strong LGBTQ+ advocate who is currently looking to adopt a child as a single parent. She wants to be a counselor to support mental health. She values inclusivity, has a close support network of friends from her home country, and is not deeply religious but driven by empathy.
Melanie: A married mother of two who has been with her husband for 5 years. She is highly supportive of the LGBTQ+ community despite being in a heterosexual marriage. She loves the outdoors, frequently goes camping and running, and expresses her creativity through painting nature and playing the violin, indicating an affinity for classical music and arts.

================
Input Memory Facts:
{all_memory_facts}
"""




#调用llm对记忆文本进行整理

class MemoryWriter:
    def __init__(self, embed_model_name: str = "BAAI/bge-small-zh-v1.5"):
        # 使用 sentence-transformers 加载 embedding 模型
        self.embed_model = SentenceTransformer(embed_model_name)
        self.agent = chain
        self.latest_time = None
        self.llm_client = LLMClient()

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
            line = "\n".join(f"{turn['speaker']}: {turn['text']}" for turn in sess["turns"])
            lines.append(line)

        executor_results = [None] * num_sessions
        print_lock = threading.Lock()
        max_workers = min(num_sessions, 40)

        
        def _process_one(idx: int):
            prompt = (
                "You are an expert dialogue memory extractor. Your task is to extract comprehensive, "
            "but strictly non-redundant memory units from the following conversation.\n\n"
            "Context:\n"
            f"- Current Session Time: {date_str_list[idx]}\n\n"
            "Extraction Rules:\n"
            "1. Coreference Resolution (CRITICAL): NEVER use pronouns (I, you, he, she, it, they). "
            "Replace all pronouns with the explicit names of the speakers or specific entities.\n"
            "2. Atomic but Cohesive: Do NOT over-fragment. 'Melanie painted a lake sunrise last year' "
            "is ONE fact. Do not split it into three separate statements.\n"
            "3. Deduplication & Filtering (CRITICAL): Do NOT repeat the same fact in different ways. "
            "Ignore greetings, pleasantries, passing agreements, and meaningless filler talk.\n"
            "4. Comprehensiveness: Extract information across: Personal Background, Events (with absolute time), "
            "Opinions/Sentiments, and Future Plans.\n"
            "5. Temporal Anchoring: Translate relative times (e.g., 'yesterday', 'last year') "
            "to absolute context based on Current Session Time.\n\n"
            "Constraint: Aim for high-quality, dense facts. A typical session should yield 10 to 25 statements. "
            "DO NOT exceed 30 statements unless absolutely necessary.\n\n"
            "Output ONLY raw statements, one per line. No numbering, no bullet points, no extra text.\n\n"
            "One statement must be no more than 20 characters long.\n\n"
            f"Conversation:\n{lines[idx]}\n"
            )
            sessions_str = f"[session {idx+1}/{num_sessions}]"

            response = ""
            max_retries = 5
            for retry in range(max_retries):
                try:
                    response = self.agent.invoke(prompt).strip()
                except Exception as e:
                    if retry < max_retries - 1:
                        with print_lock:
                            print(f"  {sessions_str} 调用异常，重试 {retry+2}/{max_retries}: {e}")
                        time.sleep(2 ** retry)
                        continue
                    with print_lock:
                        print(f"  {sessions_str} 提取 0 条记忆 (LLM 调用失败)")
                    return idx, []

                if response:
                    break
                if retry < max_retries - 1:
                    with print_lock:
                        print(f"  {sessions_str} 返回为空，重试 {retry+2}/{max_retries}")
                    time.sleep(2 ** retry)

            memories = []
            if response:
                for stmt in response.split("\n"):
                    stmt = stmt.strip()
                    if not stmt:
                        continue
                    while stmt and stmt[0].isdigit():
                        stmt = stmt[1:]
                    if stmt and stmt[0] in ".、)）":
                        stmt = stmt[1:].strip()
                    if len(stmt) < 5:
                        continue
                    memories.append({
                        "text": stmt,
                        "last_access_timestamp": date_ts_list[idx],
                        "creation_timestamp": date_ts_list[idx],
                        "type": "observation"
                    })

            with print_lock:
                print(f"  {sessions_str} 提取 {len(memories)} 条记忆")
            return idx, memories

        print(f"  正在并发处理 {num_sessions} 个 session（max_workers={max_workers}）...")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_process_one, i): i for i in range(num_sessions)}
            for fut in as_completed(futures):
                idx, memories = fut.result()
                executor_results[idx] = memories
        print(f"  全部 {num_sessions} 个 session 处理完成")

        all_memories = []
        for memories in executor_results:
            if memories:
                all_memories.extend(memories)
        return all_memories

    # 调用 LLM 对记忆文本进行重要性评分，返回 1-10 的整数
    # 解析失败时默认返回 5 分
    def score_importance(self, memory_text: str, category: str) -> int:
        prompt = IMPORTANCE_PROMPT.format(memory_text=memory_text, category=category)
        raw = self.agent.invoke(prompt).strip()
        try:
            # 从原始输出中提取整数评分
            score = int("".join(c for c in raw if c.isdigit()))
            return max(1, min(10, score))
        except ValueError:
            return 5

    # 并发批量评分，max_workers 控制最大并发数
    # 每条记忆独立调用 LLM 打分，大幅缩短总体耗时
    def score_importance_batch(self, texts: list[str], category: str, max_workers: int = 500) -> list[int]:
        num = len(texts)
        scores = [5] * num
        print_lock = threading.Lock()
        print(f"  正在并发评分 {num} 条记忆（max_workers={min(num, max_workers)}）...")
        completed = [0]

        def _score_one(idx: int):
            try:
                prompt = IMPORTANCE_PROMPT.format(memory_text=texts[idx], category=category)
                raw = self.agent.invoke(prompt).strip()
                score = int("".join(c for c in raw if c.isdigit()))
                scores[idx] = max(1, min(10, score))
            except Exception:
                scores[idx] = 5
            with print_lock:
                completed[0] += 1
                if completed[0] % 100 == 0 or completed[0] == num:
                    print(f"  评分进度: {completed[0]}/{num}")

        with ThreadPoolExecutor(max_workers=min(num, max_workers)) as ex:
            futures = [ex.submit(_score_one, i) for i in range(num)]
            for fut in as_completed(futures):
                fut.result()
        print(f"  评分完成: {num}/{num}")
        return scores

 
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

    # 基于所有记忆事实，调用高级模型生成两个人物总结
    # 返回格式如 "Caroline: ...\nMelanie: ..."
    def summarize_personas(self, all_memory_facts: str) -> str:
        prompt = Summary_PROMPT.format(all_memory_facts=all_memory_facts)
        result = self.agent.invoke(prompt).strip()
        print(f"  人物总结生成完成，长度: {len(result)} 字符")
        return result
    