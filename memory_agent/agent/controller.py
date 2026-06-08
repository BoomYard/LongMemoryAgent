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
    retrieval_top_k: int = 30

    # ── 反思触发参数 ──
    #修改为450
    reflection_threshold: int = 450
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


ANSWER_PROMPT = """You are an expert psychological detective answering questions about a person based on their Persona Profile and Memory Logs.

[Persona Profiles]
{persona_context}

[Memory Logs]
{memory_context}

Question: {question}

CRITICAL INSTRUCTIONS:

1. NEVER SAY UNKNOWN: You are strictly forbidden from answering "unknown", "I don't know", or "not mentioned". You MUST make an educated guess based on the context.

2. DEDUCE PREFERENCES, BUT DO NOT INVENT FACTS: You should deduce implicit preferences (e.g., if they play violin, deduce they like classical music). HOWEVER, DO NOT hallucinate specific names, bands, brands, or exact dates that do not exist in the memories.

3. PARTIAL INFO IS BETTER THAN GUESSING BLINDLY: If you know the month but not the exact day, just state the month. If you know the category but not the specific item, state the category.

4. DO NOT LIST ALL MEMORIES: In your reasoning, focus only on the most relevant clues. However, if the question asks for a list (e.g., activities, artists), you MUST include ALL items supported by the memories.

5. TEMPORAL REASONING (CRITICAL):
- Each memory ends with a recording date, e.g., "[Recorded on: August 14th, 2023]". This is ONLY the day they talked about it.
- If the memory contains a specific event date (e.g., "on 11 August 2023"), THAT is the answer. NEVER use the "Recorded on" date if an event date exists.
- If an event is discussed in the past tense, it happened BEFORE the "Recorded on" date. (e.g., If they talk about a camping trip on June 27th, the trip happened "before June 27th").

6. OCCAM'S RAZOR (DIRECT FACTS FIRST): If you find a memory that directly matches the entities in the question (e.g., a specific item, a specific event), USE IT IMMEDIATELY. Do not overcomplicate or let other similar memories confuse you. Only use deduction if direct evidence is completely missing.

7. EVIDENCE HIERARCHY (NEW - OVERRIDES ALL OTHER RULES):
Memories are tagged by their reliability:
- [direct evidence]: Highly relevant and reliable. This is the strongest evidence.
- [related memories]: Somewhat relevant. Use with caution.
- [background]: Background context. Use only if no stronger evidence exists.
STRICT RULE: If any memory tagged as [direct evidence] directly answers the question, YOU MUST base your answer primarily on that memory. Other memories can only be used to supplement, never to contradict or override a [direct evidence].

8. MANDATORY OPPOSING-EVIDENCE CHECK (NEW):
Before finalizing your answer, you MUST actively search for any memory that suggests the OPPOSITE conclusion. If your initial reasoning points in one direction, force yourself to find at least one clue that might point the other way. This prevents confirmation bias. You will document this in the "opposing_clues" field of your output.

OUTPUT FORMAT:
You MUST output ONLY a valid JSON object. Do not include any other text, markdown formatting, or tags outside the JSON.

{
  "thinking": {
    "supporting_clues": "List the strongest evidence that supports your final answer, with brief reasoning.",
    "opposing_clues": "List any evidence that could point to a different answer, and explain why you ultimately did not choose it. If none exists, state 'No opposing evidence found in the provided memories.'",
    "final_reasoning": "Explain how you weighed the supporting and opposing clues to reach your final conclusion. If dates are involved, explicitly state which date you used and why."
  },
  "answer": "A concise, complete sentence that directly answers the question AND briefly includes the core reason or context. (e.g., 'Yes, she would likely enjoy it because she enjoys classical music like Bach and Mozart.' or 'She attended the conference on 10 July 2023.')"
}"""



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
        # 调用 retrieval 进行双因子检索
        results = self.retriever.retrieve(question, query_embedding, top_k=self.settings.retrieval_top_k)

        if not results:
            return "unknown"
        context = "\n".join(f"- {mem.text_description}" for mem, _ in results)
        prompt = ANSWER_PROMPT.format(memory_context=context, question=question, persona_context=self.persona_summaries)
        # 喂给 LLM 生成答案，解析 JSON 提取 answer 字段
        raw = self.llm.generate(prompt, max_tokens=4096).strip()
        import re
        try:
            data = json.loads(raw)
            return data.get("answer", raw)
        except (json.JSONDecodeError, TypeError):
            # JSON 被截断时，正则兜底提取 "answer" 字段的值
            m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
            if m:
                return m.group(1)
            return raw
