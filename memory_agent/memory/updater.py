"""
本文件实现了记忆的反思更新：
sum_score：累计所有新记忆的 importance_score，超过 150 触发反思
reflect 方法：触发反思流程，取最近 100 条记忆生成高阶见解
reflect_agent：反思的具体实现 —— 生成问题 → 检索 → 总结洞察
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "eval_kit", "eval_kit"))

from llm_client import LLMClient
from memory_agent.memory.store import MemoryStore
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

# 将最近记忆喂给 LLM，生成高层次问题
REFLECT_PROMPT = """You are given the following recent memories:

{memory_list}

From these statements, generate exactly {reflection_question_count} high-level insightful questions that could be answered by synthesizing the information above.

Format: one question per line, no numbering.
Just return the questions, don't contain any other text.
"""


# 对每个问题检索到的信息，LLM 总结出高层次洞察
# 格式示例："Klaus Mueller is dedicated to his research on gentrification (because 1,2,8,15)"
REFLECT_INSIGHT_PROMPT = """Based on the following retrieved information:
{retrieved_info}
From the above, infer exactly {reflection_insight_per_q} high-level insights. Format each as:
"Insight (because 1,2,3)" where the numbers reference the statements above.

Example: "Klaus Mueller is dedicated to his research on gentrification (because 1,2,8,15)"
Format: one insight per line, no numbering.
Just return the insights, don't contain any other text.
"""





class MemoryUpdater:
    def __init__(self, store: MemoryStore, retriever=None, writer=None,
                 reflection_threshold: int = 150,
                 reflection_memory_limit: int = 100,
                 reflection_question_count: int = 3,
                 reflection_insight_per_q: int = 3,
                 reflection_retrieval_top_k: int = 20,
                 reflection_max_tokens: int = 256,
                 reflection_temperature: float = 0.2):
        self.store = store
        self.retriever = retriever
        self.writer = writer
        self.llm = LLMClient()
        # 累加每条新记忆的重要性分数
        self._accumulated_importance: float = 0.0
        # 反思相关参数（来自 Settings）
        self.reflection_threshold = reflection_threshold
        self.reflection_memory_limit = reflection_memory_limit
        self.reflection_question_count = reflection_question_count
        self.reflection_insight_per_q = reflection_insight_per_q
        self.reflection_retrieval_top_k = reflection_retrieval_top_k
        self.reflection_max_tokens = reflection_max_tokens
        self.reflection_temperature = reflection_temperature
        self.chain = chain
    # 当 sum_score >= 150 时触发
    def check_and_reflect(self) -> bool:
        if self._accumulated_importance < self.reflection_threshold:
            return False
        if self.retriever is None or self.writer is None:
            return False
        #  取最近 100 条记忆，按创建时间倒序
        memories = self.store.get_all()
        recent_memories = sorted(memories, key=lambda m: m.creation_timestamp, reverse=True)[:self.reflection_memory_limit]

        if not recent_memories:
            return False
        # 将 100 条记忆格式化后喂给 LLM
        memory_list = "\n".join(
            f"{i+1}. {m.text_description}" for i, m in enumerate(recent_memories)
        )

        # reflect_agent 输出3 个高层次问题
        questions_raw = chain.invoke(
            REFLECT_PROMPT.format(memory_list=memory_list, reflection_question_count=self.reflection_question_count)
        )
        questions = [q.strip() for q in questions_raw.strip().split("\n") if q.strip()][:self.reflection_question_count]

        # 对每个问题做检索再总结洞察
        all_insights = []
        for question in questions:
            # 对问题调用 retrieval 工具进行三因子检索
            query_emb = self.writer.embed_text(question)
            retrieved = self.retriever.retrieve(question, query_emb, top_k=self.reflection_retrieval_top_k)
            retrieved_info = "\n".join(f"{i+1}. {m.text_description}" for i, (m, _) in enumerate(retrieved))
            
            # 基于检索结果总结高层次洞察
            # 格式如："Klaus Mueller 致力于他关于绅士化的研究（因为 1,2,8,15）"
            insights_raw = chain.invoke(
                REFLECT_INSIGHT_PROMPT.format(retrieved_info=retrieved_info, reflection_insight_per_q=self.reflection_insight_per_q)
            )
            insight_lines = [l.strip() for l in insights_raw.strip().split("\n") if l.strip()][:self.reflection_insight_per_q]
            all_insights.extend(insight_lines)

        # 将反思记忆写入数据库
        # 存入时 type 标记为 "reflection"
        for insight_text in all_insights:
            importance = self.writer.score_importance(insight_text, category="reflection")
            embedding = self.writer.embed_text(insight_text)
            self.store.add(
                text=insight_text,
                memory_type="reflection",
                importance_score=importance,
                embedding=embedding,
                timestamp=self.writer.latest_time,
            )

        # 触发反思后减150
        self._accumulated_importance -= self.reflection_threshold
        return True

    # 每写入一条记忆就累加其 importance 到 sum_score
    # 然后自动检查是否触发 reflect
    def add_importance(self, score: float):
        self._accumulated_importance += score
        self.check_and_reflect()
