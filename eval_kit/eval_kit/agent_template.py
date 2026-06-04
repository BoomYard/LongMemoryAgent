"""
Agent 接口模板。

学生需要在自己的文件中（例如 agent/my_agent.py）实现一个 MemoryAgent 类。
run_generation.py 会导入这个类，并依次调用 ingest() 和 answer()。

接口约定：
  - __init__(self): 每次构造都是全新的状态，不接受必填参数
  - ingest(self, conversation): 读入一段完整的多会话对话
  - answer(self, question) -> str: 基于已有记忆回答问题

重要：评测集中的每一段对话都会 new 一个新的 Agent 实例，
      不同对话之间状态不共享。你所有的记忆逻辑必须在单个实例内闭环。
"""

from typing import Protocol


class MemoryAgent(Protocol):
    """接口约定。在你自己的类中实现这些方法即可。"""

    def ingest(self, conversation: dict) -> None:
        """读入一段多会话对话。

        参数：
            conversation: dict，包含以下键：
                - speaker_a: str，说话人 A 的名字
                - speaker_b: str，说话人 B 的名字
                - sessions: list，每项为 {session_id, date_time, turns}
                    其中 turns = list of {speaker, dia_id, text}

        你应该在这里把对话加工成自己设计的记忆表示并存下来。
        对于每段对话，本方法只会被调用一次，且在所有 answer() 之前。
        """
        ...

    def answer(self, question: str) -> str:
        """基于已有记忆回答一个问题。

        参数：
            question: str

        返回：
            简短的自然语言答案（str）。请尽量简洁，参考答案通常是短语或单句。
        """
        ...


# -----------------------------------------------------------------------------
# 参考实现：最朴素的 "Full-Context" 基线
# ——把整段对话塞进 prompt，完全不做记忆管理。
# 你可以拿它当作流水线的烟雾测试（sanity check），
# 但不要当成自己的最终方案交上来。
#
# 截断策略：优先使用 tokenizer 做精确 token 级截断（匹配模型上下文窗口）；
#           若 tokenizer 不可用则退化为字符数估算。始终保留最近的对话内容。
# -----------------------------------------------------------------------------

class FullContextAgent:
    """把整段对话塞进 prompt，按 token 预算自动截断。

    max_tokens: 模型上下文窗口总大小（含 prompt + completion）
    completion_reserve: 为模型回答预留的 token 数
    """

    _tokenizer = None

    def __init__(self, max_tokens: int = 8000, completion_reserve: int = 128):
        from llm_client import LLMClient
        self.llm = LLMClient()
        self.max_tokens = max_tokens
        self.completion_reserve = completion_reserve
        self._lines = []

    @classmethod
    def _get_tokenizer(cls):
        if cls._tokenizer is not None:
            return cls._tokenizer
        try:
            from transformers import AutoTokenizer
            import os
            model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct-AWQ")
            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=False
            )
        except Exception:
            cls._tokenizer = None
        return cls._tokenizer

    def ingest(self, conversation: dict) -> None:
        lines = []
        for sess in conversation["sessions"]:
            lines.append(f"[Session {sess['session_id']} @ {sess['date_time']}]")
            for turn in sess["turns"]:
                lines.append(f"{turn['speaker']}: {turn['text']}")
        self._lines = lines

    def answer(self, question: str) -> str:
        template_before = (
            "You are an assistant with access to a long conversation between two people. "
            "Answer the user's question using only information from the conversation. "
            "Keep the answer short (a phrase or one sentence). "
            "If the conversation does not contain the answer, reply 'unknown'.\n\n"
            "=== Conversation ===\n"
        )
        template_after = (
            "\n\n"
            "=== Question ===\n"
            f"{question}\n\n"
            "=== Answer ==="
        )

        tokenizer = self._get_tokenizer()

        if tokenizer is not None:
            overhead_tokens = (
                len(tokenizer.encode(template_before))
                + len(tokenizer.encode(template_after))
            )
            available = self.max_tokens - overhead_tokens - self.completion_reserve

            lo, hi = 0, len(self._lines)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                candidate = "\n".join(self._lines[-mid:])
                if len(tokenizer.encode(candidate)) <= available:
                    lo = mid
                else:
                    hi = mid - 1

            history_text = "\n".join(self._lines[-lo:]) if lo > 0 else ""
        else:
            total_overhead = len(template_before) + len(template_after)
            available_chars = (self.max_tokens - self.completion_reserve) * 3.5 - total_overhead
            full = "\n".join(self._lines)
            if len(full) <= available_chars:
                history_text = full
            else:
                history_text = full[-int(available_chars):]
                first_nl = history_text.find("\n")
                if first_nl != -1:
                    history_text = history_text[first_nl + 1:]

        prompt = template_before + history_text + template_after
        return self.llm.generate(prompt, max_tokens=self.completion_reserve).strip()


# -----------------------------------------------------------------------------
# No-Memory 基线 —— 完全不使用对话历史，只把当前问题喂给 LLM。
# 这是"下界"：如果你的系统连这个都打不过，说明记忆模块反而有害。
# -----------------------------------------------------------------------------

class NoMemoryAgent:
    """零记忆基线：忽视所有对话，仅基于问题本身作答。"""

    def __init__(self):
        from llm_client import LLMClient
        self.llm = LLMClient()

    def ingest(self, conversation: dict) -> None:
        pass

    def answer(self, question: str) -> str:
        prompt = (
            "You are an assistant. The user asks a question but you have NO access "
            "to any conversation history. If the question appears to reference past "
            "events, people, or context that you cannot know, simply reply 'unknown'.\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=64).strip()


# -----------------------------------------------------------------------------
# Vanilla RAG 基线 —— 逐轮切片 + 向量检索。
# 把对话切成以"轮次"为单位的 chunk，用 sentence-transformers 编码为向量；
# 回答时检索 top-k 相似轮次，塞进 prompt。
#
# 这是一个故意做得很弱的基线：不做记忆抽取/摘要/更新/去重。
# -----------------------------------------------------------------------------

class VanillaRAGAgent:
    """逐轮切片 RAG 基线：embed 每条对话轮次，回答时检索 top-k。"""

    def __init__(self, top_k: int = 5):
        from llm_client import LLMClient
        self.llm = LLMClient()
        from sentence_transformers import SentenceTransformer
        import os
        embed_model = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")
        self.embed_model = SentenceTransformer(embed_model)
        self.top_k = top_k
        self.chunks: list[str] = []
        self.embeddings = None

    def ingest(self, conversation: dict) -> None:
        import numpy as np
        chunks = []
        for sess in conversation["sessions"]:
            for turn in sess["turns"]:
                chunks.append(
                    f"[{sess['date_time']}] {turn['speaker']}: {turn['text']}"
                )
        self.chunks = chunks
        vecs = self.embed_model.encode(chunks, normalize_embeddings=True)
        self.embeddings = np.array(vecs, dtype=np.float32)

    def _retrieve(self, query: str, k: int) -> list[str]:
        import numpy as np
        qvec = self.embed_model.encode([query], normalize_embeddings=True)[0]
        sims = self.embeddings @ qvec.astype(np.float32)
        idx = np.argsort(-sims)[:k]
        return [self.chunks[i] for i in idx]

    def answer(self, question: str) -> str:
        retrieved = self._retrieve(question, self.top_k)
        ctx = "\n".join(retrieved)
        prompt = (
            "You are answering a question about a past conversation. "
            "Use only the retrieved dialogue snippets below. Keep the answer short "
            "(a phrase or one sentence). If the snippets do not contain the answer, "
            "reply 'unknown'.\n\n"
            f"=== Retrieved snippets ===\n{ctx}\n\n"
            f"=== Question ===\n{question}\n\n"
            "=== Answer ==="
        )
        return self.llm.generate(prompt, max_tokens=64).strip()
