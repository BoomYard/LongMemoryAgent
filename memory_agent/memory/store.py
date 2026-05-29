"""
本文件实现了本地 DB 数据库的增删改查操作
记忆数据结构：{id, text_description, type, creation_timestamp, last_access_timestamp, importance_score, embedding}
底层使用 ChromaDB PersistentClient
每次创建 MemoryStore 实例时自动清空旧数据，保证每个 Agent 实例状态独立。
"""

import time
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import chromadb


DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")
COLLECTION_NAME = "agent_memories"


#   记忆数据结构 
#   id: 唯一标识符
#   text_description: 记忆的自然语言描述
#   type: 记忆类型（observation 观察 / reflection 反思）
#   creation_timestamp: 创建时间戳
#   last_access_timestamp: 最后一次被访问的时间戳
#   importance_score: 重要性得分（1到10分）
#   embedding: 文本的向量表示
@dataclass
class MemoryUnit:
    id: int
    text_description: str
    type: str
    creation_timestamp: float
    last_access_timestamp: float
    importance_score: float
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


# add方法会将记忆存在本地的 DB 数据库中
# 使用 ChromaDB PersistentClient，数据落盘到 chroma_db/ 目录
# HNSW 空间设为 cosine，与 SentenceTransformer normalize 后的 dot product 一致
class MemoryStore:
    def __init__(self, persist_dir: str = None):
        if persist_dir is None:
            persist_dir = DEFAULT_PERSIST_DIR
        os.makedirs(persist_dir, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_dir)

        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        # 创建记忆集合
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        # 记忆 ID 从 0 开始递增
        self._next_id: int = 0

    # 内部方法：将整数 ID 转换为字符串
    def _id_str(self, memory_id: int) -> str:
        return str(memory_id)

    # 内部方法：将 ChromaDB 中的行数据转换为 MemoryUnit
    def _row_to_memory(self, metadatas, embeddings, ids, idx: int) -> MemoryUnit:
        meta = metadatas[idx]
        mem_id = int(ids[idx])

        embedding: Optional[np.ndarray] = None
        if embeddings is not None:
            emb_data = embeddings[idx]
            if isinstance(emb_data, np.ndarray):
                embedding = emb_data.astype(np.float32)
            else:
                embedding = np.array(emb_data, dtype=np.float32)

        return MemoryUnit(
            id=mem_id,
            text_description=meta.get("text_description", ""),
            type=meta.get("type", "observation"),
            creation_timestamp=float(meta.get("creation_timestamp", 0)),
            last_access_timestamp=float(meta.get("last_access_timestamp", 0)),
            importance_score=float(meta.get("importance_score", 5)),
            embedding=embedding,
        )

    # 写入：将格式化后的记忆存入 ChromaDB
    # embedding 以 list 形式存储，读取时还原为 np.ndarray
    def add(self, text: str, memory_type: str, importance_score: float,
            embedding: np.ndarray, timestamp: Optional[float] = None) -> int:
        ts = timestamp
        mem_id = self._id_str(self._next_id)
        self.collection.add(
            ids=[mem_id],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "text_description": text,
                "type": memory_type,
                "creation_timestamp": ts,
                "last_access_timestamp": ts,
                "importance_score": importance_score,
            }],
            documents=[text],
        )
        self._next_id += 1
        return int(mem_id)

    def get(self, memory_id: int) -> Optional[MemoryUnit]:
        results = self.collection.get(
            ids=[self._id_str(memory_id)],
            include=["embeddings", "metadatas"],
        )
        if not results["ids"]:
            return None
        return self._row_to_memory(results["metadatas"], results["embeddings"], results["ids"], 0)

    # 检索命中后将 last_access_timestamp 更新为当前时间
    # ChromaDB update 为部分更新，只覆盖指定字段，其余保留
    def update_access_time(self, memory_id: int, timestamp: Optional[float] = None):
        ts = timestamp
        self.collection.update(
            ids=[self._id_str(memory_id)],
            metadatas=[{"last_access_timestamp": ts}],
        )

    def remove(self, memory_id: int):
        self.collection.delete(ids=[self._id_str(memory_id)])

    #获取并返回所有记忆，用于三因子检索
    def get_all(self) -> list[MemoryUnit]:
        results = self.collection.get(include=["embeddings", "metadatas"])
        if not results["ids"]:
            return []
        return [
            self._row_to_memory(results["metadatas"], results["embeddings"], results["ids"], i)
            for i in range(len(results["ids"]))
        ]

    def __len__(self):
        return self.collection.count()
