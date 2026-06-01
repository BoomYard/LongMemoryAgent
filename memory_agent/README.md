# 长期记忆对话 Agent

基于 Generative Agents 的三因子检索 + Reflection 机制实现的长期记忆对话 Agent。

## 环境

- Python 3.10+
- 依赖：`pip install openai sentence-transformers chromadb python-dotenv`
- LLM：Qwen2.5-3B-Instruct-AWQ（vLLM 部署）
- Embedding：BAAI/bge-small-zh-v1.5（SentenceTransformer 本地加载）

## 运行

详见项目根目录的 `指令.md`。

```powershell
# 1. 启动 vLLM（新终端）
vllm serve Qwen/Qwen2.5-3B-Instruct-AWQ --port 8000 --max-model-len 8192

# 2. 准备数据
cd eval_kit\eval_kit
python prepare_eval_set.py --output eval_set.json --per_category 10 --seed 42

# 3. 运行 MemoryAgent + Judge
cd memory_agent\eval
python run_eval.py --eval_set ..\..\eval_kit\eval_kit\eval_set.json --output predictions.json --skip_judge
python run_eval.py --skip_generation --output predictions.json --results results.json --judge_base_url https://api.deepseek.com/v1 --judge_model deepseek-v4-flash --judge_api_key sk-你的key
```

## 模块结构

```
memory_agent/
├── memory/
│   ├── store.py          ChromaDB 记忆存储
│   ├── writer.py         记忆提取 / 重要性评分 / 向量化
│   ├── retriever.py      三因子打分检索（Recency × Importance × Relevance）
│   └── updater.py        反思机制（sum_score 触发 reflect）
├── agent/
│   └── controller.py     Agent 主流程 + Settings 参数
├── eval/
│   └── run_eval.py       评测入口
└── experiments/results/  实验结果
```

## 可调参数

编辑 `agent/controller.py` 中 `Settings` 类的默认值：

| 参数 | 默认 | 含义 |
|---|---|---|
| retrieval_top_k | 5 | 检索记忆条数 |
| decay_factor | 0.995 | Recency 衰减系数 |
| reflection_threshold | 150 | 累计重要性触发反思阈值 |
| reflection_memory_limit | 100 | 反思取最近记忆数 |
