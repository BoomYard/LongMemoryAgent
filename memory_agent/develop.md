调用逻辑：
当调用ingest方法时，controller 会调用store.py中的add_memory方法将对话添加到记忆库
add_memory方法需要先提取出所有turns，把每个turns拼成一个字符串，然后调用writer.py中的mark_agent，对对话进行评分，然后调用writer.py中的向量化工具对对话进行向量化，最后调用writer.py中的formatting工具格式化文本以：
id: 唯一标识符
text_description: 记忆的自然语言描述（例如："Klaus 正在阅读一本关于士绅化的书"）
type: 记忆类型（观察 observation / 反思 reflection）
creation_timestamp: 创建时间戳
last_access_timestamp: 最后一次被访问的时间戳
importance_score: 重要性得分（1到10分）
embedding: 文本的向量表示
的格式返回，而后调用writer.py中的store_memory方法，将其存在本地的DB数据库中

同时，add_memory方法还会维护一个变量，sum_score，总得分超过150分，add_memory方法会调用 reflect.py中reflect方法，生成一条反思记忆，并将总得分-150分

reflect.py中有reflect_agent，对最近的一百条记忆进行反思，生成反思记忆，反思流程：
将一百条记忆喂给llm令其生成三个高层次问题，再根据问题调用retrieval.py中的retrieval方法方法进行三因子检索，最后对于每一个问题检索到的信息，调用llm令其总结出3条高层次信息：
"关于 Klaus Mueller 的陈述
1.Klaus Mueller 正在写一篇研究论文
2.Klaus Mueller 喜欢阅读一本关于绅士化的书
3.Klaus Mueller 正在与 Ayesha Khan 讨论锻炼 [...]
从上述陈述中你能推断出哪 3 个高层次的见解？（示例格式：见解（因为 1,5,3））"
这个过程生成诸如 "Klaus Mueller 致力于他关于绅士化的研究（因为 1,2,8,15）" 这样的陈述，reflect_agent接受一百条记忆，返回九条高层次的见解，reflect方法调用formatting工具，将九条见解格式化为对应格式，而后将其作为反思存储在数据库中。

三重检索（retrieval方法）
维度 A：Recency（近期性 / 衰减度）
概念：越近发生的事情，或者越近刚刚回想起来的事情，越容易被记住。
算法：论文采用指数衰减函数。
衰减系数=0.995
公式：0.995**Δt，其中 Δt是自上次访问该记忆以来经过的单位时间数。
维度 B：Importance（重要性）
概念：区分日常琐事（比如刷牙）和核心记忆（比如分手、考上大学）。
算法：借助 LLM 打分。在记忆刚刚生成并存入数据库时，就已经生成。提示词如下：
"在 1 到 10 的范围内，1 代表纯粹的日常琐事（如刷牙、叠被子），10 代表极其深刻的经历（如分手、大学录取），请为以下记忆片段评估其可能的心酸/重要程度。
记忆内容：[在柳树市场买杂货]
评分：<填入数字>"
(得到分数后，存入数据库的 importance_score 字段。)
维度 C：Relevance（相关性）
概念：当前发生的事，和哪些历史记忆相关。比如你在讨论化学考试，早饭吃了什么就不相关。
算法：计算当前 Query 文本的 Embedding 向量与数据库中所有记忆的 Embedding 向量的余弦相似度 (Cosine Similarity)。
综合得分计算：
这三个分数原本的值域不同（比如相关性是 0-1，重要性是 1-10），需要用 Min-Max Scaling（归一化），如果出现了max和min相同的，则加一个极小值保护：(X - Min) / (Max - Min + 1e-5)。 把它们都缩放到 [0, 1] 区间，然后相加：
Final_Score=(1.0×Recency)+(1.0×Importance)+(1.0×Relevance)
根据这个 Final_Score 进行倒序排序，在大模型的上下文限制内取足够多的排名靠前的记忆组合成字符串，返回这个字符串，同时，对于被检索到的记忆，将它在数据库中的last_access_timestamp更新为当前时间。
