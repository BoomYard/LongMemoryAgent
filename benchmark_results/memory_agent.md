# MemoryAgent 答题情况分析

> 评测日期：2026-06-02 | 评测集：LoCoMo 分层抽样，4 类别 × 40 题 = 160 题

---

## 一、总体表现

| 指标 | 数值 | 说明 |
|------|:----:|------|
| Score | **0.2625** | |
| F1 | **0.1750** |  |
| EM (Exact Match) | **0.0563** | 唯一产生精确匹配的系统（9 题） |
| 平均延迟 | **0.733s** | ） |
| Correct | **33** (20.6%) | 正确 |
| Partial | **18** (11.3%) | 部分正确 |
| Wrong | **109** (68.1%) | 错误数 |

> 其中 81 题回答为 `"unknown"`（占全部 160 题的 50.6%），即检索阶段未能找到相关信息。

---

## 二、分类别表现

| 类别 | Score | F1 | EM | unknown 占比 | 排名 |
|------|------:|---:|---:|:---:|:---:|
| single_hop | **0.3125** | **0.1905** | **0.0250** | 27.5% (11/40) | 🥇 第 1 |
| temporal | **0.2375** | **0.1856** | **0.0500** | 62.5% (25/40) | 🥇 第 1 |
| multi_hop | 0.1125 | 0.0563 | 0.0250 | 70.0% (28/40) | 🥈 落后 FullContext |
| open_domain | 0.3875 | **0.2676** | **0.1250** | 42.5% (17/40) | 🥈 略落后 VanillaRAG |

### 分析
- **single_hop** 表现最好，记忆检索有效定位了单步答案，unknown 率仅 27.5%。
- **temporal** 虽然排名第 1，但 62.5% 回答为 unknown，说明时间相关记忆的检索召回率仍需提升。
- **multi_hop** 是表现最差的类别，Score 仅 0.1125，unknown 率高达 70%，是主要的短板。
- **open_domain** Score 较高（0.3875），且 EM 达到 12.5%，开放性问题的回答质量较好。

---

## 三、错误最多的类别：multi_hop（多跳推理）

**multi_hop** 是 MemoryAgent 错误率最高的类别：

- Score：0.1125（落后 FullContext 的 0.2375，差距 -53%）
- F1：0.0563（落后 FullContext 的 0.0642）
- unknown 回答占比：**70.0%**（28/40），即 28 题完全无法在记忆中检索到相关信息
- 仅 1 题达到 Exact Match

多跳推理需要跨 session 整合多条记忆信息进行推理，当前检索机制未能有效跨越多个不相关的记忆片段。

---

## 四、multi_hop 错误实例

以下为 multi_hop 类别中的典型 "unknown" 回答（预测完全失败）：

### 示例 1：需要推断个人偏好
| 字段 | 内容 |
|------|------|
| 问题 | Would Melanie be more interested in going to a national park or a theme park? |
| 参考答案 | National park; she likes the outdoors |
| MemoryAgent 回答 | **unknown** |

> 该题需要从多条记忆中推断 Melanie 的户外活动偏好（如 camping、hiking、painting nature 等），再与国家公园 vs 主题公园做偏好匹配，属于典型的跨记忆推理。

### 示例 2：需要推断人物特质
| 字段 | 内容 |
|------|------|
| 问题 | Would Caroline be considered religious? |
| 参考答案 | Somewhat, but not extremely religious |
| MemoryAgent 回答 | **unknown** |

> 该题需综合 Caroline 的多条记忆（参加 LGBTQ 会议、提及信仰等）来推断其宗教倾向，单一记忆片段无法直接回答。

### 示例 3：需要跨领域推理
| 字段 | 内容 |
|------|------|
| 问题 | Would Melanie likely enjoy the song "The Four Seasons" by Vivaldi? |
| 参考答案 | Yes; it's classical music |
| MemoryAgent 回答 | **unknown** |

> 需要先检索 Melanie 的音乐偏好（她喜欢 classical music），再判断 Vivaldi 是否属于古典音乐。两步中任一步缺失都会导致 unknown。

### 示例 4：需要结合多个条件判断
| 字段 | 内容 |
|------|------|
| 问题 | Would Caroline want to move back to her home country soon? |
| 参考答案 | No; she's in the process of adopting children. |
| MemoryAgent 回答 | **unknown** |

> 需要先检索 Caroline 是否在领养孩子，再推断领养过程意味着她短期内不会搬家。属于隐含推理。

### 示例 5：需要推断人物关系
| 字段 | 内容 |
|------|------|
| 问题 | Is it likely that Nate has friends besides Joanna? |
| 参考答案 | Yes; teammates on his video game team. |
| MemoryAgent 回答 | **unknown** |

> 需要检索 Nate 在游戏团队中的队友信息，再推断这些队友也属于朋友关系。

### 示例 6：需要推断细节称呼
| 字段 | 内容 |
|------|------|
| 问题 | What nickname does Nate use for Joanna? |
| 参考答案 | Jo |
| MemoryAgent 回答 | **unknown** |

> 昵称信息通常散布在对话的多个位置，需要从对话细节中提取，单一语义检索难以定位。

---

## 五、结论

MemoryAgent 整体排名第 1，在 single_hop 和 temporal 类别上显著领先所有基线。但其 **multi_hop** 类别表现成为主要短板，70% 的题目无法检索到相关信息。根因在于当前的记忆检索是单步语义匹配，无法进行跨记忆片段的链式推理。

**优先改进方向**：增强 multi_hop 多跳检索能力（跨 session 链式推理），预期可提升 Score 10~15 个百分点。
