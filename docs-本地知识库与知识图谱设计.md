# 本地知识库与知识图谱：可行性研究与设计

> 状态：研究完成，待实施。目标：让凯伊具备「本地知识库」能力——上传资料入库、检索、回答；并研究是否/如何叠加知识图谱层。

## 一、结论先行

1. **可以做，且推荐做「轻量版知识图谱」**：不需要引 Neo4j 这类重型图数据库。用 LLM 从文档抽取实体/关系三元组 → 存 SQLite（节点表 + 边表）→ 图查询用 SQL 邻接检索（1–2 跳），个人规模完全够用。
2. 成熟先例（2025-2026 已被验证的路径）：
   - [mnemo](https://github.com/zaydmulani09/mnemo)：Local-first 记忆层 = SQLite 持久知识图谱 + LLM 实体抽取 + 语义检索，兼容任意 OpenAI 兼容后端——和凯伊的多 provider 设计天然契合。
   - [LightRAG](https://blog.hotdry.top/posts/2025/11/26/lightrag-simple-fast-rag/)：图蒸馏融合检索，双图索引，低资源可跑——「向量为主、图为辅」的检索融合思路。
   - [nano-graphrag](https://gonamlui.com/blog/brief-breakdown-of-nano-graphrag-a-lightweight-alternative-to-graphrag)：GraphRAG 的轻量替代。
   - [sqlite-vec + FTS5 混合](https://blog.csdn.net/weixin_29597551/article/details/162217527)：SQLite 原生向量 + 全文检索混合，零重型依赖——与凯伊现有 SQLite 栈一致。
3. **分三步走**：P1 向量 RAG（本地上传 + 检索工具）→ P2 知识图谱层（实体/关系抽取 + 图查询）→ P3 图谱可视化与自动入库。

## 二、为什么值得加知识图谱（相对纯向量 RAG）

纯向量 RAG 的短板正是凯伊场景里会遇到的：

- **跨文档实体关联**：「湖北大学和机器翻译方向有什么关系」「舒诗铜（2023 届、湖北大学、软件工程）和谁有关联」——这类问题靠向量相似度答不好，图谱天然支持多跳。
- **实体归一**：同一机构/作者/论文在不同文档里写法不同（湖北大学/HUBU/湖大），图谱层可以合并别名。
- **可解释性**：回答可附「检索路径/图谱路径」，延续已经做过的引用脚注思路。
- **长尾覆盖**：图谱能命中向量检索漏掉的低频但关系明确的事实。

代价与局限（如实说）：
- 抽取质量依赖 LLM，有噪声、有成本（每个入库文档一次抽取调用）。
- 图谱需要维护（实体合并、关系去重）。
- 折中方案就是 LightRAG 的思路：**向量为主、图为辅**——图检索结果作为补充上下文，不取代向量。

## 三、架构设计

### 3.1 摄取链路

```
用户上传（飞书附件/网页/笔记）或链接
  → 文本抽取（复用 kairos/tools/attachments.py 的 extract_attachment）
  → 切块（按段落/长度，~800-1200 字）
  → 并行：向量化（embedding API） + KG 抽取（LLM 三元组）
  → 写入 SQLite
```

### 3.2 存储（新增 `data/knowledge.db`，SQLite 单库）

| 表 | 字段 | 说明 |
| --- | --- | --- |
| `documents` | id, title, source, kind, created_at | 来源（飞书文件/网页/笔记） |
| `chunks` | id, doc_id, seq, text, embedding BLOB | 切块 + 向量（浮点数组打包） |
| `entities` | id, name, type, canonical | 实体（机构/人名/论文/技术/产品） |
| `relations` | subject_id, predicate, object_id, source_chunk, confidence | 三元组，带来源可溯源 |

检索三通道：
1. **向量**：sqlite-vec 扩展（BLOB + 余弦）；扩展装不上则退化为纯 Python 余弦（个人规模 <10k 块足够快）。
2. **关键词**：SQLite FTS5，中文按字符 trigram 索引（不引 jieba，避免新依赖）。
3. **图**：实体名匹配 + SQL 邻接查询取 1–2 跳邻域。

### 3.3 embedding 选型（关键决策点）

- **DeepSeek 没有 embedding 端点**，必须另配 embedding 提供方。
- 新增配置（沿用 `llm.py` 的多 provider 模式）：
  `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL` / `EMBEDDING_API_KEY`
- 候选：**SiliconFlow 的 bge-m3**（OpenAI 兼容格式、中文强、有免费额度）＞ OpenAI text-embedding-3-small ＞ 本地 sentence-transformers（无外网但重）。
- **降级保底**：embedding 服务不可用时，仅走关键词 + 图谱检索，不阻断功能。

### 3.4 KG 抽取设计

- 抽取 prompt：输入文本块，输出 JSON 三元组数组 `[{subject, type, predicate, object, confidence}]`，限定领域词表（机构/人名/论文/技术/产品/会议），**只抽取显式出现的实体关系**。
- 合并策略：实体名规范化（全角→半角、大小写、别名表）、同名归一；关系按 (subject, predicate, object) 去重。
- 去噪：confidence 阈值（如 ≥0.6）、每块关系数上限（如 20）、人工可预览（入库前显示抽取结果确认）。
- 触发：文档入库时异步抽取，不阻塞摄取。

### 3.5 Agent 工具（新增）

| 工具 | 能力 | 对齐现有 |
| --- | --- | --- |
| `knowledge_ingest` | 上传/链接资料入库（显式指令触发） | 复用附件抽取 |
| `local_knowledge_search` | 向量+关键词混合检索本地库 | 与飞书版 knowledge_search 并存 |
| `knowledge_graph_query` | 实体 1–2 跳邻域查询 | 无 |

回答时：`local_knowledge_search` 命中 → 与图检索结果一起进上下文 → 回复附引用（复用已做的脚注机制）。

## 四、分阶段实施与工作量

| 阶段 | 内容 | 预估 |
| --- | --- | --- |
| **P1 向量 RAG** | embedding 配置 + knowledge.db 建表 + 摄取工具 + 混合检索工具 + 单测 | 2–3 天 |
| **P2 知识图谱** | entities/relations 表 + 抽取管线 + 实体合并 + 图查询工具 + 评测 | 3–4 天 |
| **P3 可选** | web_panel 图谱可视化页、日报自动抽取实体入库、与飞书知识库双向同步 | 每项 1–2 天 |

## 五、风险与对策

- **embedding 依赖外部 API**：FTS5 关键词 + 图谱检索保底，不阻塞。
- **LLM 抽取噪声**：置信度阈值 + 领域词表 + 入库预览确认。
- **sqlite-vec 扩展安装**：容器构建加 wheel；失败降级纯 Python 余弦。
- **隐私**：全部本地存储，与现有设计一致（密钥/记忆/日志本地）。

## 六、参考项目

- mnemo（SQLite 图记忆层）：<https://github.com/zaydmulani09/mnemo>
- LightRAG（图蒸馏检索）：<https://blog.hotdry.top/posts/2025/11/26/lightrag-simple-fast-rag/>
- nano-graphrag：<https://gonamlui.com/blog/brief-breakdown-of-nano-graphrag-a-lightweight-alternative-to-graphrag>
- sqlite-vec + FTS5 离线记忆：<https://blog.csdn.net/weixin_29597551/article/details/162217527>
- GraphRAG 中文优化：<https://github.com/via007/graphrag-Chinese-llm>
- SurrealDB + bge-m3 本地知识库：<https://github.com/My-MC/surreal-knowledge-base>
