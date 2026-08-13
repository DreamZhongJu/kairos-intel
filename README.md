# Kairós

一个自托管的多渠道个人情报助手（原名 Kairós）。它使用 LangGraph 编排工具调用，并可接入 DeepSeek、联网搜索、飞书云文档/知识库、日程和本地长期记忆。部署者可以为助手自行设定名称与人格；本项目不绑定特定角色名。

## 能力

- 飞书私聊和群聊中的对话式研究助手
- 联网搜索、网页阅读、论文与研究趋势查询
- 已授权飞书云文档、知识库与日程读取
- 明确指令下创建云文档、归档知识库
- LangGraph 工具节点与本地 Claude-Mem 长期记忆
- 与独立的每日情报日报服务协同

## 安全说明

本仓库不会包含 API 密钥、飞书凭据、OAuth 授权结果、知识库私有标识、用户对话、附件、数据库或运行日志。请从示例配置创建自己的 `.env`。

## 运行要求

- Python 3.11+
- 一个飞书自建应用及其所需权限
- DeepSeek 或兼容 OpenAI 的模型接口
- 可选：本地 Claude-Mem 服务，用于长期记忆

安装依赖后，配置环境变量并运行 `python app.py`。

## 相关项目

每日情报日报独立维护于另一个项目，以便将飞书交互与定时信息聚合解耦。

## 本地开发与校验

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

未配置密钥时，模块仍可被导入并运行测试；执行 `python app.py` 才会校验
`LARK_APP_ID`、`LARK_APP_SECRET`、`DEEPSEEK_API_KEY` 与 `TOKEN_ENCRYPTION_KEY`。

## 代码结构

- `app.py`：飞书事件入口、显式确认流程与服务装配。
- `kairos/agent/`：LangGraph 工具调用图与模型消息兼容层。
- `kairos/tools/`：搜索、网页、论文、云文档、知识库、附件、日程和归档工具。
- `kairos/channels/`：飞书消息与 API 传输。
- `kairos/memory/`：SQLite 长期记忆与 Claude-Mem 适配。
- `tests/`：无需真实密钥或联网的启动级冒烟测试。

## 评测

离线评测套件位于 `evaluation/`，可在不依赖飞书/外部服务的情况下复现：

| 维度 | 指标 | 结果 |
| --- | --- | --- |
| 知识问答 | 平均忠实度（faithfulness） | 1.000 |
| 知识问答 | 平均回答相关性（answer relevancy） | 1.000 |
| 知识问答 | 上下文命中率 hit@3 | 1.000 |
| 知识问答 | 负例不编造率 | 1.000 |
| 工具路由 | Top-1 准确率 | 1.000 |
| 工具路由 | Top-3 准确率 | 1.000 |

运行方式：

````powershell
.\.venv\Scripts\python.exe evaluation\run_eval.py
````

说明：评测集为项目自身文档语料（31 块）、13 个问答用例（含 3 个负例）与 10 个工具路由用例；结果见 `evaluation/eval_report.md`。评测使用 DeepSeek 作为生成与打分模型，需在 `.env` 配置 `DEEPSEEK_API_KEY`。

## 可观测性与 Web 面板

每次请求（问题、工具调用链、token、耗时、状态）自动记录到 `data/assistant.db`，并在本地 Web 面板可视化：

- 仪表盘：请求量、成功率、平均耗时、Token 消耗、近 7 天趋势、工具调用排行
- 请求日志：按状态/工具筛选、分页、查看单条详情（工具调用链与完整回答）
- 运行状态：模型、工具数、Claude-Mem、数据库路径等配置概览

启动面板：

````powershell
.\.venv\Scripts\python.exe web_panel.py
````

默认地址 http://127.0.0.1:8090 ；可选设置 `PANEL_TOKEN` 启用访问令牌，`PANEL_PORT` 修改端口。日志中的 question/answer 已脱敏，owner 已哈希存储。

## 记忆分层治理

长期记忆按 mem0 / Letta 的思路分两层治理：

- **核心记忆**：用户长期身份与核心偏好（姓名、研究方向、重要偏好），每次请求都会加载，每用户最多 8 条，超出自动修剪最旧的。
- **存档记忆**：项目、习惯、决定等可检索事实，按相关性召回（召回会更新访问统计），每用户上限 120 条，按最近访问时间自动清理。

记忆写入由模型输出显式操作（`add` / `update` / `delete` / `noop`），同一事实会被合并更新而不是重复插入；用户说"忘记 XX"会删除对应记忆。敏感信息（密钥、密码）在写入前被过滤。

## MCP 客户端

支持通过 MCP（Model Context Protocol）动态接入外部工具，无需为每个工具写代码：

- 复制 `mcp_servers.example.json` 为 `mcp_servers.json`，启用需要的 server（支持 `stdio` 本地进程与 `streamable_http` 远程服务两种传输）。
- 启动时自动发现并注册这些 server 暴露的工具，Agent 可直接调用；某个 server 不可用时自动跳过，不影响内置工具。
- 依赖 `mcp` Python SDK（已加入 `requirements.txt`）。

仪表盘另含估算成本（按 token × 单价，可通过 `DEEPSEEK_INPUT_PRICE_PER_M` / `DEEPSEEK_OUTPUT_PRICE_PER_M` 调整）与最近失败列表；MCP 配置（`mcp_servers.json`）在启动时校验，无效 server 会跳过并给出明确原因。

## 记忆管理

Web 面板新增「记忆」页：按用户查看核心/存档记忆（类别、访问次数、更新时间），支持删除单条与清空某用户全部记忆。

## 评测闭环

- 真实请求回放：`evaluation\replay_eval.py` 读取请求日志中的真实问答并打分（相关性、完整性），把生产使用转化为回归信号。
- CI：`.github/workflows/ci.yml` 在无凭据环境运行全部单测；配置 `DEEPSEEK_API_KEY` secret 后追加离线评测 job。
