# Feishu Research Assistant

一个自托管的私人飞书研究助手。它使用 LangGraph 编排工具调用，并可接入 DeepSeek、联网搜索、飞书云文档/知识库、日程和本地长期记忆。部署者可以为助手自行设定名称与人格；本项目不绑定特定角色名。

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
- `assistant/agent/`：LangGraph 工具调用图与模型消息兼容层。
- `assistant/tools/`：搜索、网页、论文、云文档、知识库、附件、日程和归档工具。
- `assistant/channels/`：飞书消息与 API 传输。
- `assistant/memory/`：SQLite 长期记忆与 Claude-Mem 适配。
- `tests/`：无需真实密钥或联网的启动级冒烟测试。
