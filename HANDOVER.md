# Kairós 项目交接总结

> 更新时间：2026-08-14（由交接对话整理并归档进仓库，供后续接手者阅读）。
> 注意：本文件不含任何密钥、口令或令牌；凭据请从服务器 `.env` / 私密保管处获取。

## 一句话

Kairós 是一个自托管的多渠道个人情报助手：飞书里的「凯伊」是它的人格，每日情报日报已并入同一个项目，部署在一台 Ubuntu 服务器上。GitHub 仓库已改名并推送，服务器已切换上线。

## 已完成的事（方案 → 上线）

1. 方案确定：凯伊做大脑、每日新闻收编、koishi（QQ）可选。
2. 开源准备：MIT 许可证、`THIRD_PARTY_NOTICES.md`、依赖许可证审计（82 个依赖全宽松、无 copyleft）、git 历史密钥扫描（干净）。
3. 模型层多 provider 重构：`kairos/infrastructure/llm.py`，预置 deepseek/openai/qwen/moonshot/zhipu/siliconflow/openrouter/ollama，向后兼容旧变量。
4. 包重命名：`assistant/` → `kairos/`，全部导入引用已替换。
5. 每日新闻收编：`kairos/reports/generator.py`，复用 `llm.build_client()` / `model_name()` / `settings`（代码互通，不是文件搬家）。
6. 日报调度：`kairos/reports/scheduler.py`（APScheduler，每天 09:00 Asia/Shanghai），接入 `app.py`。
7. 分两个 commit 推送到 GitHub，完成服务器切换部署。

## 本轮追加（2026-08-14）

- 日报推送通道统一：`send_feishu()` 优先走 `FEISHU_REPORT_CHAT_ID`（大脑飞书通道，tenant token，`im/v1/messages`），未配置时回退 `FEISHU_WEBHOOK_URL`（向后兼容）。
- 文档批量改名「飞书研究助手」→ Kairós：README、架构设计说明、对比分析、评测套件（`run_eval.py` / `eval_result.json` / `eval_report.md`）、Web 面板、测试。
- README 更新：每日情报日报已并入本仓库（不再是独立项目），旧 horizon 项目已归档说明。
- 部署确认：2026-08-14 09:03 日报生成并推送成功（webhook 200，`structured-2026-08-14.md` 已生成）；大脑通道向目标会话发送测试消息返回 `code:0`。

## 本轮追加（2026-08-14 ~ 08-17）

- 回答可信度：回答末尾自动追加「📎 参考来源 + 检索路径」脚注——来源从真实工具调用链提取（搜索 JSON 的 url、`read_webpage`/`read_feishu_document` 入参、纯文本 URL 兜底），去重上限 6 条；模型自选引用（回答末尾「参考来源：」段）经运行时校验，只保留真实出现在工具输出中的 URL，杜绝幻觉链接；工具纪律约束 `github_research` 仅用于开源主题。
- GitHub Trending 纳入日报：`fetch_github_trending()` 抓取 `github.com/trending?since=daily`（可选语言），解析服务端渲染 HTML、跳过赞助位，并入「开源社区动态」与「每日一个新技术」候选池；`report_config.json` 可配置 `github_trending` / `trending_since` / `trending_language`。2026-08-17 部署生效，次日 09:00 起日报包含 Trending 热门。
- QQ（koishi/豫康）维持现状：调研后决定不合并（独立栈、Telegram 生态、人格体系差异大；桥接方案已评估，随时可做）。

## 本轮追加（2026-08-18）

- 周报/月报：`kairos/reports/periodic.py` 汇总近期日报 + 请求日志（用量/工具/高频问题）+ 长期记忆，生成「每周情报周报」（周一 09:05）与「每月情报月报」（每月 1 日 09:10），经大脑通道推送；`WEEKLY_REPORT` / `MONTHLY_REPORT` / `DAILY_REPORT_WEEKLY_DAY` / `DAILY_REPORT_MONTHLY_DAY` 可配置。
- 流式/分段回复：LLM 长回答前先发「正在检索整理」占位消息，完成后按段落分块（≤1600 字）发送并撤回占位（`chunk_text` / `recall_message`）。
- 本地知识库与知识图谱：完成可行性研究，见 `docs-本地知识库与知识图谱设计.md`——结论是可做轻量版（LLM 抽取三元组 + SQLite 图存储 + 向量/关键词/图三通道检索），P1 向量 RAG / P2 图谱层 / P3 可视化，待实施。

## 关键位置 / 账号

- 项目名：Kairós（品牌），仓库/包名 `kairos`，GitHub 仓库 `DreamZhongJu/kairos-intel`，主分支 `main`。
- 本地代码：`D:\Document\MyCodeProject\kairos`（原 `feishu-research-assistant`）。
- 服务器：`ssh root@192.168.10.13 -p 22`（口令见私密保管处；曾在对话中暴露，建议尽快轮换）。
- 服务器项目目录：`/srv/games/daily-intel/feishu-assistant`（是仓库的 git clone，部署 = `git pull` + `docker compose up -d --build kairos`）。
- 旧日报代码：`/srv/games/daily-intel/horizon`（已不再调度，可归档）。

## 部署状态

- 容器 `kairos`（docker compose）运行中：飞书 WS 已连、日报调度器已启动（每天 09:00 Asia/Shanghai）。
- 旧 `daily-intel-report.timer` 已停用、horizon 已停止。
- 其他容器未动：`koishi`、`napcat`、`we-mp-rss`、`cloudflared`。
- 日报输出：容器内 `/reports`，绑定挂载 `./reports:/reports`（`structured-<date>.md`）。
- 容器内代理：`HTTP(S)_PROXY=http://host.docker.internal:7890`，`NO_PROXY` 含 `api.deepseek.com,open.feishu.cn`。

## 架构要点

- 技术栈：Python + LangGraph + DeepSeek（OpenAI 兼容）+ 20 个工具 + 分层记忆（SQLite + Claude-Mem）+ MCP client + Web 面板 + 评测套件。
- 目录：`kairos/`（agent / tools / memory / channels / infrastructure / observability / reports）。
- 模型层：`kairos/infrastructure/llm.py`，`MODEL_PROVIDER` 切换 provider。
- 日报：`kairos/reports/generator.py`（生成）+ `scheduler.py`（调度）。
- 人格：凯伊 = 飞书；豫康 = QQ（Koishi 独立，未合并）。

## 遗留待办

1. 轮换泄露过的 DeepSeek key（`koishi.yml` 里曾明文存过）。
2. 去 GitHub 归档旧的「每日日报」仓库（代码已并入 kairos）。
3. koishi（QQ）还没并进来（之前决定「难就跳过」）。
4. 服务器 root 口令轮换。
5. 服务器上残留的未跟踪文件可清理：`assistant/`（旧包目录）、`IMPLEMENTATION_LOG.md`、`skills/huggingface-papers/`。
6. 日报切到大脑通道后，确认次日 09:00 从新通道送达飞书。
