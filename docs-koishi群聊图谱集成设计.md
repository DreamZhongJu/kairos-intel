# Koishi 群聊 → 凯伊知识图谱 集成设计

> 2026-08 调研 · 基于 192.168.10.13 实机勘察 · 配套仓库现状：KG 抽取管线（分块+gleanings+并行）、别名去重、可视化面板均已就绪

## 一、现状盘点（实机勘察结论）

### 1.1 服务器容器全景（192.168.10.13，Ubuntu 22.04，ThinkPad T430s）

| 容器 | 镜像 | 端口 | 角色 |
|---|---|---|---|
| napcat | mlikiowa/napcat-docker | 6099 | QQ NTQQ 协议端，对外提供 OneBot v11 |
| koishi | koishijs/koishi:latest-lite | 5140 | Bot 框架，`ws-reverse` 挂在 `/onebot` 等 NapCat 反连 |
| kairos | feishu-assistant-kairos | 8095 | **凯伊本体**（REST API + MCP） |
| cloudflared | cloudflare/cloudflared | — | 公网隧道 |
| we-mp-rss | we-mp-rss | 8001 | 微信读书 RSS |

消息链路：**QQ ⇄ NapCat ⇄（OneBot v11, ws-reverse）⇄ Koishi**。Bot QQ 号 selfId=`3434290172`。

### 1.2 关键路径与版本

- Koishi 数据：宿主 `/srv/games/koishi/data` → 容器 `/koishi`；Koishi `4.18.11`，Node v24
- 凯伊数据：宿主 `/srv/games/daily-intel/feishu-assistant/data/knowledge.db`（即容器内 `/app/data`），**宿主可直接读写**
- 凯伊源码：服务器上有 git clone（`/srv/games/daily-intel/feishu-assistant`，目前落后多个提交），部署走 `git pull && docker compose build`
- 容器网络：koishi 在 `botnet`，kairos 在 `feishu-assistant_default` —— **不互通**，需打通（见 §5）

### 1.3 Koishi 插件生态（已装）

OneBot 适配器（`@pynickle` 版）、ChatLuna 全家桶（core / agent / character / deepseek-adapter / search-service / storage-service）、`touhou-mcp`、database-sqlite、analytics、cron、puppeteer、fish-audio-tts、arknights-card 等。

- ChatLuna character 已应用于群 `830070676`（人格化聊天主战场）
- **重要事实：koishi.db 目前没有任何原始聊天记录落库**。`analytics.message` 仅 818 行按小时聚合的计数；ChatLuna 会话表近乎全空
- 现成的「人物种子数据」：`user` 表 175 个用户（含昵称）、`binding` 表 173 条身份绑定、`channel` 表 8 个 QQ 群 + 1 个 TG 群

在册 QQ 群：`830070676, 965239839, 498153657, 177790151, 429632559, 667852905, 1006143437, 870699338`

### 1.4 凯伊现有对外能力与缺口

已有：REST `POST /api/chat`（完整 agent）、`GET /api/knowledge/stats`；MCP 工具 `knowledge_ingest / knowledge_graph_query / knowledge_search(本地块检索)`。

缺口（本方案要补的）：
1. 没有**结构化批量入库**端点（聊天窗口 → 图谱）
2. 没有**轻量检索**端点（关键词 + 图谱子图，供 Koishi 低延迟问答）
3. 引擎没有**确定性人物节点**支持（以 QQ 号为主键，昵称可变而节点不分裂）

## 二、目标架构

```
QQ群 ⇄ NapCat ⇄ Koishi ──┬── [采集] koishi-plugin-kairos-collector
   (ws-reverse /onebot)   │     ├─ ctx.on('message') 白名单群落库(koiros.db 自建表)
                          │     ├─ 窗口聚合(每群 N 条 / T 分钟) 
                          │     └─ POST http://<kairos>:8095/api/knowledge/ingest
                          │
                          └── [问答] @bot 提问 → GET /api/knowledge/query
                                └─ 返回检索结果拼装文本 → 回复群里
                                
凯伊侧：ingest 端点 → 确定性人物节点注册 + 对话文本走既有抽取管线
        （分块 + gleanings + 并行 + 别名去重，全部现成）→ SQLite 图谱增量更新
```

## 三、关键设计决策

### D1 人物节点 = QQ 号（确定性 ID）

昵称随时会改，不能当主键。引擎 `upsert_entity` 增加 `canonical` 显式参数：

```python
engine.upsert_entity("小明", "人名", canonical="qq:123456")
```

- 同一 QQ 号换昵称 → 同一节点改名，历史关系不丢
- 不同昵称指向同一 QQ → 自然合并（现有别名机制反向受益）
- `graph_query` 的查找链已是「精确 canonical → 别名基 → 模糊候选」，昵称/QQ 号均可命中
- 类型词汇表扩充：`TYPE_WHITELIST` 增加 `"群组"`（群号作实体，如 `group:830070676`）

### D2 采集用自写轻量插件，而非 msgdb

`msgdb` 只管存库，而我们要的是**白名单过滤 + 窗口聚合 + 推送凯伊**三件事一体；且表结构要为「按群打包对话窗口」服务。自写约 150 行 TS，同时把原始消息先落 `koishi.db` 自建表（`kairos_messages`），推送失败可重试——**原始记录永远先保底在 Koishi 侧**。

### D3 抽取靠窗口聚合，绝不逐条过 LLM

- 每群攒满 `flushCount`(默认 40) 条或 `flushInterval`(默认 600s) 即 flush
- 窗口渲染成带说话人的对话体：

```
[08-23 21:02] 小明(qq:123456): 今天那个部署脚本又炸了
[08-23 21:03] 小红(qq:234567): 我看看，是不是证书又过期了
```

- 该文本直接进凯伊现有 `kg_extract.extract()`（并行分块已提速 2.6–3.2×），LLM 抽人与话题的关系；**人物节点与「活跃于群」边由确定性代码补齐**（零 LLM 成本、永不漏）
- 成本量级估算：8 群 × 每天 ~20 窗口 = ~20 次抽取调用/天，flash 模型下几乎可忽略

### D4 问答走轻量端点而非完整 agent

群内 @bot 提问要快。v0 用 `GET /api/knowledge/query`（关键词命中片段 + 图谱 1–2 跳子图拼装，纯 SQLite，毫秒级）；后续再评估接 `/api/chat` 或挂进 ChatLuna 当工具（ChatLuna 有工具扩展生态，且它本身就是 DeepSeek 脑）。

### D5 隐私边界

- 采集只对**白名单群**生效（插件配置项），默认全关
- 命令管理：`kairos.collect.on/off` 按群开关；入库内容仅存自托管 SQLite
- ⚠️ 安全提醒：勘察时 koishi.yml 中 Telegram bot token 明文可见，建议轮换

## 四、两端改造清单

### 4.1 凯伊侧（本仓库）

| # | 改动 | 说明 |
|---|---|---|
| K1 | `engine.upsert_entity(name, etype, canonical=None)` | 支持确定性 ID；`find_entity` 查找链兼容 |
| K2 | `TYPE_WHITELIST` + `"群组"` | 群实体类型 |
| K3 | 新增 `knowledge/ingest.py`：`ingest_chat_window(channel_id, messages)` | 渲染对话体 → 人物/群确定性注册 → `extract()` → 入库；返回统计 |
| K4 | REST：`POST /api/knowledge/ingest` | 批量窗口入库（带简单 token 校验，防公网裸奔） |
| K5 | REST：`GET /api/knowledge/query?q=&entity=` | 关键词片段 + 图谱子图拼装文本 |
| K6 | 面板：图谱页加来源筛选 | 区分 qq 来源实体（可选，后期） |
| K7 | 测试 | ingest 渲染/确定性注册/端到端小样 |

### 4.2 Koishi 侧（新插件 `koishi-plugin-kairos-collector`）

```typescript
import { Context, Schema } from 'koishi'

export const inject = ['database']

export interface Config {
  groups: string[]          // 白名单群号
  kairosEndpoint: string    // 默认 http://172.22.0.1:8095（宿主网桥）
  apiToken?: string         // 与凯伊侧校验一致
  flushCount: number        // 默认 40
  flushInterval: number     // 秒，默认 600
}

export const Config: Schema<Config> = Schema.object({ /* …上述字段… */ })

export function apply(ctx: Context, cfg: Config) {
  // 1) 落库表
  ctx.model.extend('kairos_messages', {
    id: 'unsigned', channelId: 'string', userId: 'string',
    nickname: 'string', content: 'text', time: 'timestamp',
    synced: 'boolean',
  }, { autoInc: true })

  // 2) 采集：仅白名单群、仅群聊、跳过bot自身
  ctx.on('message', (s) => {
    if (!cfg.groups.includes(s.channelId)) return
    if (s.selfId === s.userId) return
    ctx.database.create('kairos_messages', {
      channelId: s.channelId, userId: s.userId,
      nickname: s.author?.nick || s.author?.name || s.userId,
      content: s.content, time: new Date(), synced: false,
    })
  })

  // 3) 定时/定量 flush：未同步行按群打包 → POST ingest → 标记 synced
  //    ctx.setInterval + 计数触发，失败保留待重试

  // 4) 问答：@bot 或前缀命令 → GET /api/knowledge/query → 回文本
}
```

安装方式（lite 镜像无市场私有包）：本地 `npm pack` 出 tgz → 上传服务器 → 放入 `/srv/games/koishi/data` 后 `yarn add file:./kairos-collector.tgz` → `koishi.yml` 注册插件并配白名单。

### 4.3 网络打通（一次性运维）

```bash
docker network connect feishu-assistant_default koishi
# 之后 koishi 容器内直接访问 http://kairos:8095
```

## 五、分期实施

| 期 | 内容 | 产出 |
|---|---|---|
| P1 | 凯伊侧 K1–K5 + K7（纯本仓库） | 可独立联调的 HTTP 入库/查询接口 |
| P2 | Koishi 采集插件（落库+flush 推送） | 群聊自动进图谱，面板可看人-话题网络生长 |
| P3 | Koishi 问答命令（@bot 问 → query 端点答） | 群里可用的知识问答 |
| P4 | 进阶：ChatLuna 工具化接入、周报「群聊热点人物」、描述入库 | 体验打磨 |

P1 全部在本仓库完成、不影响线上；P2/P3 需要 Koishi 侧重启（选低峰操作）。
