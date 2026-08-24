# koishi-plugin-kairos-collector

把白名单 QQ 群的聊天记录采集进 [Kairós](../../)（凯伊）知识图谱，并提供轻量图谱问答命令。

## 工作方式

1. **采集**：监听白名单群消息，逐条写入 Koishi 数据库表 `kairos_messages`（带 `synced` 标记，崩溃不丢数据）。
2. **推送**：单群攒满 `flushCount` 条或每 `flushInterval` 秒，把未同步记录按时间窗口 POST 到凯伊
   `POST /api/knowledge/ingest`。凯伊侧先确定性注册人物/群节点（QQ 号做主键，昵称改名不分裂），
   再对窗口文本跑 LLM 抽取主题关系。
3. **查询**：
   - `kairos.query <关键词>` / `凯伊查询 <关键词>`：知识片段 + 图谱关联，毫秒级返回。
   - `kairos.status`：文档/实体/关系统计。
   - `kairos.flush`：立即推送（权限 ≥ 2）。

## 配置

```yaml
kairos-collector:on:
  groups:
    - '830070676'
  kairosEndpoint: http://kairos:8095
  apiToken: ''
  flushCount: 40
  flushInterval: 300
  batchSize: 200
```

> 默认关闭（`on:` 需显式启用）；`groups` 为空时不采集任何群。

## 安装（离线包）

```bash
docker exec -w /koishi koishi yarn add file:/koishi/_pkg/koishi-plugin-kairos-collector
```

前置条件：koishi 容器已加入 kairos 所在 Docker 网络（`docker network connect feishu-assistant_default koishi`）。
