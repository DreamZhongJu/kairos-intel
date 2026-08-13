# 真实请求回放评测

- 评测模型：`deepseek-v4-flash`
- 回放样本：1 条真实请求

| 指标 | 数值 |
| --- | --- |
| 平均相关性（relevancy） | 1.000 |
| 平均完整性（completeness） | 1.000 |

说明：样本来自请求日志（`request_logs` 表 status=ok 的记录），使用 DeepSeek 作为裁判模型。