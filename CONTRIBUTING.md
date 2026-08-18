# Contributing to Kairós

欢迎贡献！这是一个自托管个人项目，但欢迎 Issue、PR 和讨论。

## 开发环境

```bash
git clone https://github.com/DreamZhongJu/kairos-intel.git
cd kairos-intel
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 运行测试

```bash
python -m unittest discover -s tests -v
```

CI 在无凭据环境下自动运行。新增工具请同步更新 `tests/test_smoke.py` 中的工具计数。

## 提交规范

- 使用清晰的中文 / 英文 commit message
- feat: 新功能
- fix: 修复
- docs: 文档
- refactor: 重构

## 许可证

贡献代码接受 MIT 许可证。