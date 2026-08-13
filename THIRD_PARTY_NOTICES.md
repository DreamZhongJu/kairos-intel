# Third-Party Notices

This file lists third-party software and services that Kairós depends on or
integrates with. It is maintained to satisfy open-source attribution and
license-notice obligations.

## Python dependencies

All direct runtime dependencies are declared in `requirements.txt` /
`pyproject.toml` and installed from PyPI. None are vendored into this
repository.

| Package | Purpose | License |
| --- | --- | --- |
| lark-oapi | Feishu / Lark SDK | MIT |
| openai | OpenAI-compatible model calls (DeepSeek) | Apache-2.0 |
| langgraph | Agent orchestration | MIT |
| langgraph-checkpoint-sqlite | Graph state checkpointing | MIT |
| requests | HTTP client | Apache-2.0 |
| duckduckgo-search | Web search | MIT |
| Flask | Local web panel | BSD-3-Clause |
| cryptography | Token encryption | Apache-2.0 / BSD |
| pypdf | PDF parsing | BSD-3-Clause |
| python-docx | DOCX parsing | MIT |
| python-pptx | PPTX parsing | MIT |
| python-dotenv | Environment loading | BSD-3-Clause |
| mcp | Model Context Protocol client | MIT |

Transitive dependency licenses should be verified before release using a
license checker (e.g. `pip-licenses` or `licensechecker`).

## External services and tools

These are optional integrations called over HTTP or subprocess. They are not
distributed with this repository; each requires its own setup and credentials.

| Project | Purpose | Homepage / repo |
| --- | --- | --- |
| Agent-Reach | Retrieval capability | (add link) |
| Claude-Mem | Long-term memory backend | (add link) |
| Exa | Semantic web search | https://exa.ai |
| MCPorter / rdt-cli | MCP tool bridging | (add link) |
| markitdown | File parsing skill (deployment only) | (add link) |

## Notes

- Do not commit third-party skills or copied code into this repository.
- If any third-party code must be vendored, preserve its original LICENSE,
  copyright notice, and source URL, and list it in this file.
