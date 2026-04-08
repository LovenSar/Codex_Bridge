# Code_Bridge

本地 OpenAI / Anthropic 兼容 HTTP 桥，将 **Codex CLI** 和 **Claude Code CLI** 接到自定义上游 LLM（如 vLLM）。

- **18081 端口**（HTTPS）：OpenAI Responses API → 供 Codex CLI 使用
- **28082 端口**（HTTP）：Anthropic Messages API → 供 Claude Code CLI 使用
- 协议转换：Chat Completions ↔ Responses ↔ Messages
- 自动生成自签名 TLS 证书
- Thinking / Reasoning 模式支持

**启动**：`pip install -r bridge/requirements.txt && python chat_bridge.py`

详见 `README.md`。

## 版本记录

- 0.3.0（2026-04-08）：双端口架构，HTTPS + Claude Code 支持，WebSocket 优雅降级
- 0.2.0（2026-04-04）：Go 版桥接实验
- 0.1.0-alpha（2026-04-03）：初始版本
