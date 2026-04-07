# Code_Bridge

**生产用 `bridge/chat_bridge.py`（Python）**；`bridge/chat_bridge.go` 仅供对比，未与 Python 版对齐稳定性。

本地 OpenAI 兼容 HTTP 桥（默认 `127.0.0.1:18081`）：把 Codex CLI 等客户端接到仅提供部分形态的上游（如仅 Responses）。协议转换、端点、环境变量、Codex 信任与 profile、Qwen 长上下文等 **均以根目录 `README.md` 为准**。

- 协议：Chat Completions ↔ Responses 等路径的转换与代理。
- Codex：`.codex/config.toml` 自定义 provider，`supports_websockets = false`，走 HTTP。
- 可选：`plugin_spec.go`、`handler/` 为自有服务里注册运维路由的示例，主流程不依赖。

**启动与自检**：`pip install -r bridge/requirements.txt`，`python chat_bridge.py`；`curl /health`、`/v1/models`。参数与故障排查见 `README.md`。

**环境变量**：Python 桥为 `CODEX_BRIDGE_*`（完整表见 README）；Go 插件侧可选 `CODEX_CHAT_*` 等，语义以 `bridge/chat_bridge.py` 与对应 Go 源码为准。

**依赖**：Python 3.10+、可访问上游、Codex 联调验证版本 `codex-cli 0.118.0`（其它版本自行验证）。

## 版本记录

- 0.2.0（2026-04-04）：Go 版桥接实验
- 0.1.0-alpha（2026-04-03）：初始版本
