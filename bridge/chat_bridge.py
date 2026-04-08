#!/usr/bin/env python3
"""
Chat Completions -> Responses bridge for OpenAI-compatible backends.

This bridge accepts /v1/chat/completions requests and translates them to
/v1/responses for upstream servers that only support the Responses API.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_BRIDGE_RUN_SHA256: str | None = None
_BRIDGE_LOG_PATH: Path | None = None
_BRIDGE_INTERACTION_JSONL_PATH: Path | None = None
_LOG_FILE_LOCK = threading.Lock()
_LOG_MAX_PAYLOAD_STR = int(os.getenv("CODEX_BRIDGE_LOG_MAX_STR", "500000"))
# 对话/思考完整落盘与 JSONL 单行上限（默认 1GB 量级，避免误截断）
_LOG_DIALOGUE_JSONL_MAX_STR = int(os.getenv("CODEX_BRIDGE_LOG_DIALOGUE_JSONL_MAX_STR", str(10**9)))
_FULL_TRANSCRIPT_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_LOG_FULL_TRANSCRIPT", "true")).strip().lower() in ("1", "true", "yes", "on")
)
_CONSOLE_TRANSCRIPT_CHARS = int(os.getenv("CODEX_BRIDGE_LOG_CONSOLE_TRANSCRIPT_CHARS", "0"))
# 在运行本进程的终端直接打印「用户说了什么 / 思考 / 助手回了什么」（默认开启）
_CONSOLE_DIALOGUE_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_LOG_CONSOLE_DIALOGUE", "true")).strip().lower()
    in ("1", "true", "yes", "on")
)
_CONSOLE_DIALOGUE_MAX_CHARS = int(os.getenv("CODEX_BRIDGE_LOG_CONSOLE_DIALOGUE_MAX_CHARS", "50000"))
_CONSOLE_DIALOGUE_BANNER_SHOWN = False

_BRIDGE_TRANSCRIPT_PATH: Path | None = None


def _ensure_bridge_session_logging() -> None:
    """Once per process: SHA256 run id + log file paths under CODEX_BRIDGE_LOG_DIR (default: ../logs)."""
    global _BRIDGE_RUN_SHA256, _BRIDGE_LOG_PATH, _BRIDGE_INTERACTION_JSONL_PATH, _BRIDGE_TRANSCRIPT_PATH
    with _LOG_FILE_LOCK:
        if _BRIDGE_RUN_SHA256 is not None:
            return
        _BRIDGE_RUN_SHA256 = hashlib.sha256(os.urandom(32)).hexdigest()
        log_dir = Path(
            os.environ.get(
                "CODEX_BRIDGE_LOG_DIR",
                str(Path(__file__).resolve().parent.parent / "logs"),
            )
        )
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            log_dir = Path.cwd() / "codex_bridge_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
        _BRIDGE_LOG_PATH = log_dir / f"bridge-{_BRIDGE_RUN_SHA256}.log"
        _BRIDGE_INTERACTION_JSONL_PATH = log_dir / f"bridge-{_BRIDGE_RUN_SHA256}.interaction.jsonl"
        _BRIDGE_TRANSCRIPT_PATH = log_dir / f"bridge-{_BRIDGE_RUN_SHA256}.transcript.log"
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        banner = (
            f"{ts}\tbridge_session_start\trun_sha256={_BRIDGE_RUN_SHA256}\t"
            f"log={_BRIDGE_LOG_PATH.name}\tinteraction={_BRIDGE_INTERACTION_JSONL_PATH.name}\t"
            f"transcript={_BRIDGE_TRANSCRIPT_PATH.name}\n"
        )
        try:
            with open(_BRIDGE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(banner)
        except Exception:
            pass


def _truncate_for_log(obj: Any, max_str: int | None = None) -> Any:
    """Limit huge strings/lists for JSONL (avoid multi-GB files)."""
    lim = max_str if max_str is not None else _LOG_MAX_PAYLOAD_STR
    if isinstance(obj, str):
        if len(obj) <= lim:
            return obj
        return obj[: max(0, lim - 24)] + "\n...[truncated]..."
    if isinstance(obj, list):
        if len(obj) > 500:
            return [_truncate_for_log(x, lim) for x in obj[:500]] + ["...[truncated list]..."]
        return [_truncate_for_log(x, lim) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _truncate_for_log(v, lim) for k, v in obj.items()}
    return obj


def _log_interaction_json(req_id: str, event: str, data: Any, *, max_str: int | None = None) -> None:
    """Append one JSON line per event for full replay / audit (interaction.jsonl)."""
    _ensure_bridge_session_logging()
    if not _BRIDGE_INTERACTION_JSONL_PATH or not _BRIDGE_RUN_SHA256:
        return
    lim = max_str if max_str is not None else _LOG_MAX_PAYLOAD_STR
    record = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "run_sha256": _BRIDGE_RUN_SHA256,
        "req_id": req_id,
        "event": event,
        "data": _truncate_for_log(data, lim),
    }
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    try:
        with _LOG_FILE_LOCK:
            with open(_BRIDGE_INTERACTION_JSONL_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass


def bridge_run_sha256() -> str | None:
    return _BRIDGE_RUN_SHA256


def _log(message: str) -> None:
    _ensure_bridge_session_logging()
    ts = datetime.datetime.now().isoformat(timespec="milliseconds")
    rid = _BRIDGE_RUN_SHA256[:16] if _BRIDGE_RUN_SHA256 else "----------------"
    console = f"[bridge][run:{rid}] {message}"
    try:
        print(console, flush=True)
    except UnicodeEncodeError:
        print(console.encode("utf-8", errors="replace").decode("utf-8", errors="replace"), flush=True)
    if _BRIDGE_LOG_PATH:
        try:
            line = f"{ts}\t[run:{rid}]\t{message}\n"
            with _LOG_FILE_LOCK:
                with open(_BRIDGE_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass


def _json_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return str(obj)


def _extract_inbound_text_for_transcript(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    return _json_pretty(input_value)


def _format_upstream_assistant_message(msg: dict[str, Any]) -> str:
    if not isinstance(msg, dict):
        return str(msg)
    parts: list[str] = []
    for key in ("reasoning_content", "reasoning", "content"):
        if key in msg and msg.get(key) is not None:
            val = msg.get(key)
            s = val if isinstance(val, str) else _json_pretty(val)
            parts.append(f"--- {key} ---\n{s}")
    if not parts:
        return _json_pretty(msg)
    return "\n\n".join(parts)


def _extract_user_text_for_console(inbound_input: Any) -> str:
    """从 Responses input 或 messages 列表中抽出用户可见文本。"""
    if inbound_input is None:
        return ""
    if isinstance(inbound_input, str):
        return inbound_input.strip()
    if not isinstance(inbound_input, list):
        return _json_pretty(inbound_input)
    parts: list[str] = []
    for item in inbound_input:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role != "user":
            continue
        c = item.get("content")
        if isinstance(c, str):
            parts.append(c.strip())
        elif isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                pt = str(p.get("type") or "").strip().lower()
                if pt in ("input_text", "text"):
                    parts.append(str(p.get("text") or p.get("input_text") or "").strip())
        else:
            parts.append(str(c or "").strip())
    return "\n---\n".join(parts) if parts else _json_pretty(inbound_input)


def _reasoning_plain_text_from_item(item: dict[str, Any]) -> str:
    """提取 Responses output 中 reasoning 条目的可读文本（兼容官方 reasoning_text 与旧版顶层 text）。"""
    if not isinstance(item, dict):
        return ""
    parts: list[str] = []
    for p in item.get("content") or []:
        if not isinstance(p, dict):
            continue
        pt = str(p.get("type") or "").strip().lower()
        if pt == "reasoning_text":
            parts.append(str(p.get("text") or ""))
        elif pt in ("text", "output_text"):
            parts.append(str(p.get("text") or p.get("output_text") or ""))
    if parts:
        return "".join(parts)
    t = item.get("text")
    if isinstance(t, str) and t.strip():
        return t
    return ""


def _reasoning_reply_from_responses_dict(d: dict[str, Any]) -> tuple[str, str]:
    """从上游 Responses JSON 拆出思考与正文。"""
    reason = ""
    reply = ""
    for item in d.get("output") or []:
        if not isinstance(item, dict):
            continue
        it = str(item.get("type") or "").strip().lower()
        if it == "reasoning":
            reason = _reasoning_plain_text_from_item(item)
        elif it == "message":
            for p in item.get("content") or []:
                if isinstance(p, dict) and str(p.get("type") or "") in ("output_text", "text"):
                    reply += str(p.get("text") or p.get("output_text") or "")
    return reason.strip(), reply.strip()


def _emit_console_dialogue_summary(
    req_id: str,
    phase: str,
    inbound_input: Any,
    upstream_chat_completion: dict[str, Any] | None,
    converted_responses: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> None:
    """在运行 bridge 的终端打印用户输入、思考、回复（Codex 所在终端不会显示这里）。"""
    global _CONSOLE_DIALOGUE_BANNER_SHOWN
    if not _CONSOLE_DIALOGUE_ENABLED:
        return

    user_txt = _extract_user_text_for_console(inbound_input)
    reason = ""
    reply = ""

    if isinstance(upstream_chat_completion, dict):
        ch = upstream_chat_completion.get("choices") or []
        if ch and isinstance(ch[0], dict):
            msg = ch[0].get("message")
            if isinstance(msg, dict):
                reason = str(msg.get("reasoning_content") or msg.get("reasoning") or "")
                reply = str(msg.get("content") or "")

    if isinstance(converted_responses, dict):
        if not reason or not reply:
            r2, p2 = _reasoning_reply_from_responses_dict(converted_responses)
            if not reason:
                reason = r2
            if not reply:
                reply = p2

    if isinstance(extra, dict):
        uj = extra.get("upstream_responses_json")
        if isinstance(uj, dict) and (not reason or not reply):
            r3, p3 = _reasoning_reply_from_responses_dict(uj)
            if not reason:
                reason = r3
            if not reply:
                reply = p3
        atm = extra.get("assistant_text_merged")
        if isinstance(atm, str) and atm.strip():
            reply = atm
        ocr = extra.get("openai_chat_completion_result")
        if isinstance(ocr, dict):
            ch = ocr.get("choices") or []
            if ch and isinstance(ch[0], dict):
                msg = ch[0].get("message")
                if isinstance(msg, dict) and not reply:
                    reply = str(msg.get("content") or "")

    reply_st = reply.strip() if isinstance(reply, str) else str(reply or "").strip()
    reason_st = reason.strip() if isinstance(reason, str) else str(reason or "").strip()

    lines = [
        "",
        "╔" + "═" * 70 + "╗",
        f"║ bridge 对话摘要  req={req_id}  {phase}",
        "╚" + "═" * 70 + "╝",
        "[用户发送]",
        user_txt if user_txt.strip() else "(无)",
        "",
    ]
    if reason_st:
        lines.append("[思考 / reasoning]")
        lines.append(reason_st)
        lines.append("")
    lines.append("[助手回复]")
    lines.append(reply_st if reply_st else "(无)")
    lines.append("")
    text = "\n".join(lines)
    if len(text) > _CONSOLE_DIALOGUE_MAX_CHARS:
        text = text[: _CONSOLE_DIALOGUE_MAX_CHARS] + "\n...[truncated，完整内容见 .transcript.log]"

    if not _CONSOLE_DIALOGUE_BANNER_SHOWN:
        _CONSOLE_DIALOGUE_BANNER_SHOWN = True
        _log(
            "提示：以下 [用户发送]/[思考]/[助手回复] 出现在「运行 python chat_bridge.py 的终端」。"
            "Codex 界面是另一个进程，不会在这里显示。"
        )
    _log(text)


def _responses_output_plain_text(converted: dict[str, Any]) -> str:
    out = converted.get("output") or []
    parts: list[str] = []
    for item in out:
        if not isinstance(item, dict):
            continue
        it = str(item.get("type") or "").strip().lower()
        if it == "reasoning":
            parts.append("【思考 / reasoning】\n" + _reasoning_plain_text_from_item(item))
        elif it == "message":
            c = item.get("content")
            if isinstance(c, list):
                for p in c:
                    if isinstance(p, dict) and str(p.get("type") or "") in ("output_text", "text"):
                        parts.append(
                            "【助手正文】\n"
                            + str(p.get("text") or p.get("output_text") or "")
                        )
        elif it == "function_call":
            parts.append("【function_call】\n" + _json_pretty(item))
    return "\n\n".join(parts) if parts else _json_pretty(converted)


def _log_dialogue_transcript(
    req_id: str,
    phase: str,
    *,
    inbound_input: Any = None,
    chat_payload: dict[str, Any] | None = None,
    upstream_chat_completion: dict[str, Any] | None = None,
    converted_responses: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """终端摘要 + 将完整内容写入 .transcript.log / JSONL。"""
    try:
        _emit_console_dialogue_summary(
            req_id,
            phase,
            inbound_input,
            upstream_chat_completion,
            converted_responses,
            extra,
        )
    except Exception as exc:
        _log(f"req={req_id} console_dialogue_summary_error: {_short_text(exc)}")

    if not _FULL_TRANSCRIPT_ENABLED:
        return
    _ensure_bridge_session_logging()
    if not _BRIDGE_TRANSCRIPT_PATH:
        return
    ts = datetime.datetime.now().isoformat(timespec="milliseconds")
    block_parts: list[str] = [
        f"\n{'=' * 72}\n{ts}  req={req_id}  {phase}\n{'=' * 72}",
    ]
    if inbound_input is not None:
        block_parts.append("\n--- inbound input（完整）---\n")
        block_parts.append(_extract_inbound_text_for_transcript(inbound_input))
    if chat_payload is not None:
        block_parts.append("\n--- 发往上游 chat/completions（完整 JSON）---\n")
        block_parts.append(_json_pretty(chat_payload))
    if upstream_chat_completion is not None:
        block_parts.append("\n--- 上游 chat/completions 响应（完整 JSON）---\n")
        block_parts.append(_json_pretty(upstream_chat_completion))
        choices = upstream_chat_completion.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg0 = choices[0].get("message")
            if isinstance(msg0, dict):
                block_parts.append("\n--- 上游 assistant 消息（纯文本拆分）---\n")
                block_parts.append(_format_upstream_assistant_message(msg0))
    if converted_responses is not None:
        block_parts.append("\n--- 转换后的 Responses（完整 JSON）---\n")
        block_parts.append(_json_pretty(converted_responses))
        block_parts.append("\n--- 转换后 output 可读摘要（思考+正文）---\n")
        block_parts.append(_responses_output_plain_text(converted_responses))
    if extra:
        block_parts.append("\n--- extra ---\n")
        block_parts.append(_json_pretty(extra))
    block_parts.append("\n")
    block = "".join(block_parts)
    try:
        with _LOG_FILE_LOCK:
            with open(_BRIDGE_TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
                f.write(block)
    except Exception:
        pass

    total_chars = len(block)
    _log(
        f"req={req_id} dialogue_transcript phase={phase} transcript_chars={total_chars} "
        f"file={_BRIDGE_TRANSCRIPT_PATH.name}"
    )
    if _CONSOLE_TRANSCRIPT_CHARS > 0:
        preview = block[:_CONSOLE_TRANSCRIPT_CHARS]
        _log(f"req={req_id} transcript_preview_begin\n{preview}")
        if len(block) > _CONSOLE_TRANSCRIPT_CHARS:
            _log(f"req={req_id} transcript_preview_truncated total_chars={len(block)}")

    _log_interaction_json(
        req_id,
        f"dialogue.transcript.{phase}",
        {
            "inbound_input": inbound_input,
            "chat_payload": chat_payload,
            "upstream_chat_completion": upstream_chat_completion,
            "converted_responses": converted_responses,
            "extra": extra,
        },
        max_str=_LOG_DIALOGUE_JSONL_MAX_STR,
    )


RESPONSES_COMPAT_CHAT_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_RESPONSES_COMPAT_CHAT", "true")).strip().lower()
    in ("1", "true", "yes", "on")
)

# Some /v1/responses backends only accept input as a plain string, not a message list.
RESPONSES_INPUT_AS_STRING_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_RESPONSES_INPUT_AS_STRING", "false")).strip().lower()
    in ("1", "true", "yes", "on")
)
# On upstream 400, retry once with list input flattened to a string (matches common vLLM validation).
RESPONSES_INPUT_STRING_RETRY_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_RESPONSES_INPUT_STRING_RETRY", "true")).strip().lower()
    in ("1", "true", "yes", "on")
)

# When tools + stream: try upstream chat/completions with stream, translate SSE to Responses-style (Codex).
RESPONSES_COMPAT_CHAT_STREAM_ENABLED = (
    str(os.getenv("CODEX_BRIDGE_COMPAT_CHAT_STREAM", "true")).strip().lower()
    in ("1", "true", "yes", "on")
)

# /v1/responses compat：上游为非流式 chat.completions，回包后再发 SSE。若为 true，则合成
# response.created / output_item.added / reasoning_text.delta / output_text.delta 等事件，供 Codex TUI 渲染。
CODEX_BRIDGE_TUI_SSE_COMPAT = (
    str(os.getenv("CODEX_BRIDGE_TUI_SSE_COMPAT", "true")).strip().lower()
    in ("1", "true", "yes", "on")
)

MAX_TOOL_CALL_ROUNDS = int(os.getenv("CODEX_BRIDGE_MAX_TOOL_CALL_ROUNDS", "40"))

# Model aliasing: map Codex internal model names to your actual model
# Format: comma-separated pairs like "gpt-5.4-mini=Qwen/Qwen3.5-27B,gpt-4o=Qwen/Qwen3.5-27B"
_MODEL_ALIAS_MAP: dict[str, str] = {}
_alias_config = os.getenv("CODEX_BRIDGE_MODEL_ALIASES", "").strip()
if _alias_config:
    for pair in _alias_config.split(","):
        if "=" in pair:
            src, dst = pair.split("=", 1)
            _MODEL_ALIAS_MAP[src.strip()] = dst.strip()
else:
    # Default: map common GPT models to Qwen
    _MODEL_ALIAS_MAP = {
        "gpt-5.4-mini": "Qwen/Qwen3.5-27B",
        "gpt-5-mini": "Qwen/Qwen3.5-27B",
        "gpt-4o": "Qwen/Qwen3.5-27B",
        "gpt-4o-mini": "Qwen/Qwen3.5-27B",
        "gpt-4": "Qwen/Qwen3.5-27B",
        "gpt-3.5-turbo": "Qwen/Qwen3.5-27B",
    }


def _resolve_model_name(raw_model: str, default_model: str) -> tuple[str, str, bool]:
    """
    Resolve the actual model name with aliasing.
    Returns: (final_model, model_source, was_aliased)
    """
    if not raw_model or not raw_model.strip():
        return default_model, "default", False
    
    model = raw_model.strip()
    was_aliased = False
    
    # Check if model should be aliased
    if model in _MODEL_ALIAS_MAP:
        original = model
        model = _MODEL_ALIAS_MAP[model]
        was_aliased = True
        _log(f"model_alias: {original} -> {model}")
    
    return model, "payload", was_aliased


def _count_tool_call_rounds(input_value: Any) -> int:
    """Count consecutive function_call/function_call_output pairs in the input."""
    if not isinstance(input_value, list):
        return 0
    rounds = 0
    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "function_call":
            rounds += 1
    return rounds


def _short_text(value: Any, limit: int = 180) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _role_counts(messages: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for msg in messages or []:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "unknown").strip().lower() or "unknown"
        else:
            role = "unknown"
        counts[role] = counts.get(role, 0) + 1
    if not counts:
        return "-"
    parts = [f"{k}:{counts[k]}" for k in sorted(counts.keys())]
    return ",".join(parts)


def _strip_trailing_slash(url: str) -> str:
    return url.rstrip("/")


def _normalize_base_urls(base_url: str) -> tuple[str, str, str, str]:
    """
    Return:
      - base_v1: upstream base ending with /v1
      - responses_url: /v1/responses endpoint
      - models_url: /v1/models endpoint
      - chat_completions_url: /v1/chat/completions endpoint
    """
    base = _strip_trailing_slash(base_url)
    if base.endswith("/v1"):
        return base, f"{base}/responses", f"{base}/models", f"{base}/chat/completions"
    return f"{base}/v1", f"{base}/v1/responses", f"{base}/v1/models", f"{base}/v1/chat/completions"


def _extract_text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            text = item.get("text")
            if text is not None:
                parts.append(str(text))
                continue
            if item.get("type") in ("input_text", "output_text", "text"):
                maybe_text = item.get("input_text") or item.get("output_text")
                if maybe_text is not None:
                    parts.append(str(maybe_text))
        return "\n".join([p for p in parts if p])
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _flatten_responses_input_to_string(input_value: Any) -> str:
    """
    Collapse Responses-style input (list of role/content items) into one string for
    upstreams whose schema only allows input: str.
    """
    if isinstance(input_value, str):
        return input_value
    if input_value is None:
        return ""
    if not isinstance(input_value, list):
        return _extract_text_from_content(input_value)

    chunks: list[str] = []
    for item in input_value:
        if not isinstance(item, dict):
            t = str(item).strip()
            if t:
                chunks.append(t)
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "function_call":
            name = str(item.get("name") or "").strip()
            raw_args: Any = item.get("arguments")
            if isinstance(raw_args, (dict, list)):
                args_s = json.dumps(raw_args, ensure_ascii=False)
            elif raw_args is None:
                args_s = "{}"
            else:
                args_s = str(raw_args)
            label = name or "tool"
            chunks.append(f"[function_call:{label}]\n{args_s}")
            continue
        if item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = str(item.get("call_id") or item.get("id") or "").strip()
            out_txt = _responses_output_to_text(item.get("output"))
            prefix = f"[function_output:{call_id}]" if call_id else "[function_output]"
            chunks.append(f"{prefix}\n{out_txt}")
            continue

        role = _response_item_role(item) or "user"
        text = _response_item_content_text(item).strip()
        if not text:
            continue
        chunks.append(f"{role.upper()}:\n{text}")

    return "\n\n".join(chunks)


def _error_suggests_string_input_required(message: str, data: Any) -> bool:
    """True if upstream validation rejected list-shaped input in favor of a string."""
    text_blob = str(message or "")
    try:
        text_blob += "\n" + json.dumps(data, ensure_ascii=False)
    except Exception:
        text_blob += "\n" + str(data)

    if "string_type" in text_blob and "input" in text_blob.lower():
        return True
    low = text_blob.lower()
    if "valid string" in low and "input" in low:
        return True

    def _walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            if obj.get("type") == "string_type":
                loc = obj.get("loc")
                if isinstance(loc, (list, tuple)):
                    loc_s = ".".join(str(x) for x in loc)
                    if "input" in loc_s and "str" in loc_s:
                        return True
            for v in obj.values():
                if _walk(v):
                    return True
        elif isinstance(obj, list):
            for it in obj:
                if _walk(it):
                    return True
        return False

    return _walk(data)


def _chat_messages_to_responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        role = _normalize_response_role(msg.get("role"))
        content = _extract_text_from_content(msg.get("content"))
        if content is None:
            content = ""
        items.append({"role": role, "content": content})
    if items:
        items = _coalesce_leading_system_messages(items)
    if not items:
        items.append({"role": "user", "content": "Hello"})
    return items


def _normalize_response_role(role: Any) -> str:
    r = str(role or "user").strip().lower()
    # Some OpenAI-compatible backends reject "developer" role in /responses.
    if r == "developer":
        return "system"
    if r in ("system", "user", "assistant", "tool"):
        return r
    return "user"


def _response_item_role(item: dict[str, Any]) -> str:
    role = str(item.get("role") or "").strip().lower()
    if not role and isinstance(item.get("message"), dict):
        role = str((item.get("message") or {}).get("role") or "").strip().lower()
    if not role:
        return ""
    return _normalize_response_role(role)


def _response_item_content_text(item: dict[str, Any]) -> str:
    content: Any = item.get("content")
    if content is None and isinstance(item.get("message"), dict):
        content = (item.get("message") or {}).get("content")
    return _extract_text_from_content(content)


_UTF8_FILE_TREE_COMMAND = (
    "Get-ChildItem -Recurse -Force | "
    "Select-Object FullName | "
    "ForEach-Object { $_.FullName.Replace((Get-Location).Path + '\\\\', '') }"
)


def _normalize_function_arguments_json(raw: Any) -> tuple[str, Any, bool]:
    """Return (json_text, parsed_value, repaired)."""
    if isinstance(raw, (dict, list)):
        parsed = raw
        return json.dumps(parsed, ensure_ascii=False), parsed, False

    if raw is None:
        return "{}", {}, True

    text = str(raw).strip()
    if not text:
        return "{}", {}, True

    try:
        parsed = json.loads(text)
        return json.dumps(parsed, ensure_ascii=False), parsed, False
    except Exception:
        pass

    parsed_loose = _parse_loose_tool_call_text(text)
    if parsed_loose:
        _, args = parsed_loose
        return json.dumps(args, ensure_ascii=False), args, True

    return "{}", {}, True


def _rewrite_utf8_file_tree_command(name: str, args_obj: Any) -> tuple[Any, bool]:
    if str(name or "").strip() != "shell_command":
        return args_obj, False
    if not isinstance(args_obj, dict):
        return args_obj, False
    cmd = str(args_obj.get("command") or "")
    if not re.match(r"^\s*tree(\s|$)", cmd, re.IGNORECASE):
        return args_obj, False
    out = dict(args_obj)
    out["command"] = _UTF8_FILE_TREE_COMMAND
    return out, True


def _sanitize_function_call_item(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out = dict(item)
    changed = False
    if "content" in out:
        out.pop("content", None)
        changed = True
    if "message" in out:
        out.pop("message", None)
        changed = True

    name = str(out.get("name") or "").strip()
    if not name and isinstance(out.get("function"), dict):
        name = str((out.get("function") or {}).get("name") or "").strip()
    if name and out.get("name") != name:
        out["name"] = name
        changed = True

    raw_args: Any = out.get("arguments")
    if raw_args is None and isinstance(out.get("function"), dict):
        raw_args = (out.get("function") or {}).get("arguments")
    args_json, parsed_args, repaired = _normalize_function_arguments_json(raw_args)
    if repaired:
        changed = True

    parsed_args, rewritten = _rewrite_utf8_file_tree_command(name, parsed_args)
    if rewritten:
        args_json = json.dumps(parsed_args, ensure_ascii=False)
        changed = True

    if out.get("arguments") != args_json:
        out["arguments"] = args_json
        changed = True

    return out, changed


def _coalesce_leading_system_messages(items: list[Any]) -> list[Any]:
    """Keep at most one leading system message for strict chat templates."""
    system_items: list[dict[str, Any]] = []
    other_items: list[Any] = []
    for item in items:
        if isinstance(item, dict) and _response_item_role(item) == "system":
            system_items.append(item)
            continue
        other_items.append(item)

    if not system_items:
        return items

    merged_parts: list[str] = []
    for sys_item in system_items:
        text = _response_item_content_text(sys_item).strip()
        if text:
            merged_parts.append(text)
    merged_text = "\n\n".join(merged_parts)

    merged_system = dict(system_items[0])
    if isinstance(merged_system.get("message"), dict):
        msg = dict(merged_system.get("message") or {})
        msg["role"] = "system"
        msg["content"] = merged_text
        merged_system["message"] = msg
    merged_system["role"] = "system"
    merged_system["content"] = merged_text

    return [merged_system] + other_items


def _fold_responses_instructions_into_input(payload: dict[str, Any]) -> dict[str, Any]:
    """Move top-level instruction-like fields into input[0].role=system."""
    out = dict(payload)
    instruction_parts: list[str] = []
    for key in ("instructions", "system"):
        if key in out:
            text = _extract_text_from_content(out.get(key)).strip()
            if text:
                instruction_parts.append(text)
            out.pop(key, None)
    if not instruction_parts:
        return out

    instruction_text = "\n\n".join(instruction_parts)
    current_input = out.get("input")
    if isinstance(current_input, list):
        out["input"] = [{"role": "system", "content": instruction_text}] + current_input
        return out

    if current_input is None:
        out["input"] = [{"role": "system", "content": instruction_text}]
        return out

    if isinstance(current_input, dict):
        out["input"] = [{"role": "system", "content": instruction_text}, current_input]
        return out

    out["input"] = [
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": _extract_text_from_content(current_input)},
    ]
    return out


def _sanitize_responses_input(value: Any) -> Any:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.append(_sanitize_responses_input(item))
        return _coalesce_leading_system_messages(out)
    if isinstance(value, dict):
        out = dict(value)
        item_type = str(out.get("type") or "").strip().lower()
        if item_type == "function_call":
            fixed, _ = _sanitize_function_call_item(out)
            out = fixed
        if "role" in out:
            out["role"] = _normalize_response_role(out.get("role"))
        if "content" in out:
            out["content"] = _sanitize_responses_input(out.get("content"))
        if "message" in out and isinstance(out.get("message"), dict):
            msg = dict(out.get("message") or {})
            if "role" in msg:
                msg["role"] = _normalize_response_role(msg.get("role"))
            if "content" in msg:
                msg["content"] = _sanitize_responses_input(msg.get("content"))
            out["message"] = msg
        return out
    return value


def _strip_strict_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_strip_strict_fields(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if str(k) == "strict":
                continue
            out[k] = _strip_strict_fields(v)
        return out
    return value


def _response_input_role_counts(input_value: Any) -> str:
    counts: dict[str, int] = {}
    if isinstance(input_value, list):
        for item in input_value:
            if isinstance(item, dict):
                role = _response_item_role(item) or "unknown"
                counts[role] = counts.get(role, 0) + 1
    if not counts:
        return "-"
    parts = [f"{k}:{counts[k]}" for k in sorted(counts.keys())]
    return ",".join(parts)


def _response_input_role_sequence(input_value: Any, limit: int = 16) -> str:
    if not isinstance(input_value, list) or not input_value:
        return "-"
    seq: list[str] = []
    for idx, item in enumerate(input_value):
        if idx >= max(0, limit):
            seq.append("...")
            break
        if isinstance(item, dict):
            role = _response_item_role(item) or "unknown"
            itype = str(item.get("type") or "").strip().lower()
            if itype:
                seq.append(f"{idx}:{role}/{itype}")
            else:
                seq.append(f"{idx}:{role}")
            continue
        seq.append(f"{idx}:non-dict")
    if not seq:
        return "-"
    return ",".join(seq)


def _tool_schema_summary(tools_value: Any, limit: int = 8) -> str:
    if not isinstance(tools_value, list) or not tools_value:
        return "-"
    parts: list[str] = []
    for idx, tool in enumerate(tools_value):
        if idx >= max(0, limit):
            parts.append("...")
            break
        if not isinstance(tool, dict):
            parts.append(f"{idx}:non-dict")
            continue
        ttype = str(tool.get("type") or "").strip().lower() or "unknown"
        name = ""
        if isinstance(tool.get("function"), dict):
            name = str((tool.get("function") or {}).get("name") or "").strip()
        if not name:
            name = str(tool.get("name") or "").strip()
        if name:
            parts.append(f"{idx}:{ttype}/{name}")
        else:
            parts.append(f"{idx}:{ttype}")
    return ",".join(parts) if parts else "-"


def _responses_output_summary(resp: Any, limit: int = 4) -> str:
    if not isinstance(resp, dict):
        return "-"
    output = resp.get("output")
    if not isinstance(output, list):
        return "output=-"
    parts: list[str] = []
    for idx, item in enumerate(output):
        if idx >= max(0, limit):
            parts.append("...")
            break
        if not isinstance(item, dict):
            parts.append(f"{idx}:non-dict")
            continue
        item_type = str(item.get("type") or "").strip().lower() or "unknown"
        name = ""
        if item_type == "function_call":
            name = str(item.get("name") or "").strip()
        snippet = _short_text(
            _responses_output_to_text(item.get("content") if "content" in item else item),
            40,
        )
        if name:
            parts.append(f"{idx}:{item_type}/{name}")
        elif snippet:
            parts.append(f"{idx}:{item_type}:{snippet}")
        else:
            parts.append(f"{idx}:{item_type}")
    return ",".join(parts) if parts else "output=[]"


def _responses_output_full_lines(resp: Any, max_items: int = 64) -> list[dict[str, Any]]:
    """Structured full text per output item for JSONL (interaction replay)."""
    if not isinstance(resp, dict):
        return []
    output = resp.get("output")
    if not isinstance(output, list):
        return []
    lines: list[dict[str, Any]] = []
    for idx, item in enumerate(output):
        if idx >= max_items:
            lines.append({"index": idx, "note": "truncated", "max_items": max_items})
            break
        if not isinstance(item, dict):
            lines.append({"index": idx, "raw": str(item)[: _LOG_MAX_PAYLOAD_STR]})
            continue
        item_type = str(item.get("type") or "").strip().lower() or "unknown"
        text = _responses_output_to_text(item.get("content") if "content" in item else item)
        row: dict[str, Any] = {
            "index": idx,
            "type": item_type,
            "text": text,
        }
        if item_type == "function_call":
            row["name"] = str(item.get("name") or "")
            row["arguments"] = str(item.get("arguments") or "")[:_LOG_MAX_PAYLOAD_STR]
        lines.append(row)
    return lines


def _demote_system_roles(value: Any) -> Any:
    if isinstance(value, list):
        return [_demote_system_roles(item) for item in value]
    if isinstance(value, dict):
        out = dict(value)
        role = str(out.get("role") or "").strip().lower()
        if role in ("system", "developer"):
            out["role"] = "user"
        if "content" in out:
            out["content"] = _demote_system_roles(out.get("content"))
        if "message" in out and isinstance(out.get("message"), dict):
            msg = dict(out.get("message") or {})
            msg_role = str(msg.get("role") or "").strip().lower()
            if msg_role in ("system", "developer"):
                msg["role"] = "user"
            if "content" in msg:
                msg["content"] = _demote_system_roles(msg.get("content"))
            out["message"] = msg
        return out
    return value


def _extract_upstream_error_message(payload: Any, fallback: str = "") -> str:
    if isinstance(payload, dict):
        maybe_err = payload.get("error")
        if isinstance(maybe_err, dict):
            msg = str(maybe_err.get("message") or "").strip()
            if msg:
                return msg
        if isinstance(maybe_err, str) and maybe_err.strip():
            return maybe_err.strip()
        msg = str(payload.get("message") or "").strip()
        if msg:
            return msg
    return str(fallback or "").strip()


def _responses_output_to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            text = item.get("text")
            if text is not None:
                parts.append(str(text))
                continue
            for key in ("input_text", "output_text", "content"):
                if key in item and item.get(key) is not None:
                    parts.append(str(item.get(key)))
                    break
        return "\n".join([p for p in parts if p])
    if isinstance(value, dict):
        if "content" in value:
            content_val = value.get("content")
            if isinstance(content_val, str):
                return str(content_val)
            if content_val is not None:
                text_from_content = _responses_output_to_text(content_val)
                if text_from_content:
                    return text_from_content
        for key in ("text", "output_text", "input_text"):
            if key in value and value.get(key) is not None:
                return str(value.get(key))
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return ""
    return str(value)


_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNC_RE = re.compile(r"<function=([A-Za-z0-9_.:-]+)>", re.IGNORECASE)
_TOOL_PARAM_RE = re.compile(
    r"<parameter=([A-Za-z0-9_.:-]+)>(.*?)</parameter>", re.IGNORECASE | re.DOTALL
)
_SSE_EVENT_SPLIT_RE = re.compile(r"\r?\n\r?\n")


def _parse_loose_tool_call_text(text: str) -> tuple[str, dict[str, Any]] | None:
    raw = str(text or "")
    if "<tool_call>" not in raw.lower():
        return None

    block_match = _TOOL_CALL_BLOCK_RE.search(raw)
    block = block_match.group(1) if block_match else raw

    fn_match = _TOOL_FUNC_RE.search(block)
    if not fn_match:
        return None
    fn_name = str(fn_match.group(1) or "").strip()
    if not fn_name:
        return None

    args: dict[str, Any] = {}
    for p_name, p_raw in _TOOL_PARAM_RE.findall(block):
        key = str(p_name or "").strip()
        if not key:
            continue
        val = str(p_raw or "").strip()
        if not val:
            args[key] = ""
            continue
        low = val.lower()
        if low in ("true", "false"):
            args[key] = low == "true"
            continue
        if val[:1] in ('{', '[', '"'):
            try:
                args[key] = json.loads(val)
                continue
            except Exception:
                pass
        args[key] = val

    return fn_name, args


def _coerce_textual_tool_calls_to_function_calls(resp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(resp, dict):
        return resp, False
    output = resp.get("output")
    if not isinstance(output, list) or not output:
        return resp, False

    changed = False
    converted_output: list[Any] = []
    for item in output:
        if not isinstance(item, dict):
            converted_output.append(item)
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type == "function_call":
            converted_output.append(item)
            continue
        if (
            item_type not in ("message", "", "output_text", "text")
            and "content" not in item
            and "text" not in item
            and "output_text" not in item
            and "input_text" not in item
        ):
            converted_output.append(item)
            continue

        text_source: Any = item.get("content") if "content" in item else item
        text = _responses_output_to_text(text_source)
        parsed = _parse_loose_tool_call_text(text)
        if not parsed:
            converted_output.append(item)
            continue

        fn_name, args = parsed
        converted_output.append(
            {
                "type": "function_call",
                "id": f"fc_{uuid.uuid4().hex[:8]}",
                "call_id": f"call_{uuid.uuid4().hex[:8]}",
                "name": fn_name,
                "arguments": json.dumps(args, ensure_ascii=False),
                "status": "completed",
            }
        )
        changed = True

    if not changed:
        return resp, False

    out = dict(resp)
    out["output"] = converted_output
    return out, True


def _normalize_function_call_items(resp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    if not isinstance(resp, dict):
        return resp, False
    output = resp.get("output")
    if not isinstance(output, list) or not output:
        return resp, False

    changed = False
    has_function_call = False
    normalized_output: list[Any] = []
    seen_calls: set[tuple[str, str]] = set()
    for item in output:
        if not isinstance(item, dict):
            normalized_output.append(item)
            continue
        item_type = str(item.get("type") or "").strip().lower()
        if item_type != "function_call":
            normalized_output.append(item)
            continue

        has_function_call = True
        out_item, fixed_changed = _sanitize_function_call_item(item)
        if fixed_changed:
            changed = True
        name = str(out_item.get("name") or "").strip()

        call_id = str(out_item.get("call_id") or out_item.get("id") or "").strip()
        if not call_id:
            out_item["call_id"] = f"call_{uuid.uuid4().hex[:8]}"
            changed = True
        else:
            if out_item.get("call_id") != call_id:
                out_item["call_id"] = call_id
                changed = True
        item_id = str(out_item.get("id") or "").strip()
        if not item_id:
            out_item["id"] = f"fc_{uuid.uuid4().hex[:8]}"
            changed = True

        args = out_item.get("arguments")
        if not isinstance(args, str):
            out_item["arguments"] = json.dumps(args if args is not None else {}, ensure_ascii=False)
            changed = True
        if str(out_item.get("status") or "").strip().lower() != "completed":
            out_item["status"] = "completed"
            changed = True
        signature = (str(out_item.get("name") or ""), str(out_item.get("arguments") or ""))
        if signature in seen_calls:
            changed = True
            continue
        seen_calls.add(signature)
        normalized_output.append(out_item)

    if has_function_call:
        filtered_output: list[Any] = []
        for item in normalized_output:
            if not isinstance(item, dict):
                changed = True
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in ("function_call", "function_call_output", "custom_tool_call_output"):
                filtered_output.append(item)
                continue
            # When function_call exists, suppress assistant text placeholders so
            # Codex focuses on tool execution instead of returning the text block.
            changed = True
            continue
        normalized_output = filtered_output

    if not changed:
        return resp, False
    out = dict(resp)
    out["output"] = normalized_output
    return out, True


def _normalize_response_tool_outputs(resp: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    out, changed1 = _coerce_textual_tool_calls_to_function_calls(resp)
    out, changed2 = _normalize_function_call_items(out)
    return out, (changed1 or changed2)


def _extract_response_completed_from_sse_body(text: str) -> dict[str, Any] | None:
    raw = str(text or "")
    if not raw:
        return None
    for chunk in _SSE_EVENT_SPLIT_RE.split(raw):
        if not chunk:
            continue
        event_name = ""
        data_lines: list[str] = []
        for line in chunk.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:") :].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:") :].strip())
        if event_name != "response.completed" or not data_lines:
            # OpenAI-style streams usually encode the event type inside data.type.
            pass
        data_raw = "\n".join(data_lines).strip()
        if not data_raw or data_raw == "[DONE]":
            continue
        try:
            payload = json.loads(data_raw)
        except Exception:
            continue
        if isinstance(payload, dict) and str(payload.get("type") or "").strip().lower() == "response.completed":
            if isinstance(payload.get("response"), dict):
                return payload.get("response")
            return payload
        if event_name != "response.completed":
            continue
        if isinstance(payload, dict) and isinstance(payload.get("response"), dict):
            return payload.get("response")
        if isinstance(payload, dict):
            return payload
    return None


def _sse_event_lines_enabled() -> bool:
    """Codex / OpenAI 客户端常按 SSE 的 event: 名区分帧；仅发 data: 时可能只走默认 message，界面不渲染。"""
    return str(os.getenv("CODEX_BRIDGE_SSE_EVENT_LINES", "true")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _append_sse_json_lines(lines: list[str], payload: dict[str, Any]) -> None:
    """单条 SSE 事件：可选 event: <type>，再 data: {json}，与官方 Responses 流式对齐。"""
    data = json.dumps(payload, ensure_ascii=False, default=str)
    if _sse_event_lines_enabled():
        t = str(payload.get("type") or "").strip()
        if t:
            lines.append(f"event: {t}")
    lines.append(f"data: {data}")
    lines.append("")


def _sse_streaming_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _truncate_resp_completed_reasoning_for_sse(resp: dict[str, Any]) -> None:
    """response.completed 里超长 reasoning 可能撑爆 Codex 解析；默认截断（可用环境变量调大）。"""
    try:
        lim = int(os.getenv("CODEX_BRIDGE_COMPLETED_MAX_REASONING_CHARS", "65536"))
    except ValueError:
        lim = 65536
    if lim <= 0:
        return
    out = resp.get("output")
    if not isinstance(out, list):
        return
    for item in out:
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "reasoning":
            continue
        txt = item.get("text")
        if isinstance(txt, str) and len(txt) > lim:
            item["text"] = txt[: max(0, lim - 24)] + "\n...[truncated]..."
        for p in item.get("content") or []:
            if not isinstance(p, dict):
                continue
            if str(p.get("type") or "").strip().lower() != "reasoning_text":
                continue
            t = p.get("text")
            if isinstance(t, str) and len(t) > lim:
                p["text"] = t[: max(0, lim - 24)] + "\n...[truncated]..."


def _build_minimal_response_sse(response_obj: dict[str, Any]) -> str:
    response_payload = dict(response_obj) if isinstance(response_obj, dict) else {}
    response_id = str(response_payload.get("id") or "").strip()
    if not response_id:
        response_id = f"resp_{uuid.uuid4().hex[:8]}"
        response_payload["id"] = response_id

    events: list[dict[str, Any]] = []
    output = response_payload.get("output")
    if isinstance(output, list):
        for idx, item in enumerate(output):
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in ("function_call", "message", "reasoning"):
                events.append(
                    {
                        "type": "response.output_item.done",
                        "response_id": response_id,
                        "output_index": idx,
                        "item": item,
                    }
                )
    events.append({"type": "response.completed", "response": response_payload})

    lines: list[str] = []
    for payload in events:
        _append_sse_json_lines(lines, payload)
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _tui_sse_chunk_chars() -> int:
    try:
        v = int(os.getenv("CODEX_BRIDGE_TUI_SSE_CHUNK_CHARS", "48"))
    except ValueError:
        v = 48
    return max(1, min(v, 4096))


def _split_sse_text_chunks(text: str, chunk_size: int) -> list[str]:
    if not text:
        return []
    out: list[str] = []
    i = 0
    s = str(text)
    while i < len(s):
        out.append(s[i : i + chunk_size])
        i += chunk_size
    return out


def _message_item_output_text(item: dict[str, Any]) -> str:
    parts: list[str] = []
    for p in item.get("content") or []:
        if not isinstance(p, dict):
            continue
        if str(p.get("type") or "").strip().lower() in ("output_text", "text"):
            parts.append(str(p.get("text") or p.get("output_text") or ""))
    return "".join(parts)


def _build_codex_tui_compat_stream_sse(response_obj: dict[str, Any]) -> str:
    """
    在「非流式上游 + 最终 Responses 对象」场景下，合成与 OpenAI Responses 流式相近的事件序列，
    供 Codex TUI 依赖 delta 事件时仍能显示思考与正文。
    """
    response_payload = dict(response_obj) if isinstance(response_obj, dict) else {}
    response_id = str(response_payload.get("id") or "").strip()
    if not response_id:
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        response_payload["id"] = response_id
    model = str(response_payload.get("model") or "")
    created_at = int(response_payload.get("created_at") or time.time())
    chunk_sz = _tui_sse_chunk_chars()

    events: list[dict[str, Any]] = []
    seq = 0

    def emit(ev: dict[str, Any]) -> None:
        nonlocal seq
        payload = dict(ev)
        payload["sequence_number"] = seq
        seq += 1
        events.append(payload)

    emit(
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": created_at,
                "model": model,
                "status": "in_progress",
            },
        }
    )

    output = response_payload.get("output")
    if not isinstance(output, list):
        output = []

    for output_index, item in enumerate(output):
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()

        emit(
            {
                "type": "response.output_item.added",
                "response_id": response_id,
                "output_index": output_index,
                "item": item,
            }
        )

        if item_type == "reasoning":
            rid = str(item.get("id") or f"rs_{uuid.uuid4().hex[:10]}")
            rtext = _reasoning_plain_text_from_item(item)
            for piece in _split_sse_text_chunks(rtext, chunk_sz):
                emit(
                    {
                        "type": "response.reasoning_summary_text.delta",
                        "response_id": response_id,
                        "delta": piece,
                        "item_id": rid,
                        "output_index": output_index,
                        "summary_index": 0,
                    }
                )
            emit(
                {
                    "type": "response.reasoning_summary_text.done",
                    "response_id": response_id,
                    "item_id": rid,
                    "output_index": output_index,
                    "summary_index": 0,
                    "text": rtext,
                }
            )
        elif item_type == "message":
            mid = str(item.get("id") or f"msg_{uuid.uuid4().hex[:10]}")
            mtext = _message_item_output_text(item)
            for piece in _split_sse_text_chunks(mtext, chunk_sz):
                emit(
                    {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "delta": piece,
                        "item_id": mid,
                        "output_index": output_index,
                        "content_index": 0,
                    }
                )
            emit(
                {
                    "type": "response.output_text.done",
                    "response_id": response_id,
                    "item_id": mid,
                    "output_index": output_index,
                    "content_index": 0,
                    "text": mtext,
                }
            )
        elif item_type == "function_call":
            pass

        emit(
            {
                "type": "response.output_item.done",
                "response_id": response_id,
                "output_index": output_index,
                "item": item,
            }
        )

    emit({"type": "response.completed", "response": response_payload})

    lines: list[str] = []
    for payload in events:
        _append_sse_json_lines(lines, payload)
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _sse_data_line_bytes(payload: dict[str, Any]) -> bytes:
    lines: list[str] = []
    _append_sse_json_lines(lines, payload)
    return "\n".join(lines).encode("utf-8")


def _sse_pregenerated_text_as_streaming_response(sse_body: str) -> Any:
    """
    将已拼好的 SSE 字符串按事件分块下发，避免整段 body 一次写出导致客户端/插件缓冲、界面不刷新。
    """
    from starlette.responses import StreamingResponse

    async def gen() -> Any:
        s = (sse_body or "").replace("\r\n", "\n")
        parts = [p.strip() for p in s.split("\n\n") if p.strip()]
        for p in parts:
            yield (p + "\n\n").encode("utf-8")

    return StreamingResponse(
        gen(),
        status_code=200,
        media_type="text/event-stream; charset=utf-8",
        headers=_sse_streaming_headers(),
    )


async def _httpx_streaming_body_close(resp: Any) -> None:
    try:
        await resp.aclose()
    except Exception:
        pass


async def _starlette_stream_from_httpx(resp: Any) -> Any:
    """Passthrough raw bytes from an httpx streaming response (caller checked status < 400)."""
    from starlette.responses import StreamingResponse

    media_type = str(resp.headers.get("content-type") or "text/event-stream; charset=utf-8")

    async def gen():
        try:
            async for chunk in resp.aiter_bytes():
                yield chunk
        finally:
            await _httpx_streaming_body_close(resp)

    return StreamingResponse(
        gen(),
        status_code=resp.status_code,
        media_type=media_type,
        headers=_sse_streaming_headers(),
    )


def _merge_choice_message_snapshot_into_state(state: dict[str, Any], ch0: dict[str, Any]) -> None:
    """部分上游在最后一个 chunk 只填 choices[0].message、delta 为空或 null，若不合并会丢全文。"""
    msg = ch0.get("message")
    if not isinstance(msg, dict):
        return
    if msg.get("content") is not None:
        piece = _extract_text_from_content(msg.get("content"))
        if piece.strip():
            cur = str(state.get("content") or "").strip()
            if not cur:
                state["content"] = piece
            elif len(piece) > len(str(state.get("content") or "")):
                state["content"] = piece
    for key in ("reasoning_content", "reasoning"):
        if msg.get(key) is not None:
            state[key] = str(state.get(key) or "") + str(msg.get(key))
    tcs = msg.get("tool_calls")
    if isinstance(tcs, list) and tcs:
        if "tool_calls" not in state:
            state["tool_calls"] = {}
        tc_map: dict[int, dict[str, Any]] = state["tool_calls"]
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            idx = int(tc.get("index") or 0)
            bucket = tc_map.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                bucket["id"] = str(tc.get("id"))
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name"):
                    bucket["name"] = str(fn.get("name"))
                if fn.get("arguments") is not None:
                    bucket["arguments"] = str(bucket.get("arguments") or "") + str(fn.get("arguments"))


def _accumulate_chat_completion_chunk(state: dict[str, Any], data_obj: dict[str, Any]) -> None:
    choices = data_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    ch0 = choices[0] if isinstance(choices[0], dict) else {}
    delta = ch0.get("delta")
    if not isinstance(delta, dict):
        delta = {}
    if delta.get("role"):
        state["role"] = delta.get("role")
    if delta.get("content") is not None:
        state["content"] = str(state.get("content") or "") + str(delta.get("content"))
    if delta.get("reasoning_content") is not None:
        state["reasoning_content"] = str(state.get("reasoning_content") or "") + str(
            delta.get("reasoning_content")
        )
    if delta.get("reasoning") is not None:
        state["reasoning"] = str(state.get("reasoning") or "") + str(delta.get("reasoning"))
    tcs = delta.get("tool_calls")
    if isinstance(tcs, list):
        if "tool_calls" not in state:
            state["tool_calls"] = {}
        tc_map: dict[int, dict[str, Any]] = state["tool_calls"]
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            idx = int(tc.get("index") or 0)
            bucket = tc_map.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.get("id"):
                bucket["id"] = str(tc.get("id"))
            fn = tc.get("function")
            if isinstance(fn, dict):
                if fn.get("name"):
                    bucket["name"] = str(fn.get("name"))
                if fn.get("arguments") is not None:
                    bucket["arguments"] = str(bucket.get("arguments") or "") + str(fn.get("arguments"))
    _merge_choice_message_snapshot_into_state(state, ch0)


def _accumulated_stream_state_to_message(state: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": state.get("content") or ""}
    r1 = state.get("reasoning_content") or ""
    r2 = state.get("reasoning") or ""
    combined = (r1 + r2) if (str(r1).strip() or str(r2).strip()) else ""
    if combined.strip():
        msg["reasoning_content"] = combined
    tcalls = state.get("tool_calls")
    if isinstance(tcalls, dict) and tcalls:
        out_tc: list[dict[str, Any]] = []
        for idx in sorted(tcalls.keys()):
            b = tcalls[idx]
            tid = str(b.get("id") or f"call_{uuid.uuid4().hex[:8]}")
            name = str(b.get("name") or "")
            args = str(b.get("arguments") or "{}")
            if not name:
                continue
            out_tc.append(
                {"id": tid, "type": "function", "function": {"name": name, "arguments": args}}
            )
        if out_tc:
            msg["tool_calls"] = out_tc
    return msg


async def _translate_chat_sse_to_response_sse(
    resp: Any,
    model: str,
    parallel_tool_calls: bool,
) -> Any:
    """
    Convert upstream chat.completions SSE into Responses-style SSE (Codex /v1/responses).
    发出与 OpenAI Responses 流式事件对齐的 reasoning / output_text delta，便于 TUI 渲染思考过程。
    """
    from starlette.responses import StreamingResponse

    response_id = f"resp_{uuid.uuid4().hex[:16]}"
    message_item_id = f"msg_{uuid.uuid4().hex[:10]}"

    async def gen():
        state: dict[str, Any] = {}
        seq = 0
        stream_read_err: Exception | None = None
        had_output_text_delta: bool = False
        # Codex CLI 对海量逐 token SSE 帧易在 response.completed 前超时/断开；默认不内联发 delta，只在收尾发整段。
        skip_inline_deltas = str(os.getenv("CODEX_BRIDGE_SKIP_COMPAT_STREAM_DELTAS", "1")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        try:
            yield _sse_data_line_bytes(
                {
                    "type": "response.created",
                    "sequence_number": seq,
                    "response": {
                        "id": response_id,
                        "object": "response",
                        "created_at": int(time.time()),
                        "model": model,
                        "status": "in_progress",
                    },
                }
            )
            seq += 1
            # Codex 在收到 delta 前需要先有 output_item.added，否则会在 response.completed 前断开连接。
            yield _sse_data_line_bytes(
                {
                    "type": "response.output_item.added",
                    "sequence_number": seq,
                    "response_id": response_id,
                    "output_index": 0,
                    "item": {
                        "type": "message",
                        "id": message_item_id,
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [],
                    },
                }
            )
            seq += 1
            try:
                async for line in resp.aiter_lines():
                    ls = (line or "").strip()
                    if ls.startswith("data:"):
                        raw = ls[len("data:") :].strip()
                    elif ls.startswith("{"):
                        # 少数上游直接每行一个 JSON，无 data: 前缀
                        raw = ls
                    else:
                        continue
                    if raw == "[DONE]":
                        continue
                    try:
                        data_obj = json.loads(raw)
                    except Exception:
                        continue
                    if not isinstance(data_obj, dict):
                        continue
                    _accumulate_chat_completion_chunk(state, data_obj)
                    choices = data_obj.get("choices") or []
                    ch0 = choices[0] if choices and isinstance(choices[0], dict) else {}
                    delta = ch0.get("delta")
                    if not isinstance(delta, dict):
                        delta = {}
                    rd = delta.get("reasoning_content")
                    if rd is None:
                        rd = delta.get("reasoning")
                    # 只发 output_text.delta（与上方 output_item.added 的 message 同 output_index），
                    # 不再单独发 reasoning_text.delta，避免与单 item 流式语义冲突导致 Codex 断开。
                    if not skip_inline_deltas:
                        if rd is not None and delta.get("content") is None:
                            yield _sse_data_line_bytes(
                                {
                                    "type": "response.output_text.delta",
                                    "response_id": response_id,
                                    "delta": str(rd),
                                    "item_id": message_item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "sequence_number": seq,
                                }
                            )
                            seq += 1
                            had_output_text_delta = True
                        if delta.get("content") is not None:
                            yield _sse_data_line_bytes(
                                {
                                    "type": "response.output_text.delta",
                                    "response_id": response_id,
                                    "delta": str(delta.get("content")),
                                    "item_id": message_item_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "sequence_number": seq,
                                }
                            )
                            seq += 1
                            had_output_text_delta = True
            except Exception as exc:
                stream_read_err = exc
                _log(f"translate_sse upstream stream read error: {_short_text(exc)}")

            msg = _accumulated_stream_state_to_message(state)
            if stream_read_err:
                c0 = str(msg.get("content") or "").strip()
                r0 = str(msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
                if not c0 and not r0:
                    msg["content"] = f"[bridge] upstream stream aborted: {stream_read_err!s}"
            r_full = str(state.get("reasoning_content") or "") + str(state.get("reasoning") or "")
            ct = str(state.get("content") or "").strip()
            out_done_text = ct if ct else r_full.strip()
            if out_done_text:
                # 上游只在最后一包带 message、无 delta 时，循环内可能从未发过 delta；Codex 要求先有 delta 再有 done。
                if not had_output_text_delta:
                    yield _sse_data_line_bytes(
                        {
                            "type": "response.output_text.delta",
                            "response_id": response_id,
                            "delta": out_done_text,
                            "item_id": message_item_id,
                            "output_index": 0,
                            "content_index": 0,
                            "sequence_number": seq,
                        }
                    )
                    seq += 1
                yield _sse_data_line_bytes(
                    {
                        "type": "response.output_text.done",
                        "response_id": response_id,
                        "item_id": message_item_id,
                        "output_index": 0,
                        "content_index": 0,
                        "sequence_number": seq,
                        "text": out_done_text,
                    }
                )
                seq += 1
            fake_chat = {
                "id": f"chatcmpl_{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            }
            conv = _chat_completion_to_responses_payload(
                fake_chat, model, parallel_tool_calls=parallel_tool_calls
            )
            conv["id"] = response_id
            _truncate_resp_completed_reasoning_for_sse(conv)
            msg_out: dict[str, Any] | None = None
            for it in conv.get("output") or []:
                if isinstance(it, dict) and str(it.get("type") or "").strip().lower() == "message":
                    msg_out = it
            if msg_out:
                yield _sse_data_line_bytes(
                    {
                        "type": "response.output_item.done",
                        "sequence_number": seq,
                        "response_id": response_id,
                        "output_index": 0,
                        "item": msg_out,
                    }
                )
                seq += 1
            yield _sse_data_line_bytes(
                {
                    "type": "response.completed",
                    "sequence_number": seq,
                    "response": conv,
                }
            )
            yield b"data: [DONE]\n\n"
        finally:
            await _httpx_streaming_body_close(resp)

    return StreamingResponse(
        gen(),
        status_code=200,
        media_type="text/event-stream; charset=utf-8",
        headers=_sse_streaming_headers(),
    )


def _normalize_chat_role(role: Any) -> str:
    r = str(role or "user").strip().lower()
    if r == "developer":
        return "system"
    if r in ("system", "user", "assistant", "tool"):
        return r
    return "user"


def _responses_input_to_chat_messages(input_value: Any) -> list[dict[str, Any]]:
    if not isinstance(input_value, list):
        return [{"role": "user", "content": "Hello"}]

    messages: list[dict[str, Any]] = []
    for item in input_value:
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").strip().lower()

        if item_type == "function_call":
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            call_id = str(item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:8]}")
            arguments = item.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif arguments is None:
                arguments = "{}"
            else:
                arguments = str(arguments)
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            )
            continue

        if item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = str(item.get("call_id") or "")
            output_text = _responses_output_to_text(item.get("output"))
            tool_msg: dict[str, Any] = {"role": "tool", "content": output_text}
            if call_id:
                tool_msg["tool_call_id"] = call_id
            messages.append(tool_msg)
            continue

        if item_type in ("message", "") or "role" in item:
            role = _normalize_chat_role(item.get("role"))
            content_text = _extract_text_from_content(item.get("content"))
            msg: dict[str, Any] = {"role": role, "content": content_text}
            if role == "tool" and item.get("call_id"):
                msg["tool_call_id"] = str(item.get("call_id"))
            messages.append(msg)
            continue

    if not messages:
        messages.append({"role": "user", "content": "Hello"})

    merged = _coalesce_leading_system_messages(messages)
    return [m for m in merged if isinstance(m, dict)]


def _responses_tools_to_chat_tools(tools_value: Any) -> list[dict[str, Any]]:
    if not isinstance(tools_value, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools_value:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()
        if tool_type != "function":
            continue

        if isinstance(tool.get("function"), dict):
            fn = dict(tool.get("function") or {})
            name = str(fn.get("name") or "").strip()
            if not name:
                continue
            converted = {"type": "function", "function": {"name": name}}
            if "description" in fn and fn.get("description") is not None:
                converted["function"]["description"] = fn.get("description")
            if "parameters" in fn and fn.get("parameters") is not None:
                converted["function"]["parameters"] = fn.get("parameters")
            if "strict" in fn and fn.get("strict") is not None:
                converted["function"]["strict"] = fn.get("strict")
            out.append(converted)
            continue

        name = str(tool.get("name") or "").strip()
        if not name:
            continue
        converted = {"type": "function", "function": {"name": name}}
        if "description" in tool and tool.get("description") is not None:
            converted["function"]["description"] = tool.get("description")
        if "parameters" in tool and tool.get("parameters") is not None:
            converted["function"]["parameters"] = tool.get("parameters")
        if "strict" in tool and tool.get("strict") is not None:
            converted["function"]["strict"] = tool.get("strict")
        out.append(converted)
    return out


def _normalize_tools_for_vllm_responses(tools_value: Any) -> list[dict[str, Any]]:
    if not isinstance(tools_value, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools_value:
        if not isinstance(tool, dict):
            continue
        tool_type = str(tool.get("type") or "").strip().lower()

        if tool_type == "function":
            cleaned = _strip_strict_fields(tool)
            if isinstance(cleaned, dict):
                out.append(cleaned)
            continue

        if tool_type == "custom":
            name = str(tool.get("name") or "").strip()
            if not name and isinstance(tool.get("custom"), dict):
                name = str((tool.get("custom") or {}).get("name") or "").strip()
            if not name:
                continue
            converted: dict[str, Any] = {"type": "function", "name": name}
            description = tool.get("description")
            if description is not None:
                converted["description"] = description
            params = tool.get("input_schema")
            if params is None:
                params = tool.get("parameters")
            if params is not None:
                converted["parameters"] = params
            out.append(_strip_strict_fields(converted))
            continue
    return out


def _responses_payload_to_chat_payload(payload: dict[str, Any], model: str, app_state: Any = None, thinking_mode: bool = None) -> dict[str, Any]:
    chat_payload: dict[str, Any] = {
        "model": model,
        "messages": _responses_input_to_chat_messages(payload.get("input")),
        "stream": False,
    }
    tools = _responses_tools_to_chat_tools(payload.get("tools"))
    if tools:
        chat_payload["tools"] = tools

    if "tool_choice" in payload and payload.get("tool_choice") is not None:
        chat_payload["tool_choice"] = payload.get("tool_choice")

    def _cfg(name: str) -> Any:
        if app_state is None:
            return None
        return getattr(app_state, name, None)

    if isinstance(payload.get("temperature"), (int, float)):
        chat_payload["temperature"] = payload.get("temperature")
    else:
        t = _cfg("config_temperature")
        if isinstance(t, (int, float)):
            chat_payload["temperature"] = t

    if isinstance(payload.get("top_p"), (int, float)):
        chat_payload["top_p"] = payload.get("top_p")
    else:
        tp = _cfg("config_top_p")
        if isinstance(tp, (int, float)):
            chat_payload["top_p"] = tp

    if isinstance(payload.get("presence_penalty"), (int, float)):
        chat_payload["presence_penalty"] = payload.get("presence_penalty")
    else:
        pp = _cfg("config_presence_penalty")
        if isinstance(pp, (int, float)):
            chat_payload["presence_penalty"] = pp

    top_k = payload.get("top_k")
    if isinstance(top_k, (int, float)) and top_k > 0:
        chat_payload["top_k"] = int(top_k)
    else:
        tk = _cfg("config_top_k")
        if isinstance(tk, (int, float)) and tk > 0:
            chat_payload["top_k"] = int(tk)

    max_output_tokens = payload.get("max_output_tokens")
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        chat_payload["max_tokens"] = max_output_tokens

    # thinking：显式请求 > 调用参数 > llm_config（Codex 可能不带 enable_thinking，需靠桥接默认）
    effective_thinking = thinking_mode
    if effective_thinking is None and app_state is not None:
        effective_thinking = _coerce_bool(getattr(app_state, "config_thinking_mode", None))
    pe = payload.get("enable_thinking")
    if pe is True:
        effective_thinking = True
    elif pe is False:
        effective_thinking = False

    if effective_thinking:
        chat_payload["enable_thinking"] = True
        reff = payload.get("reasoning_effort")
        if isinstance(reff, str) and reff.strip():
            chat_payload["reasoning_effort"] = reff.strip()
        else:
            chat_payload["reasoning_effort"] = "high"
        _log(
            f"thinking_mode enabled: enable_thinking=True, reasoning_effort={chat_payload.get('reasoning_effort')}"
        )

    return chat_payload


def _build_reasoning_item_for_responses(reasoning_text: str) -> dict[str, Any]:
    """
    OpenAI Responses 中 Reasoning 条目。
    - summary: 客户端（Codex TUI 等）实际展示的文本，放完整思考内容。
    - content: 内部推理 token（OpenAI 加密，本地模型直接放原文）。
    """
    return {
        "type": "reasoning",
        "id": f"rs_{uuid.uuid4().hex[:10]}",
        "summary": [{"type": "summary_text", "text": reasoning_text}],
        "content": [{"type": "reasoning_text", "text": reasoning_text}],
        "status": "completed",
    }


def _normalize_reasoning_items_in_response(resp: dict[str, Any]) -> dict[str, Any]:
    """若上游 reasoning 缺少 content/summary，从 text 等字段补全为完整形状。"""
    if not isinstance(resp, dict):
        return resp
    out = resp.get("output")
    if not isinstance(out, list):
        return resp
    changed = False
    new_out: list[Any] = []
    for item in out:
        if not isinstance(item, dict):
            new_out.append(item)
            continue
        if str(item.get("type") or "").strip().lower() != "reasoning":
            new_out.append(item)
            continue
        has_rt = False
        for p in item.get("content") or []:
            if isinstance(p, dict) and str(p.get("type") or "").strip().lower() == "reasoning_text":
                has_rt = True
                break
        has_summary = False
        for p in item.get("summary") or []:
            if isinstance(p, dict) and str(p.get("type") or "").strip().lower() == "summary_text":
                has_summary = True
                break
        if has_rt and has_summary:
            new_out.append(item)
            continue
        txt = item.get("text")
        if isinstance(txt, str) and txt.strip():
            it2 = dict(item)
            if not has_rt:
                it2["content"] = [{"type": "reasoning_text", "text": txt}]
            if not has_summary:
                it2["summary"] = [{"type": "summary_text", "text": txt}]
            it2.setdefault("status", "completed")
            new_out.append(it2)
            changed = True
        elif has_rt and not has_summary:
            rt_text = _reasoning_plain_text_from_item(item)
            if rt_text:
                it2 = dict(item)
                it2["summary"] = [{"type": "summary_text", "text": rt_text}]
                new_out.append(it2)
                changed = True
            else:
                new_out.append(item)
        else:
            new_out.append(item)
    if not changed:
        return resp
    r2 = dict(resp)
    r2["output"] = new_out
    return r2


def _chat_completion_to_responses_payload(
    chat_data: dict[str, Any], model: str, parallel_tool_calls: bool
) -> dict[str, Any]:
    created_at = int(chat_data.get("created") or time.time())
    choices = chat_data.get("choices") or []
    choice = choices[0] if isinstance(choices, list) and choices else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    if not isinstance(message, dict):
        message = {}

    output: list[dict[str, Any]] = []

    reasoning_content = _extract_text_from_content(message.get("reasoning_content"))
    if not reasoning_content:
        reasoning_content = _extract_text_from_content(message.get("reasoning"))

    tool_calls = message.get("tool_calls")
    has_structured_tool_calls = isinstance(tool_calls, list) and len(tool_calls) > 0

    # 有工具调用时 reasoning 作为独立 item（没有 message 正文可嵌入）；
    # 无工具调用时 reasoning 嵌入 message content（规避 Codex TUI 不渲染 reasoning item 的 bug）。
    if reasoning_content and has_structured_tool_calls:
        output.append(_build_reasoning_item_for_responses(reasoning_content))
    if has_structured_tool_calls:
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function")
            if not isinstance(function, dict):
                continue
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            arguments = function.get("arguments")
            if isinstance(arguments, (dict, list)):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif arguments is None:
                arguments = "{}"
            else:
                arguments = str(arguments)
            call_id = str(tc.get("id") or f"call_{uuid.uuid4().hex[:8]}")
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:8]}",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }
            )

    if not has_structured_tool_calls:
        content = _extract_text_from_content(message.get("content"))
        parsed_text_tc = _parse_loose_tool_call_text(content)
        if parsed_text_tc:
            fn_name, args = parsed_text_tc
            output.append(
                {
                    "type": "function_call",
                    "id": f"fc_{uuid.uuid4().hex[:8]}",
                    "call_id": f"call_{uuid.uuid4().hex[:8]}",
                    "name": fn_name,
                    "arguments": json.dumps(args, ensure_ascii=False),
                    "status": "completed",
                }
            )
        else:
            text_body = content if isinstance(content, str) else str(content or "")
            body_st = text_body.strip()
            reason_st = (reasoning_content or "").strip()
            if body_st and reason_st:
                display_text = f"<think>\n{reason_st}\n</think>\n\n{body_st}"
            elif body_st:
                display_text = body_st
            elif reason_st:
                display_text = f"<think>\n{reason_st}\n</think>"
            else:
                display_text = ""
            if display_text:
                output.append(
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:8]}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": display_text}],
                    }
                )

    usage_chat = chat_data.get("usage") if isinstance(chat_data.get("usage"), dict) else {}
    prompt_tokens = int((usage_chat or {}).get("prompt_tokens", 0) or 0)
    completion_tokens = int((usage_chat or {}).get("completion_tokens", 0) or 0)
    total_tokens = int((usage_chat or {}).get("total_tokens", prompt_tokens + completion_tokens) or 0)

    return {
        "id": f"resp_{uuid.uuid4().hex[:16]}",
        "object": "response",
        "created_at": created_at,
        "model": model,
        "output": output,
        "parallel_tool_calls": bool(parallel_tool_calls),
        "status": "completed",
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }


def _extract_output_text(resp: dict[str, Any]) -> str:
    """Extract only the assistant reply text (excluding reasoning) from a Responses payload."""
    if not isinstance(resp, dict):
        return ""
    output = resp.get("output")
    assistant_text = ""
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() == "reasoning":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") in ("output_text", "text"):
                        text = part.get("text") or part.get("output_text")
                        if text:
                            parts.append(str(text))
            elif isinstance(content, str):
                parts.append(content)
        if parts:
            assistant_text = "".join(parts)
    if not assistant_text and "output_text" in resp:
        assistant_text = _extract_text_from_content(resp.get("output_text"))
    return assistant_text


def _extract_reasoning_text(resp: dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""

    def _append_reasoning(parts: list[str], value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if text:
                parts.append(text)
            return
        if isinstance(value, list):
            for item in value:
                _append_reasoning(parts, item)
            return
        if isinstance(value, dict):
            # Common keys seen in reasoning payloads.
            for key in ("text", "summary_text", "content", "summary", "reasoning"):
                if key in value:
                    _append_reasoning(parts, value.get(key))

    parts: list[str] = []
    _append_reasoning(parts, resp.get("reasoning"))
    output = resp.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "").strip().lower() == "reasoning":
                _append_reasoning(parts, item)
                continue
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if not isinstance(c, dict):
                        continue
                    ctype = str(c.get("type") or "").strip().lower()
                    if ctype in ("reasoning", "reasoning_text", "summary_text"):
                        _append_reasoning(parts, c)
    if not parts:
        return ""
    merged = "\n".join([p for p in parts if p]).strip()
    if not merged:
        return ""
    return merged


def _usage_from_responses(resp: dict[str, Any]) -> dict[str, Any]:
    usage = resp.get("usage") or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _usage_from_chat_completion(resp: dict[str, Any]) -> dict[str, int]:
    usage = {}
    if isinstance(resp, dict):
        maybe_usage = resp.get("usage")
        if isinstance(maybe_usage, dict):
            usage = maybe_usage
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_chat_completion(
    model: str, text: str, usage: dict[str, Any], reasoning_content: str = ""
) -> dict[str, Any]:
    created_at = int(time.time())
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if reasoning_content:
        message["reasoning_content"] = reasoning_content
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
    }


def _resolve_upstream_authorization_header(
    incoming_authorization: str | None, fallback_api_key: str
) -> str:
    """
    Prefer the client's Authorization (e.g. Codex auth.json OPENAI_API_KEY) for upstream calls.
    Fallback: Bearer <api_key from llm_config / CLI>. Intended for localhost bridge only.
    """
    if incoming_authorization:
        raw = incoming_authorization.strip()
        if raw.lower().startswith("bearer ") and len(raw) > 7:
            return raw
    tok = (fallback_api_key or "").strip()
    return f"Bearer {tok}" if tok else "Bearer "


def create_app(
    upstream_base_url: str,
    upstream_api_key: str,
    default_model: str,
    timeout_sec: float,
    config_temperature: float = None,
    config_thinking_mode: bool = None,
    config_context_window: int = None,
    config_top_p: float = None,
    config_top_k: int = None,
    config_presence_penalty: float = None,
) -> Any:
    _ensure_bridge_session_logging()
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, Response
    import httpx

    _g = globals()
    for _name, _obj in [("Request", Request), ("JSONResponse", JSONResponse), ("Response", Response),
                         ("HTTPException", HTTPException)]:
        _g[_name] = _obj

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _lifespan(application: FastAPI):
        application.state.upstream_base_v1 = base_v1
        application.state.upstream_responses_url = responses_url
        application.state.upstream_models_url = models_url
        application.state.upstream_chat_completions_url = chat_completions_url
        application.state.default_model = default_model
        application.state.upstream_api_key = upstream_api_key
        # 保存配置参数
        application.state.config_temperature = config_temperature
        application.state.config_thinking_mode = config_thinking_mode
        application.state.config_context_window = config_context_window
        application.state.config_top_p = config_top_p
        application.state.config_top_k = config_top_k
        application.state.config_presence_penalty = config_presence_penalty
        _log(
            f"app.state.config_thinking_mode={config_thinking_mode} "
            f"config_top_p={config_top_p} config_top_k={config_top_k} "
            f"config_presence_penalty={config_presence_penalty}"
        )
        application.state.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_sec, connect=10.0),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
            transport=httpx.AsyncHTTPTransport(http1=True, http2=False),
        )
        yield
        await application.state.client.aclose()

    base_v1, responses_url, models_url, chat_completions_url = _normalize_base_urls(upstream_base_url)

    app = FastAPI(title="Codex Chat->Responses Bridge", version="0.1.0", lifespan=_lifespan)

    _log(
        "fastapi mode ready "
        f"run_sha256={_BRIDGE_RUN_SHA256} log_file={_BRIDGE_LOG_PATH} interaction_jsonl={_BRIDGE_INTERACTION_JSONL_PATH} "
        f"upstream_base_v1={base_v1} responses_url={responses_url} "
        f"chat_completions_url={chat_completions_url} default_model={default_model} "
        f"responses_compat_chat={str(RESPONSES_COMPAT_CHAT_ENABLED).lower()} "
        f"responses_input_as_string={str(RESPONSES_INPUT_AS_STRING_ENABLED).lower()} "
        f"responses_input_string_retry={str(RESPONSES_INPUT_STRING_RETRY_ENABLED).lower()} "
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()} "
        f"tui_sse_compat={str(CODEX_BRIDGE_TUI_SSE_COMPAT).lower()}"
    )

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "run_sha256": _BRIDGE_RUN_SHA256,
            "log_file": str(_BRIDGE_LOG_PATH) if _BRIDGE_LOG_PATH else None,
            "interaction_jsonl": str(_BRIDGE_INTERACTION_JSONL_PATH) if _BRIDGE_INTERACTION_JSONL_PATH else None,
            "upstream_base_v1": app.state.upstream_base_v1,
            "responses_endpoint": app.state.upstream_responses_url,
            "chat_completions_endpoint": app.state.upstream_chat_completions_url,
        }

    @app.get("/v1/models")
    @app.get("/models")
    async def models(request: Request) -> JSONResponse:
        auth = _resolve_upstream_authorization_header(
            request.headers.get("authorization"), app.state.upstream_api_key
        )
        headers = {"Authorization": auth}
        try:
            resp = await app.state.client.get(app.state.upstream_models_url, headers=headers)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"Upstream models request failed: {exc}") from exc
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    @app.post("/v1/responses")
    @app.post("/responses")
    async def responses_passthrough(request: Request) -> Any:
        req_id = uuid.uuid4().hex[:8]
        started_at = time.time()
        try:
            payload = await request.json()
        except Exception as exc:
            _log(f"req={req_id} invalid_json error={_short_text(exc)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON object body is required")

        raw_model = str(payload.get("model") or "").strip()
        model, model_source, _ = _resolve_model_name(raw_model, app.state.default_model)
        if not model:
            _log(f"req={req_id} rejected missing_model")
            raise HTTPException(status_code=400, detail="model is required")

        proxied_payload = dict(payload)
        proxied_payload["model"] = model
        proxied_payload = _fold_responses_instructions_into_input(proxied_payload)
        if "input" in proxied_payload:
            proxied_payload["input"] = _sanitize_responses_input(proxied_payload.get("input"))
        if "tools" in proxied_payload:
            proxied_payload["tools"] = _normalize_tools_for_vllm_responses(
                _strip_strict_fields(proxied_payload.get("tools"))
            )

        input_list_snapshot: list[Any] | None = None
        if isinstance(proxied_payload.get("input"), list):
            input_list_snapshot = copy.deepcopy(proxied_payload["input"])
        forced_string_input = False
        if RESPONSES_INPUT_AS_STRING_ENABLED and input_list_snapshot is not None:
            proxied_payload["input"] = _flatten_responses_input_to_string(input_list_snapshot)
            forced_string_input = True
            _log(f"req={req_id} responses_input mode=string reason=env_CODEX_BRIDGE_RESPONSES_INPUT_AS_STRING")

        _log(
            f"req={req_id} inbound path=/v1/responses model={model} "
            f"model_source={model_source} input_roles={_response_input_role_counts(proxied_payload.get('input'))} "
            f"input_role_seq={_response_input_role_sequence(proxied_payload.get('input'))} "
            f"tools={_tool_schema_summary(proxied_payload.get('tools'))} "
            f"stream={str(proxied_payload.get('stream') is True).lower()} "
            f"tool_call_rounds={_count_tool_call_rounds(proxied_payload.get('input'))}"
        )
        _log_interaction_json(
            req_id,
            "responses.inbound",
            {
                "path": "/v1/responses",
                "model": model,
                "model_source": model_source,
                "stream": proxied_payload.get("stream"),
                "parallel_tool_calls": proxied_payload.get("parallel_tool_calls"),
                "tools": proxied_payload.get("tools"),
                "input": proxied_payload.get("input"),
                "forced_string_input": forced_string_input,
                "input_roles": _response_input_role_counts(proxied_payload.get("input")),
                "input_role_seq": _response_input_role_sequence(proxied_payload.get("input")),
            },
        )

        auth = _resolve_upstream_authorization_header(
            request.headers.get("authorization"), app.state.upstream_api_key
        )
        headers = {
            "Authorization": auth,
            "Content-Type": "application/json",
        }

        tools_count = len(proxied_payload.get("tools") or []) if isinstance(proxied_payload.get("tools"), list) else 0
        is_streaming_request = proxied_payload.get("stream") is True
        tool_call_rounds = _count_tool_call_rounds(proxied_payload.get("input"))

        if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
            _log(
                f"req={req_id} tool_call_loop_breaker rounds={tool_call_rounds} "
                f"max={MAX_TOOL_CALL_ROUNDS}"
            )
            error_resp = {
                "id": f"resp_{uuid.uuid4().hex[:16]}",
                "object": "response",
                "model": model,
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex[:8]}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": (
                                    f"Tool call loop detected ({tool_call_rounds} rounds). "
                                    "Stopping to prevent infinite loop. "
                                    "Please rephrase your request or try a different approach."
                                ),
                            }
                        ],
                    }
                ],
                "status": "completed",
            }
            if is_streaming_request:
                sse_body = _build_minimal_response_sse(error_resp)
                return _sse_pregenerated_text_as_streaming_response(sse_body)
            return JSONResponse(status_code=200, content=error_resp)

        if RESPONSES_COMPAT_CHAT_ENABLED:
            compat_mode = "compat_stream" if is_streaming_request else "compat"
            chat_payload = _responses_payload_to_chat_payload(proxied_payload, model, app.state)
            _log(
                f"req={req_id} responses_to_chat mode={compat_mode} tool_count={tools_count} "
                f"message_count={len(chat_payload.get('messages') or [])}"
            )

            # 客户端要求 stream 时优先走上游 chat.completions 真流式，便于 Codex 界面逐段刷新。
            # 若请求带 tools：SSE transcode 未覆盖 function_call 等帧，Codex 会在 response.completed 前断开，改用非流式 + TUI SSE。
            if is_streaming_request and RESPONSES_COMPAT_CHAT_STREAM_ENABLED and tools_count == 0:
                stream_payload = dict(chat_payload)
                stream_payload["stream"] = True
                _log(
                    f"req={req_id} responses_to_chat mode=compat_stream_try tool_count={tools_count} "
                    f"message_count={len(stream_payload.get('messages') or [])}"
                )
                compat_r: Any = None
                try:
                    creq = app.state.client.build_request(
                        "POST",
                        app.state.upstream_chat_completions_url,
                        json=stream_payload,
                        headers=headers,
                    )
                    compat_r = await app.state.client.send(creq, stream=True)
                except httpx.HTTPError as exc:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(
                        f"req={req_id} compat_stream_upstream_error latency_ms={elapsed_ms} "
                        f"error={_short_text(exc)} fallback=compat_nonstream"
                    )
                else:
                    if compat_r is not None and compat_r.status_code < 400:
                        elapsed_ms = int((time.time() - started_at) * 1000)
                        _log(
                            f"req={req_id} compat_stream_ok latency_ms={elapsed_ms} model={model} "
                            f"mode=responses_sse_transcode"
                        )
                        return await _translate_chat_sse_to_response_sse(
                            compat_r,
                            model,
                            bool(proxied_payload.get("parallel_tool_calls", True)),
                        )
                    if compat_r is not None:
                        err_raw = await compat_r.aread()
                        await _httpx_streaming_body_close(compat_r)
                        try:
                            err_data = json.loads(err_raw.decode("utf-8")) if err_raw else {}
                        except Exception:
                            err_data = {"error": {"message": err_raw.decode("utf-8", errors="ignore")}}
                        compat_err = _extract_upstream_error_message(
                            err_data, err_raw.decode("utf-8", errors="ignore")
                        )
                        _log(
                            f"req={req_id} compat_stream_status={compat_r.status_code} "
                            f"error={_short_text(compat_err)} fallback=compat_nonstream"
                        )
            elif is_streaming_request and RESPONSES_COMPAT_CHAT_STREAM_ENABLED and tools_count > 0:
                _log(
                    f"req={req_id} compat_stream_skip reason=tools_present tool_count={tools_count} "
                    f"fallback=compat_nonstream_tui_sse"
                )

            try:
                compat_resp = await app.state.client.post(
                    app.state.upstream_chat_completions_url, json=chat_payload, headers=headers
                )
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(f"req={req_id} compat_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                raise HTTPException(status_code=502, detail=f"Upstream chat.completions request failed: {exc}") from exc

            elapsed_ms = int((time.time() - started_at) * 1000)
            if compat_resp.status_code < 400:
                try:
                    chat_data = compat_resp.json()
                except Exception:
                    chat_data = {}
                converted = _chat_completion_to_responses_payload(
                    chat_data,
                    model,
                    bool(proxied_payload.get("parallel_tool_calls", True)),
                )
                _log(
                    f"req={req_id} compat_done status=200 latency_ms={elapsed_ms} "
                    f"model={model} output_items={len(converted.get('output') or [])}"
                )
                _log_dialogue_transcript(
                    req_id,
                    "responses.compat_chat",
                    inbound_input=proxied_payload.get("input"),
                    chat_payload=chat_payload,
                    upstream_chat_completion=chat_data,
                    converted_responses=converted,
                    extra={"output_lines": _responses_output_full_lines(converted)},
                )
                if is_streaming_request:
                    sse_body = (
                        _build_codex_tui_compat_stream_sse(converted)
                        if CODEX_BRIDGE_TUI_SSE_COMPAT
                        else _build_minimal_response_sse(converted)
                    )
                    return _sse_pregenerated_text_as_streaming_response(sse_body)
                return JSONResponse(status_code=200, content=converted)

            compat_err = ""
            compat_data: dict[str, Any]
            try:
                compat_data = compat_resp.json()
            except Exception:
                compat_err = compat_resp.text
                compat_data = {"error": {"message": compat_err}}
            compat_err = _extract_upstream_error_message(compat_data, compat_resp.text)
            _log(
                f"req={req_id} compat_status={compat_resp.status_code} latency_ms={elapsed_ms} "
                f"error={_short_text(compat_err)} fallback=responses_passthrough"
            )

        async def _open_responses_stream(body: dict[str, Any]) -> Any:
            req = app.state.client.build_request(
                "POST", app.state.upstream_responses_url, json=body, headers=headers
            )
            return await app.state.client.send(req, stream=True)

        async def _read_stream_error(r: Any) -> tuple[dict[str, Any], bytes]:
            raw = await r.aread()
            await _httpx_streaming_body_close(r)
            try:
                data = json.loads(raw.decode("utf-8")) if raw else {}
            except Exception:
                data = {"error": {"message": raw.decode("utf-8", errors="ignore")}}
            if not isinstance(data, dict):
                data = {"error": {"message": str(data)}}
            return data, raw

        if is_streaming_request:
            r = await _open_responses_stream(proxied_payload)
            err_status = 0
            err_data: dict[str, Any] = {}
            if r.status_code >= 400:
                err_status = r.status_code
                err_data, _raw_err = await _read_stream_error(r)
                err_msg = _extract_upstream_error_message(err_data, "")
                if (
                    not forced_string_input
                    and input_list_snapshot is not None
                    and RESPONSES_INPUT_STRING_RETRY_ENABLED
                    and _error_suggests_string_input_required(err_msg, err_data)
                ):
                    retry_payload = dict(proxied_payload)
                    retry_payload["input"] = _flatten_responses_input_to_string(input_list_snapshot)
                    _log(
                        f"req={req_id} compat_retry reason=responses_input_string stream=true "
                        f"input_chars={len(str(retry_payload.get('input') or ''))}"
                    )
                    r = await _open_responses_stream(retry_payload)
                else:
                    r = None

            if r is None:
                lower_err = str(_extract_upstream_error_message(err_data, "")).lower()
                if "system message must be at the beginning" in lower_err:
                    retry_payload = dict(proxied_payload)
                    retry_payload["input"] = _demote_system_roles(retry_payload.get("input"))
                    _log(
                        f"req={req_id} compat_retry reason=system_role_order stream=true "
                        f"input_role_seq={_response_input_role_sequence(retry_payload.get('input'))}"
                    )
                    r = await _open_responses_stream(retry_payload)
                else:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(
                        f"req={req_id} upstream_status={err_status} "
                        f"latency_ms={elapsed_ms} error={_short_text(_extract_upstream_error_message(err_data, ''))}"
                    )
                    return JSONResponse(status_code=err_status, content=err_data)

            if r.status_code >= 400:
                err_data, _raw_err = await _read_stream_error(r)
                err_msg = _extract_upstream_error_message(err_data, "")
                lower_err = err_msg.lower()
                if "system message must be at the beginning" in lower_err:
                    retry_payload = dict(proxied_payload)
                    retry_payload["input"] = _demote_system_roles(retry_payload.get("input"))
                    _log(
                        f"req={req_id} compat_retry reason=system_role_order stream=true "
                        f"input_role_seq={_response_input_role_sequence(retry_payload.get('input'))}"
                    )
                    r = await _open_responses_stream(retry_payload)
                else:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(
                        f"req={req_id} upstream_status={r.status_code} "
                        f"latency_ms={elapsed_ms} error={_short_text(err_msg)}"
                    )
                    return JSONResponse(status_code=r.status_code, content=err_data)

            if r.status_code >= 400:
                err_data, _raw_err = await _read_stream_error(r)
                err_msg = _extract_upstream_error_message(err_data, "")
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(
                    f"req={req_id} upstream_status={r.status_code} "
                    f"latency_ms={elapsed_ms} error={_short_text(err_msg)}"
                )
                return JSONResponse(status_code=r.status_code, content=err_data)

            ct = str(r.headers.get("content-type") or "").lower()
            elapsed_ms = int((time.time() - started_at) * 1000)
            if "text/event-stream" in ct:
                _log(
                    f"req={req_id} done status={r.status_code} stream=passthrough "
                    f"latency_ms={elapsed_ms} model={model}"
                )
                _log_interaction_json(
                    req_id,
                    "responses.stream_passthrough",
                    {
                        "content_type": ct,
                        "note": "SSE bytes streamed to client; body not duplicated here",
                    },
                )
                return await _starlette_stream_from_httpx(r)
            raw_body = await r.aread()
            await _httpx_streaming_body_close(r)
            try:
                data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                data = {}
            if isinstance(data, dict):
                _log(
                    f"req={req_id} output_summary mode=responses_stream_json "
                    f"{_responses_output_summary(data)}"
                )
                _log_interaction_json(
                    req_id,
                    "responses.stream_json_upstream",
                    {
                        "mode": "responses_stream_json",
                        "summary": _responses_output_summary(data, limit=64),
                        "output_items": _responses_output_full_lines(data),
                        "raw_response": data,
                    },
                    max_str=_LOG_DIALOGUE_JSONL_MAX_STR,
                )
                _log_dialogue_transcript(
                    req_id,
                    "responses.stream_json_upstream",
                    inbound_input=proxied_payload.get("input"),
                    extra={
                        "upstream_responses_output_plain": _responses_output_plain_text(data),
                        "upstream_responses_json": data,
                    },
                )
                data, converted_textual_tool = _normalize_response_tool_outputs(data)
                data = _normalize_reasoning_items_in_response(data)
                body_text = _build_minimal_response_sse(data)
                if converted_textual_tool:
                    _log(f"req={req_id} coerced_text_tool_call mode=responses_stream model={model}")
                else:
                    _log(f"req={req_id} wrapped_json_to_sse mode=responses_stream model={model}")
                return _sse_pregenerated_text_as_streaming_response(body_text)
            _log(f"req={req_id} done status={r.status_code} latency_ms={elapsed_ms} model={model}")
            return Response(
                content=raw_body,
                status_code=r.status_code,
                media_type=ct or "application/json",
            )

        try:
            upstream_resp = await app.state.client.post(
                app.state.upstream_responses_url, json=proxied_payload, headers=headers
            )
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(f"req={req_id} upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
            raise HTTPException(status_code=502, detail=f"Upstream responses request failed: {exc}") from exc

        elapsed_ms = int((time.time() - started_at) * 1000)
        if upstream_resp.status_code >= 400:
            err_msg = ""
            try:
                data = upstream_resp.json()
            except Exception:
                err_msg = upstream_resp.text
                data = {"error": {"message": err_msg}}
            err_msg = _extract_upstream_error_message(data, upstream_resp.text)

            if (
                not forced_string_input
                and input_list_snapshot is not None
                and RESPONSES_INPUT_STRING_RETRY_ENABLED
                and _error_suggests_string_input_required(err_msg, data)
            ):
                retry_payload = dict(proxied_payload)
                retry_payload["input"] = _flatten_responses_input_to_string(input_list_snapshot)
                _log(
                    f"req={req_id} compat_retry reason=responses_input_string "
                    f"input_chars={len(str(retry_payload.get('input') or ''))}"
                )
                try:
                    retry_resp = await app.state.client.post(
                        app.state.upstream_responses_url, json=retry_payload, headers=headers
                    )
                    upstream_resp = retry_resp
                    elapsed_ms = int((time.time() - started_at) * 1000)
                except httpx.HTTPError as exc:
                    _log(
                        f"req={req_id} compat_retry_error latency_ms={elapsed_ms} "
                        f"error={_short_text(exc)}"
                    )

            if upstream_resp.status_code >= 400:
                try:
                    data = upstream_resp.json()
                except Exception:
                    err_msg = upstream_resp.text
                    data = {"error": {"message": err_msg}}
                err_msg = _extract_upstream_error_message(data, upstream_resp.text)
                lower_err = err_msg.lower()
                if "system message must be at the beginning" in lower_err:
                    retry_payload = dict(proxied_payload)
                    retry_payload["input"] = _demote_system_roles(retry_payload.get("input"))
                    _log(
                        f"req={req_id} compat_retry reason=system_role_order "
                        f"input_role_seq={_response_input_role_sequence(retry_payload.get('input'))}"
                    )
                    try:
                        retry_resp = await app.state.client.post(
                            app.state.upstream_responses_url, json=retry_payload, headers=headers
                        )
                        upstream_resp = retry_resp
                        elapsed_ms = int((time.time() - started_at) * 1000)
                    except httpx.HTTPError as exc:
                        _log(
                            f"req={req_id} compat_retry_error latency_ms={elapsed_ms} "
                            f"error={_short_text(exc)}"
                        )

            if upstream_resp.status_code >= 400:
                err_msg = ""
                try:
                    data = upstream_resp.json()
                except Exception:
                    err_msg = upstream_resp.text
                    data = {"error": {"message": err_msg}}
                err_msg = _extract_upstream_error_message(data, upstream_resp.text)
                _log(
                    f"req={req_id} upstream_status={upstream_resp.status_code} "
                    f"latency_ms={elapsed_ms} error={_short_text(err_msg)}"
                )
                _log_interaction_json(
                    req_id,
                    "responses.upstream_error",
                    {"status": upstream_resp.status_code, "body": data},
                )
                return JSONResponse(status_code=upstream_resp.status_code, content=data)

        try:
            data = upstream_resp.json()
        except Exception:
            data = {}
        if (
            isinstance(data, dict)
            and tools_count > 0
        ):
            _log(
                f"req={req_id} output_summary mode=responses "
                f"{_responses_output_summary(data)}"
            )
            data, converted_textual_tool = _normalize_response_tool_outputs(data)
            if converted_textual_tool:
                _log(
                    f"req={req_id} coerced_text_tool_call mode=responses "
                    f"model={model}"
                )
        if isinstance(data, dict):
            _log_interaction_json(
                req_id,
                "responses.upstream_json",
                {
                    "status": upstream_resp.status_code,
                    "latency_ms": elapsed_ms,
                    "summary": _responses_output_summary(data, limit=64),
                    "output_items": _responses_output_full_lines(data),
                    "raw_response": data,
                },
                max_str=_LOG_DIALOGUE_JSONL_MAX_STR,
            )
            _log_dialogue_transcript(
                req_id,
                "responses.passthrough_upstream",
                inbound_input=proxied_payload.get("input"),
                extra={
                    "proxied_request_full": proxied_payload,
                    "upstream_responses_output_plain": _responses_output_plain_text(data),
                    "upstream_responses_json": data,
                },
            )
        _log(
            f"req={req_id} done status={upstream_resp.status_code} latency_ms={elapsed_ms} "
            f"model={model}"
        )
        return JSONResponse(status_code=upstream_resp.status_code, content=data)

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(request: Request) -> Any:
        req_id = uuid.uuid4().hex[:8]
        started_at = time.time()
        try:
            payload = await request.json()
        except Exception as exc:
            _log(f"req={req_id} invalid_json error={_short_text(exc)}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc}") from exc

        raw_model = str(payload.get("model") or "").strip()
        model, model_source, _ = _resolve_model_name(raw_model, app.state.default_model)
        if not model:
            _log(f"req={req_id} rejected missing_model")
            raise HTTPException(status_code=400, detail="model is required")

        messages = payload.get("messages") or []
        stream_requested = payload.get("stream") is True

        auth = _resolve_upstream_authorization_header(
            request.headers.get("authorization"), app.state.upstream_api_key
        )
        headers = {
            "Authorization": auth,
            "Content-Type": "application/json",
        }

        if stream_requested:
            up_payload = dict(payload)
            up_payload["model"] = model
            up_payload["stream"] = True
            _log_interaction_json(
                req_id,
                "chat_completions.inbound_stream",
                {"stream": True, "model": model, "messages": messages, "upstream_payload": up_payload},
            )
            _log(
                f"req={req_id} inbound path=/v1/chat/completions model={model} "
                f"model_source={model_source} message_count={len(messages)} stream=true"
            )
            try:
                sreq = app.state.client.build_request(
                    "POST", app.state.upstream_chat_completions_url, json=up_payload, headers=headers
                )
                sresp = await app.state.client.send(sreq, stream=True)
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(f"req={req_id} chat_stream_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                raise HTTPException(
                    status_code=502, detail=f"Upstream chat.completions stream failed: {exc}"
                ) from exc
            if sresp.status_code >= 400:
                err_raw = await sresp.aread()
                await _httpx_streaming_body_close(sresp)
                try:
                    err_json = json.loads(err_raw.decode("utf-8")) if err_raw else {}
                except Exception:
                    err_json = {"error": {"message": err_raw.decode("utf-8", errors="ignore")}}
                if not isinstance(err_json, dict):
                    err_json = {"error": {"message": str(err_json)}}
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(
                    f"req={req_id} chat_stream_status={sresp.status_code} latency_ms={elapsed_ms} "
                    f"error={_short_text(_extract_upstream_error_message(err_json, ''))}"
                )
                return JSONResponse(status_code=sresp.status_code, content=err_json)
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(f"req={req_id} chat_stream_ok status=200 latency_ms={elapsed_ms} model={model}")
            _log_interaction_json(
                req_id,
                "chat_completions.stream_proxy",
                {"note": "SSE streamed from upstream; body not duplicated"},
            )
            return await _starlette_stream_from_httpx(sresp)

        _log(
            f"req={req_id} inbound path=/v1/chat/completions model={model} "
            f"model_source={model_source} message_count={len(messages)} roles={_role_counts(messages)}"
        )
        input_items = _chat_messages_to_responses_input(messages)
        _log_interaction_json(
            req_id,
            "chat_completions.inbound",
            {
                "stream": False,
                "model": model,
                "messages": messages,
                "responses_payload_preview": {"model": model, "input": input_items},
            },
        )

        responses_payload: dict[str, Any] = {
            "model": model,
            "input": input_items,
        }

        max_tokens = payload.get("max_tokens")
        if isinstance(max_tokens, int) and max_tokens > 0:
            responses_payload["max_output_tokens"] = max_tokens
        # 使用 payload 中的 temperature，如果没有则使用配置文件中的默认值
        temperature = payload.get("temperature")
        if temperature is None and hasattr(app.state, "config_temperature") and app.state.config_temperature is not None:
            temperature = app.state.config_temperature
        if isinstance(temperature, (int, float)):
            responses_payload["temperature"] = temperature
        
        # thinking_mode: 优先使用客户端请求中的 enable_thinking，其次使用配置文件
        enable_thinking = payload.get("enable_thinking")
        if enable_thinking is None and hasattr(app.state, "config_thinking_mode") and app.state.config_thinking_mode:
            enable_thinking = True
        
        if enable_thinking is not None:
            responses_payload["enable_thinking"] = enable_thinking
            # 也保留 reasoning_effort 以兼容其他实现
            if enable_thinking:
                reasoning_effort = payload.get("reasoning_effort", "high")
                responses_payload["reasoning_effort"] = reasoning_effort
                _log(f"req={req_id} thinking_mode enabled: enable_thinking=True, reasoning_effort={reasoning_effort}")
        
        # context_window: 如果有设置，添加到 metadata 或作为额外参数
        if hasattr(app.state, "config_context_window") and app.state.config_context_window:
            responses_payload["context_window"] = app.state.config_context_window
        top_p = payload.get("top_p")
        if isinstance(top_p, (int, float)):
            responses_payload["top_p"] = top_p

        # top_k: vLLM 扩展参数，需要通过 extra_body 传递
        top_k = payload.get("top_k")
        if isinstance(top_k, (int, float)) and top_k > 0:
            # vLLM 要求 top_k 放在 extra_body 中
            if "extra_body" not in responses_payload:
                responses_payload["extra_body"] = {}
            responses_payload["extra_body"]["top_k"] = int(top_k)

        # presence_penalty: OpenAI 标准参数
        presence_penalty = payload.get("presence_penalty")
        if isinstance(presence_penalty, (int, float)):
            responses_payload["presence_penalty"] = presence_penalty

        try:
            upstream_resp = await app.state.client.post(
                app.state.upstream_responses_url, json=responses_payload, headers=headers
            )
        except httpx.HTTPError as exc:
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(f"req={req_id} upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
            raise HTTPException(status_code=502, detail=f"Upstream responses request failed: {exc}") from exc

        if upstream_resp.status_code in (404, 405):
            _log(
                f"req={req_id} responses_unavailable status={upstream_resp.status_code} "
                f"fallback=chat_completions"
            )
            fallback_payload = dict(payload)
            fallback_payload["model"] = model
            fallback_payload["stream"] = False
            if not isinstance(fallback_payload.get("messages"), list):
                fallback_payload["messages"] = messages if isinstance(messages, list) else []

            try:
                fallback_resp = await app.state.client.post(
                    app.state.upstream_chat_completions_url, json=fallback_payload, headers=headers
                )
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(f"req={req_id} fallback_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                raise HTTPException(
                    status_code=502, detail=f"Upstream chat.completions request failed: {exc}"
                ) from exc

            if fallback_resp.status_code >= 400:
                try:
                    data = fallback_resp.json()
                except Exception:
                    data = {"error": {"message": fallback_resp.text}}
                elapsed_ms = int((time.time() - started_at) * 1000)
                err_msg = ""
                if isinstance(data, dict):
                    maybe_err = data.get("error")
                    if isinstance(maybe_err, dict):
                        err_msg = str(maybe_err.get("message") or "")
                if not err_msg:
                    err_msg = fallback_resp.text
                _log(
                    f"req={req_id} fallback_status={fallback_resp.status_code} "
                    f"latency_ms={elapsed_ms} error={_short_text(err_msg)}"
                )
                return JSONResponse(status_code=fallback_resp.status_code, content=data)

            fallback_data = fallback_resp.json()
            fallback_usage = _usage_from_chat_completion(fallback_data)
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(
                f"req={req_id} done status=200 mode=chat_completions_fallback latency_ms={elapsed_ms} "
                f"model={model} prompt_tokens={fallback_usage.get('prompt_tokens', 0)} "
                f"completion_tokens={fallback_usage.get('completion_tokens', 0)} "
                f"total_tokens={fallback_usage.get('total_tokens', 0)}"
            )
            _log_dialogue_transcript(
                req_id,
                "chat_completions.fallback_direct",
                inbound_input=messages,
                chat_payload=fallback_payload,
                upstream_chat_completion=fallback_data,
            )
            return JSONResponse(status_code=200, content=fallback_data)

        if upstream_resp.status_code >= 400:
            try:
                data = upstream_resp.json()
            except Exception:
                data = {"error": {"message": upstream_resp.text}}
            elapsed_ms = int((time.time() - started_at) * 1000)
            err_msg = ""
            if isinstance(data, dict):
                maybe_err = data.get("error")
                if isinstance(maybe_err, dict):
                    err_msg = str(maybe_err.get("message") or "")
            if not err_msg:
                err_msg = upstream_resp.text
            _log(
                f"req={req_id} upstream_status={upstream_resp.status_code} "
                f"latency_ms={elapsed_ms} error={_short_text(err_msg)}"
            )
            return JSONResponse(status_code=upstream_resp.status_code, content=data)

        resp_data = upstream_resp.json()
        assistant_text = _extract_output_text(resp_data)
        reasoning_text = _extract_reasoning_text(resp_data)
        usage = _usage_from_responses(resp_data)
        result = _build_chat_completion(model, assistant_text, usage, reasoning_content=reasoning_text)
        elapsed_ms = int((time.time() - started_at) * 1000)
        _log(
            f"req={req_id} done status=200 latency_ms={elapsed_ms} model={model} "
            f"prompt_tokens={usage.get('prompt_tokens', 0)} completion_tokens={usage.get('completion_tokens', 0)} "
            f"total_tokens={usage.get('total_tokens', 0)} output_chars={len(assistant_text)} "
            f"reasoning_chars={len(reasoning_text)}"
        )
        _log_interaction_json(
            req_id,
            "chat_completions.done",
            {
                "assistant_text": assistant_text,
                "usage": usage,
                "upstream_responses_json": resp_data,
            },
            max_str=_LOG_DIALOGUE_JSONL_MAX_STR,
        )
        _log_dialogue_transcript(
            req_id,
            "chat_completions.from_upstream_responses",
            inbound_input=responses_payload.get("input"),
            extra={
                "responses_payload": responses_payload,
                "upstream_responses_json": resp_data,
                "assistant_text_merged": assistant_text,
                "openai_chat_completion_result": result,
            },
        )
        return JSONResponse(status_code=200, content=result)

    return app


def _resolve_codex_home() -> Path:
    override = os.environ.get("CODEX_HOME", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".codex"


def _load_codex_cli_config() -> tuple[str, str, str]:
    codex_home = _resolve_codex_home()
    config_path = codex_home / "config.toml"
    auth_path = codex_home / "auth.json"

    if not config_path.exists():
        raise RuntimeError(f"Codex config.toml not found: {config_path}")
    if not auth_path.exists():
        raise RuntimeError(f"Codex auth.json not found: {auth_path}")

    try:
        import tomllib  # Python 3.11+
    except Exception as exc:
        raise RuntimeError(f"tomllib not available: {exc}") from exc

    cfg = tomllib.loads(config_path.read_text(encoding="utf-8"))
    provider_key = str(cfg.get("model_provider") or "").strip()
    model = str(cfg.get("model") or "").strip()
    providers = cfg.get("model_providers") or {}
    base_url = ""
    if provider_key and isinstance(providers, dict):
        provider_cfg = providers.get(provider_key) or {}
        if isinstance(provider_cfg, dict):
            base_url = str(provider_cfg.get("base_url") or "").strip()

    auth = json.loads(auth_path.read_text(encoding="utf-8"))
    api_key = str(auth.get("OPENAI_API_KEY") or "").strip()

    if not base_url:
        raise RuntimeError("Codex base_url not found in config.toml")
    if not model:
        raise RuntimeError("Codex model not found in config.toml")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in auth.json")

    return base_url, api_key, model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Codex Chat->Responses bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--upstream-base-url", default=None)
    parser.add_argument("--upstream-api-key", default=None)
    parser.add_argument("--default-model", default=None)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--use-codex-config", action="store_true", help="Load upstream settings from ~/.codex")
    return parser.parse_args()


def _coerce_bool(value: Any) -> bool | None:
    """Parse llm_config / env style booleans."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off", ""):
        return False
    return None


def _ensure_bridge_config(args: argparse.Namespace) -> None:
    """Fill upstream_base_url / api_key / default_model from llm_config.json or ~/.codex."""
    plugin_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_config.json")
    plugin_cfg: dict[str, Any] | None = None
    if os.path.exists(plugin_config_path):
        try:
            with open(plugin_config_path, "r", encoding="utf-8") as f:
                plugin_cfg = json.load(f)
        except Exception as e:
            _log(f"failed to read plugin llm_config.json: {e}")
            plugin_cfg = None

    if plugin_cfg:
        if not getattr(args, "upstream_base_url", None):
            args.upstream_base_url = plugin_cfg.get("api_base_url")
        if not getattr(args, "upstream_api_key", None):
            args.upstream_api_key = plugin_cfg.get("api_key", "dummy")
        if not getattr(args, "default_model", None):
            args.default_model = plugin_cfg.get("model_name")
        # 与上游无关的桥接参数：即使 CLI 已指定上游 URL，也要从 llm_config 合并（否则 thinking_mode 等会丢失）
        if getattr(args, "config_temperature", None) is None:
            args.config_temperature = plugin_cfg.get("temperature")
        if getattr(args, "config_thinking_mode", None) is None:
            args.config_thinking_mode = _coerce_bool(plugin_cfg.get("thinking_mode"))
        if getattr(args, "config_context_window", None) is None:
            args.config_context_window = plugin_cfg.get("context_window")
        if getattr(args, "config_top_p", None) is None:
            args.config_top_p = plugin_cfg.get("top_p")
        if getattr(args, "config_top_k", None) is None:
            args.config_top_k = plugin_cfg.get("top_k")
        if getattr(args, "config_presence_penalty", None) is None:
            args.config_presence_penalty = plugin_cfg.get("presence_penalty")
        _log(f"merged config from llm_config.json: {plugin_config_path}")

    if args.use_codex_config:
        base_url, api_key, model = _load_codex_cli_config()
        if not args.upstream_base_url:
            args.upstream_base_url = base_url
        if not args.upstream_api_key:
            args.upstream_api_key = api_key
        if not args.default_model:
            args.default_model = model


def create_asgi_app() -> Any:
    """
    ASGI factory for: uvicorn chat_bridge:create_asgi_app --factory --host 127.0.0.1 --port 18081
    (run with cwd=repository root: set PYTHONPATH to include the `bridge` package if needed).
    """
    args = argparse.Namespace(
        host="127.0.0.1",
        port=18081,
        upstream_base_url=None,
        upstream_api_key=None,
        default_model=None,
        timeout_sec=float(os.getenv("CODEX_BRIDGE_TIMEOUT_SEC", "999999")),
        use_codex_config=False,
        config_temperature=None,
        config_thinking_mode=None,
        config_context_window=None,
        config_top_p=None,
        config_top_k=None,
        config_presence_penalty=None,
    )
    _ensure_bridge_config(args)
    if not args.upstream_base_url or not args.upstream_api_key or not args.default_model:
        raise RuntimeError(
            "Missing upstream settings. Add llm_config.json next to the plugin or set CLI via python chat_bridge.py."
        )
    return create_app(
        args.upstream_base_url,
        args.upstream_api_key,
        args.default_model,
        args.timeout_sec,
        getattr(args, "config_temperature", None),
        getattr(args, "config_thinking_mode", None),
        getattr(args, "config_context_window", None),
        getattr(args, "config_top_p", None),
        getattr(args, "config_top_k", None),
        getattr(args, "config_presence_penalty", None),
    )


def _run_simple_server(
    host: str,
    port: int,
    upstream_base_url: str,
    upstream_api_key: str,
    default_model: str,
    timeout_sec: float,
    config_temperature: float = None,
    config_thinking_mode: bool = None,
    config_context_window: int = None,
    config_top_p: float = None,
    config_top_k: int = None,
    config_presence_penalty: float = None,
) -> None:
    import json as _json
    import urllib.request as _urlreq
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    _ensure_bridge_session_logging()
    base_v1, responses_url, models_url, chat_completions_url = _normalize_base_urls(upstream_base_url)
    _log(
        "simple-http mode ready "
        f"run_sha256={_BRIDGE_RUN_SHA256} log_file={_BRIDGE_LOG_PATH} interaction_jsonl={_BRIDGE_INTERACTION_JSONL_PATH} "
        f"upstream_base_v1={base_v1} responses_url={responses_url} "
        f"chat_completions_url={chat_completions_url} default_model={default_model} "
        f"responses_compat_chat={str(RESPONSES_COMPAT_CHAT_ENABLED).lower()} "
        f"responses_input_as_string={str(RESPONSES_INPUT_AS_STRING_ENABLED).lower()} "
        f"responses_input_string_retry={str(RESPONSES_INPUT_STRING_RETRY_ENABLED).lower()} "
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()} "
        f"tui_sse_compat={str(CODEX_BRIDGE_TUI_SSE_COMPAT).lower()}"
    )

    compat_app_state = SimpleNamespace(
        config_temperature=config_temperature,
        config_thinking_mode=config_thinking_mode,
        config_context_window=config_context_window,
        config_top_p=config_top_p,
        config_top_k=config_top_k,
        config_presence_penalty=config_presence_penalty,
    )

    class Handler(BaseHTTPRequestHandler):
        # Codex / 流式客户端常要求 HTTP/1.1；默认 BaseHTTPRequestHandler 为 1.0 会触发
        # “WebSocket protocol error: HTTP version must be 1.1 or higher” 等握手问题。
        protocol_version = "HTTP/1.1"

        def _send_json(self, status: int, payload: dict) -> None:
            data = _json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _proxy_get(self, url: str, auth_header: str) -> None:
            import http.client as _hc
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            try:
                if parsed.scheme == "https":
                    conn = _hc.HTTPSConnection(host, port, timeout=int(timeout_sec))
                else:
                    conn = _hc.HTTPConnection(host, port, timeout=int(timeout_sec))
                conn.request("GET", path, headers={"Authorization": auth_header, "Accept": "*/*"})
                resp = conn.getresponse()
                body = resp.read()
                ct = dict(resp.getheaders()).get("Content-Type", "application/json")
                self.send_response(resp.status)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                conn.close()
            except Exception as exc:
                self._send_json(502, {"error": {"message": f"Upstream models request failed: {exc}"}})

        def _proxy_stream_post(self, url: str, body_obj: dict[str, Any], auth_header: str) -> None:
            """POST JSON upstream and stream response body (chat completions SSE)."""
            import http.client as _hc
            from urllib.parse import urlparse
            parsed = urlparse(url)
            host = parsed.hostname or "localhost"
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            path = parsed.path or "/"
            if parsed.query:
                path = f"{path}?{parsed.query}"
            payload_bytes = _json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
            if parsed.scheme == "https":
                conn = _hc.HTTPSConnection(host, port, timeout=int(timeout_sec))
            else:
                conn = _hc.HTTPConnection(host, port, timeout=int(timeout_sec))
            try:
                conn.request("POST", path, body=payload_bytes, headers={
                    "Authorization": auth_header,
                    "Content-Type": "application/json",
                    "Accept": "*/*",
                })
                resp = conn.getresponse()
            except Exception as exc:
                err_body = _json.dumps({"error": {"message": f"Upstream stream request failed: {exc}"}}).encode("utf-8")
                self.send_response(502)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(err_body)))
                self.end_headers()
                self.wfile.write(err_body)
                conn.close()
                return
            if resp.status >= 400:
                raw = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                conn.close()
                return
            ct = dict(resp.getheaders()).get("Content-Type", "text/event-stream")
            self.send_response(resp.status)
            self.send_header("Content-Type", ct)
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
            finally:
                conn.close()

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/health/"):
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "upstream_base_v1": base_v1,
                        "responses_endpoint": responses_url,
                        "chat_completions_endpoint": chat_completions_url,
                    },
                )
                return
            if self.path in ("/v1/models", "/models"):
                auth_header = _resolve_upstream_authorization_header(
                    self.headers.get("Authorization") or self.headers.get("authorization"),
                    upstream_api_key,
                )
                self._proxy_get(models_url, auth_header)
                return
            self._send_json(404, {"error": {"message": "not_found"}})

        def do_POST(self) -> None:  # noqa: N802
            is_responses = self.path in ("/v1/responses", "/responses")
            is_chat_completions = self.path in ("/v1/chat/completions", "/chat/completions")
            if not (is_responses or is_chat_completions):
                self._send_json(404, {"error": {"message": "not_found"}})
                return
            req_id = uuid.uuid4().hex[:8]
            started_at = time.time()

            length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = _json.loads(raw.decode("utf-8"))
            except Exception as exc:
                _log(f"req={req_id} invalid_json error={_short_text(exc)}")
                self._send_json(400, {"error": {"message": f"Invalid JSON body: {exc}"}})
                return

            raw_model = str(payload.get("model") or "").strip()
            model, model_source, _ = _resolve_model_name(raw_model, default_model)
            if not model:
                _log(f"req={req_id} rejected missing_model")
                self._send_json(400, {"error": {"message": "model is required"}})
                return

            auth_header = _resolve_upstream_authorization_header(
                self.headers.get("Authorization") or self.headers.get("authorization"),
                upstream_api_key,
            )

            def _post_json(url: str, body_obj: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
                import http.client as _hc
                from urllib.parse import urlparse
                parsed = urlparse(url)
                host = parsed.hostname or "localhost"
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                path = parsed.path or "/"
                if parsed.query:
                    path = f"{path}?{parsed.query}"
                payload_bytes = _json.dumps(body_obj, ensure_ascii=False).encode("utf-8")
                if parsed.scheme == "https":
                    conn = _hc.HTTPSConnection(host, port, timeout=int(timeout_sec))
                else:
                    conn = _hc.HTTPConnection(host, port, timeout=int(timeout_sec))
                try:
                    conn.request("POST", path, body=payload_bytes, headers={
                        "Authorization": auth_header,
                        "Content-Type": "application/json",
                        "Accept": "*/*",
                    })
                    resp = conn.getresponse()
                    body = resp.read()
                    hdrs = {k: v for k, v in resp.getheaders()}
                    return int(resp.status), hdrs, body
                except Exception as exc:
                    raise ConnectionError(f"http.client request to {url} failed: {exc}") from exc
                finally:
                    conn.close()

            if is_responses:
                proxied_payload = dict(payload)
                proxied_payload["model"] = model
                proxied_payload = _fold_responses_instructions_into_input(proxied_payload)
                if "input" in proxied_payload:
                    proxied_payload["input"] = _sanitize_responses_input(proxied_payload.get("input"))
                if "tools" in proxied_payload:
                    proxied_payload["tools"] = _normalize_tools_for_vllm_responses(
                        _strip_strict_fields(proxied_payload.get("tools"))
                    )
                input_list_snapshot: list[Any] | None = None
                if isinstance(proxied_payload.get("input"), list):
                    input_list_snapshot = copy.deepcopy(proxied_payload["input"])
                forced_string_input = False
                if RESPONSES_INPUT_AS_STRING_ENABLED and input_list_snapshot is not None:
                    proxied_payload["input"] = _flatten_responses_input_to_string(input_list_snapshot)
                    forced_string_input = True
                    _log(
                        f"req={req_id} responses_input mode=string reason=env_CODEX_BRIDGE_RESPONSES_INPUT_AS_STRING"
                    )
                tools_count = (
                    len(proxied_payload.get("tools") or [])
                    if isinstance(proxied_payload.get("tools"), list)
                    else 0
                )
                is_streaming_request = proxied_payload.get("stream") is True
                tool_call_rounds = _count_tool_call_rounds(proxied_payload.get("input"))
                _log(
                    f"req={req_id} inbound path={self.path} model={model} "
                    f"model_source={model_source} input_roles={_response_input_role_counts(proxied_payload.get('input'))} "
                    f"input_role_seq={_response_input_role_sequence(proxied_payload.get('input'))} "
                    f"tools={_tool_schema_summary(proxied_payload.get('tools'))} "
                    f"stream={str(is_streaming_request).lower()} "
                    f"tool_call_rounds={tool_call_rounds}"
                )

                if tool_call_rounds >= MAX_TOOL_CALL_ROUNDS:
                    _log(
                        f"req={req_id} tool_call_loop_breaker rounds={tool_call_rounds} "
                        f"max={MAX_TOOL_CALL_ROUNDS}"
                    )
                    error_resp = {
                        "id": f"resp_{uuid.uuid4().hex[:16]}",
                        "object": "response",
                        "model": model,
                        "output": [
                            {
                                "type": "message",
                                "id": f"msg_{uuid.uuid4().hex[:8]}",
                                "role": "assistant",
                                "status": "completed",
                                "content": [
                                    {
                                        "type": "output_text",
                                        "text": (
                                            f"Tool call loop detected ({tool_call_rounds} rounds). "
                                            "Stopping to prevent infinite loop. "
                                            "Please rephrase your request or try a different approach."
                                        ),
                                    }
                                ],
                            }
                        ],
                        "status": "completed",
                    }
                    if is_streaming_request:
                        sse_body = _build_minimal_response_sse(error_resp).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Content-Length", str(len(sse_body)))
                        self.end_headers()
                        self.wfile.write(sse_body)
                    else:
                        self._send_json(200, error_resp)
                    return

                if RESPONSES_COMPAT_CHAT_ENABLED:
                    compat_mode = "compat_stream" if is_streaming_request else "compat"
                    compat_payload = _responses_payload_to_chat_payload(proxied_payload, model, compat_app_state)
                    _log(
                        f"req={req_id} responses_to_chat mode={compat_mode} tool_count={tools_count} "
                        f"message_count={len(compat_payload.get('messages') or [])}"
                    )
                    try:
                        compat_status, compat_headers, compat_body = _post_json(
                            chat_completions_url, compat_payload
                        )
                    except Exception as exc:
                        elapsed_ms = int((time.time() - started_at) * 1000)
                        _log(f"req={req_id} compat_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                        self._send_json(502, {"error": {"message": f"Upstream chat.completions request failed: {exc}"}})
                        return

                    elapsed_ms = int((time.time() - started_at) * 1000)
                    if compat_status < 400:
                        compat_json: dict[str, Any] = {}
                        if compat_body:
                            try:
                                parsed_compat = _json.loads(compat_body.decode("utf-8"))
                                if isinstance(parsed_compat, dict):
                                    compat_json = parsed_compat
                            except Exception:
                                compat_json = {}
                        converted = _chat_completion_to_responses_payload(
                            compat_json,
                            model,
                            bool(proxied_payload.get("parallel_tool_calls", True)),
                        )
                        _log(
                            f"req={req_id} compat_done status=200 latency_ms={elapsed_ms} "
                            f"model={model} output_items={len(converted.get('output') or [])}"
                        )
                        _log_dialogue_transcript(
                            req_id,
                            "responses.compat_chat",
                            inbound_input=proxied_payload.get("input"),
                            chat_payload=compat_payload,
                            upstream_chat_completion=compat_json,
                            converted_responses=converted,
                        )
                        if is_streaming_request:
                            sse_raw = (
                                _build_codex_tui_compat_stream_sse(converted)
                                if CODEX_BRIDGE_TUI_SSE_COMPAT
                                else _build_minimal_response_sse(converted)
                            )
                            sse_body = sse_raw.encode("utf-8")
                            self.send_response(200)
                            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                            self.send_header("Content-Length", str(len(sse_body)))
                            self.end_headers()
                            self.wfile.write(sse_body)
                        else:
                            self._send_json(200, converted)
                        return

                    compat_err_msg = _extract_upstream_error_message(
                        (
                            _json.loads(compat_body.decode("utf-8"))
                            if compat_body and compat_body[:1] in (b"{", b"[")
                            else {}
                        ),
                        compat_body.decode("utf-8", errors="ignore") if compat_body else "",
                    )
                    _log(
                        f"req={req_id} compat_status={compat_status} latency_ms={elapsed_ms} "
                        f"error={_short_text(compat_err_msg)} fallback=responses_passthrough"
                    )

                upstream_payload = proxied_payload

                try:
                    status, headers, body = _post_json(responses_url, upstream_payload)
                except Exception as exc:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(f"req={req_id} upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                    self._send_json(502, {"error": {"message": f"Upstream responses request failed: {exc}"}})
                    return

                if status >= 400:
                    parsed: Any = None
                    try:
                        parsed = _json.loads(body.decode("utf-8")) if body else {}
                    except Exception:
                        parsed = None
                    err_msg = _extract_upstream_error_message(
                        parsed, body.decode("utf-8", errors="ignore") if body else ""
                    )
                    if (
                        not forced_string_input
                        and input_list_snapshot is not None
                        and RESPONSES_INPUT_STRING_RETRY_ENABLED
                        and _error_suggests_string_input_required(err_msg, parsed)
                    ):
                        retry_payload = dict(proxied_payload)
                        retry_payload["input"] = _flatten_responses_input_to_string(input_list_snapshot)
                        _log(
                            f"req={req_id} compat_retry reason=responses_input_string "
                            f"input_chars={len(str(retry_payload.get('input') or ''))}"
                        )
                        try:
                            status, headers, body = _post_json(responses_url, retry_payload)
                        except Exception as exc:
                            elapsed_ms = int((time.time() - started_at) * 1000)
                            _log(
                                f"req={req_id} compat_retry_error latency_ms={elapsed_ms} "
                                f"error={_short_text(exc)}"
                            )
                            self._send_json(
                                502, {"error": {"message": f"Upstream responses retry failed: {exc}"}}
                            )
                            return

                    if status >= 400:
                        try:
                            parsed = _json.loads(body.decode("utf-8")) if body else {}
                        except Exception:
                            parsed = None
                        err_msg = _extract_upstream_error_message(
                            parsed, body.decode("utf-8", errors="ignore") if body else ""
                        )
                        if "system message must be at the beginning" in err_msg.lower():
                            retry_payload = dict(upstream_payload)
                            retry_payload["input"] = _demote_system_roles(retry_payload.get("input"))
                            _log(
                                f"req={req_id} compat_retry reason=system_role_order "
                                f"input_role_seq={_response_input_role_sequence(retry_payload.get('input'))}"
                            )
                            try:
                                status, headers, body = _post_json(responses_url, retry_payload)
                            except Exception as exc:
                                elapsed_ms = int((time.time() - started_at) * 1000)
                                _log(
                                    f"req={req_id} compat_retry_error latency_ms={elapsed_ms} "
                                    f"error={_short_text(exc)}"
                                )
                                self._send_json(
                                    502, {"error": {"message": f"Upstream responses retry failed: {exc}"}}
                                )
                                return

                if status >= 400:
                    parsed_err: Any = {}
                    if body:
                        try:
                            parsed_err = _json.loads(body.decode("utf-8"))
                        except Exception:
                            parsed_err = {}
                    err_msg = _extract_upstream_error_message(
                        parsed_err, body.decode("utf-8", errors="ignore") if body else ""
                    )
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(
                        f"req={req_id} upstream_status={status} latency_ms={elapsed_ms} "
                        f"error={_short_text(err_msg)}"
                    )

                if status < 400 and not is_streaming_request and tools_count > 0 and body:
                    try:
                        parsed_ok = _json.loads(body.decode("utf-8"))
                    except Exception:
                        parsed_ok = None
                    if isinstance(parsed_ok, dict):
                        _log(
                            f"req={req_id} output_summary mode=responses "
                            f"{_responses_output_summary(parsed_ok)}"
                        )
                        parsed_ok, converted_textual_tool = _normalize_response_tool_outputs(parsed_ok)
                        if converted_textual_tool:
                            body = _json.dumps(parsed_ok, ensure_ascii=False).encode("utf-8")
                            headers = dict(headers)
                            headers["Content-Type"] = "application/json; charset=utf-8"
                            _log(f"req={req_id} coerced_text_tool_call mode=responses model={model}")

                if status < 400 and is_streaming_request:
                    content_type = str(headers.get("Content-Type") or "").lower()
                    sse_ok = "text/event-stream" in content_type
                    parsed_stream: Any = None
                    if not sse_ok and body:
                        try:
                            parsed_stream = _json.loads(body.decode("utf-8"))
                        except Exception:
                            parsed_stream = None
                    if isinstance(parsed_stream, dict):
                        _log(
                            f"req={req_id} output_summary mode=responses_stream_json "
                            f"{_responses_output_summary(parsed_stream)}"
                        )
                        parsed_stream, converted_textual_tool = _normalize_response_tool_outputs(
                            parsed_stream
                        )
                        parsed_stream = _normalize_reasoning_items_in_response(parsed_stream)
                        body = _build_minimal_response_sse(parsed_stream).encode("utf-8")
                        headers = dict(headers)
                        headers["Content-Type"] = "text/event-stream; charset=utf-8"
                        if converted_textual_tool:
                            _log(f"req={req_id} coerced_text_tool_call mode=responses_stream model={model}")
                        else:
                            _log(f"req={req_id} wrapped_json_to_sse mode=responses_stream model={model}")
                    elif sse_ok and body:
                        sse_text = body.decode("utf-8", errors="ignore")
                        completed = _extract_response_completed_from_sse_body(sse_text)
                        if isinstance(completed, dict):
                            _log(
                                f"req={req_id} output_summary mode=responses_stream_sse "
                                f"{_responses_output_summary(completed)}"
                            )
                            completed, converted_textual_tool = _normalize_response_tool_outputs(
                                completed
                            )
                            completed = _normalize_reasoning_items_in_response(completed)
                            if converted_textual_tool:
                                body = _build_minimal_response_sse(completed).encode("utf-8")
                                headers = dict(headers)
                                headers["Content-Type"] = "text/event-stream; charset=utf-8"
                                _log(
                                    f"req={req_id} coerced_text_tool_call mode=responses_stream_sse_completed "
                                    f"model={model}"
                                )
                        else:
                            _log(
                                f"req={req_id} warn=sse_missing_response_completed "
                                f"model={model} body_prefix={_short_text(body[:220])}"
                            )

                content_type = headers.get("Content-Type", "application/json; charset=utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if is_chat_completions and payload.get("stream") is True:
                up_payload = dict(payload)
                up_payload["model"] = model
                up_payload["stream"] = True
                _log(f"req={req_id} inbound path={self.path} model={model} stream=true mode=upstream_proxy")
                try:
                    self._proxy_stream_post(chat_completions_url, up_payload, auth_header)
                except Exception as exc:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(f"req={req_id} chat_stream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                    self._send_json(502, {"error": {"message": f"Upstream chat stream failed: {exc}"}})
                return

            messages = payload.get("messages") or []
            _log(
                f"req={req_id} inbound path={self.path} model={model} "
                f"model_source={model_source} message_count={len(messages)} roles={_role_counts(messages)}"
            )
            input_items = _chat_messages_to_responses_input(messages)
            responses_payload: dict[str, Any] = {"model": model, "input": input_items}

            max_tokens = payload.get("max_tokens")
            if isinstance(max_tokens, int) and max_tokens > 0:
                responses_payload["max_output_tokens"] = max_tokens
            # 使用 payload 中的 temperature，如果没有则使用配置文件中的默认值
            temperature = payload.get("temperature")
            if temperature is None and hasattr(app.state, "config_temperature") and app.state.config_temperature is not None:
                temperature = app.state.config_temperature
            if isinstance(temperature, (int, float)):
                responses_payload["temperature"] = temperature
            
            # thinking_mode: 优先使用客户端请求中的 enable_thinking，其次使用配置文件
            enable_thinking = payload.get("enable_thinking")
            if enable_thinking is None and hasattr(app.state, "config_thinking_mode") and app.state.config_thinking_mode:
                enable_thinking = True
            
            if enable_thinking is not None:
                responses_payload["enable_thinking"] = enable_thinking
                # 也保留 reasoning_effort 以兼容其他实现
                if enable_thinking:
                    reasoning_effort = payload.get("reasoning_effort", "high")
                    responses_payload["reasoning_effort"] = reasoning_effort
                    _log(f"req={req_id} thinking_mode enabled: enable_thinking=True, reasoning_effort={reasoning_effort}")
            
            # context_window: 如果有设置，添加到 metadata 或作为额外参数
            if hasattr(app.state, "config_context_window") and app.state.config_context_window:
                responses_payload["context_window"] = app.state.config_context_window
            top_p = payload.get("top_p")
            if isinstance(top_p, (int, float)):
                responses_payload["top_p"] = top_p

            # top_k: vLLM 扩展参数，需要通过 extra_body 传递
            top_k = payload.get("top_k")
            if isinstance(top_k, (int, float)) and top_k > 0:
                # vLLM 要求 top_k 放在 extra_body 中
                if "extra_body" not in responses_payload:
                    responses_payload["extra_body"] = {}
                responses_payload["extra_body"]["top_k"] = int(top_k)

            # presence_penalty: OpenAI 标准参数
            presence_penalty = payload.get("presence_penalty")
            if isinstance(presence_penalty, (int, float)):
                responses_payload["presence_penalty"] = presence_penalty

            try:
                status, headers, body = _post_json(responses_url, responses_payload)
            except Exception as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(f"req={req_id} upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                self._send_json(502, {"error": {"message": f"Upstream responses request failed: {exc}"}})
                return

            if status in (404, 405):
                _log(f"req={req_id} responses_unavailable status={status} fallback=chat_completions")
                fallback_payload = dict(payload)
                fallback_payload["model"] = model
                fallback_payload["stream"] = False
                if not isinstance(fallback_payload.get("messages"), list):
                    fallback_payload["messages"] = messages if isinstance(messages, list) else []
                try:
                    fb_status, fb_headers, fb_body = _post_json(chat_completions_url, fallback_payload)
                except Exception as exc:
                    elapsed_ms = int((time.time() - started_at) * 1000)
                    _log(f"req={req_id} fallback_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                    self._send_json(502, {"error": {"message": f"Upstream chat.completions request failed: {exc}"}})
                    return

                if fb_status >= 400:
                    content_type = fb_headers.get("Content-Type", "application/json; charset=utf-8")
                    self.send_response(fb_status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(fb_body)))
                    self.end_headers()
                    self.wfile.write(fb_body)
                    return

                try:
                    fb_json = _json.loads(fb_body.decode("utf-8")) if fb_body else {}
                except Exception:
                    fb_json = {}
                fb_usage = _usage_from_chat_completion(fb_json if isinstance(fb_json, dict) else {})
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(
                    f"req={req_id} done status=200 mode=chat_completions_fallback latency_ms={elapsed_ms} "
                    f"model={model} prompt_tokens={fb_usage.get('prompt_tokens', 0)} "
                    f"completion_tokens={fb_usage.get('completion_tokens', 0)} "
                    f"total_tokens={fb_usage.get('total_tokens', 0)}"
                )
                content_type = fb_headers.get("Content-Type", "application/json; charset=utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(fb_body)))
                self.end_headers()
                self.wfile.write(fb_body)
                return

            if status >= 400:
                content_type = headers.get("Content-Type", "application/json; charset=utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            resp_data = _json.loads(body.decode("utf-8"))
            assistant_text = _extract_output_text(resp_data)
            reasoning_text = _extract_reasoning_text(resp_data)
            usage = _usage_from_responses(resp_data)
            result = _build_chat_completion(model, assistant_text, usage, reasoning_content=reasoning_text)
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(
                f"req={req_id} done status=200 latency_ms={elapsed_ms} model={model} "
                f"prompt_tokens={usage.get('prompt_tokens', 0)} completion_tokens={usage.get('completion_tokens', 0)} "
                f"total_tokens={usage.get('total_tokens', 0)} output_chars={len(assistant_text)} "
                f"reasoning_chars={len(reasoning_text)}"
            )
            self._send_json(200, result)

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


def _kill_process_on_port(port: int) -> None:
    """Kill any process listening on the given port (cross-platform)."""
    import subprocess
    import signal
    
    try:
        # Try lsof first (macOS/Linux)
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid_str in pids:
                try:
                    pid = int(pid_str.strip())
                    _log(f"killing process {pid} using port {port}")
                    os.kill(pid, signal.SIGTERM)
                except (ValueError, ProcessLookupError, PermissionError) as e:
                    _log(f"failed to kill pid {pid}: {e}")
            # Wait a moment for graceful shutdown
            time.sleep(0.5)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback to fuser (Linux)
    try:
        result = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            _log(f"killed process on port {port} using fuser")
            time.sleep(0.5)
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Fallback to netstat + taskkill (Windows)
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    if len(parts) >= 5:
                        pid = parts[-1]
                        try:
                            pid_int = int(pid)
                            _log(f"killing windows process {pid_int} on port {port}")
                            subprocess.run(
                                ["taskkill", "/PID", str(pid_int), "/F"],
                                capture_output=True,
                                timeout=5
                            )
                        except (ValueError, subprocess.TimeoutExpired):
                            pass
                    break
            time.sleep(0.5)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def main() -> None:
    args = _parse_args()
    _ensure_bridge_config(args)

    if not args.upstream_base_url or not args.upstream_api_key or not args.default_model:
        raise RuntimeError("Missing upstream settings (base_url/api_key/model). Provide args or use --use-codex-config or ensure llm_config.json exists.")

    # Kill any process using the target port before starting
    _kill_process_on_port(args.port)

    base_v1, responses_url, _, chat_completions_url = _normalize_base_urls(args.upstream_base_url)
    _log(
        f"startup listen={args.host}:{args.port} run_sha256={_BRIDGE_RUN_SHA256} "
        f"log_file={_BRIDGE_LOG_PATH} interaction_jsonl={_BRIDGE_INTERACTION_JSONL_PATH} "
        f"use_codex_config={str(args.use_codex_config).lower()} "
        f"codex_home={_resolve_codex_home()} upstream_base_v1={base_v1} responses_url={responses_url} "
        f"chat_completions_url={chat_completions_url} "
        f"default_model={args.default_model} responses_compat_chat={str(RESPONSES_COMPAT_CHAT_ENABLED).lower()} "
        f"responses_input_as_string={str(RESPONSES_INPUT_AS_STRING_ENABLED).lower()} "
        f"responses_input_string_retry={str(RESPONSES_INPUT_STRING_RETRY_ENABLED).lower()} "
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()} "
        f"tui_sse_compat={str(CODEX_BRIDGE_TUI_SSE_COMPAT).lower()}"
    )

    try:
        import uvicorn  # noqa: F401
        app = create_app(
            args.upstream_base_url,
            args.upstream_api_key,
            args.default_model,
            args.timeout_sec,
            getattr(args, "config_temperature", None),
            getattr(args, "config_thinking_mode", None),
            getattr(args, "config_context_window", None),
            getattr(args, "config_top_p", None),
            getattr(args, "config_top_k", None),
            getattr(args, "config_presence_penalty", None),
        )
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    except Exception as exc:
        _log(f"fastapi/uvicorn unavailable, falling back to simple-http: {_short_text(exc)}")
        _run_simple_server(
            args.host,
            args.port,
            args.upstream_base_url,
            args.upstream_api_key,
            args.default_model,
            args.timeout_sec,
            getattr(args, "config_temperature", None),
            getattr(args, "config_thinking_mode", None),
            getattr(args, "config_context_window", None),
            getattr(args, "config_top_p", None),
            getattr(args, "config_top_k", None),
            getattr(args, "config_presence_penalty", None),
        )


if __name__ == "__main__":
    main()
