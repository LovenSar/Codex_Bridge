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
from typing import Any

_BRIDGE_RUN_SHA256: str | None = None
_BRIDGE_LOG_PATH: Path | None = None
_BRIDGE_INTERACTION_JSONL_PATH: Path | None = None
_LOG_FILE_LOCK = threading.Lock()
_LOG_MAX_PAYLOAD_STR = int(os.getenv("CODEX_BRIDGE_LOG_MAX_STR", "500000"))


def _ensure_bridge_session_logging() -> None:
    """Once per process: SHA256 run id + log file paths under CODEX_BRIDGE_LOG_DIR (default: ../logs)."""
    global _BRIDGE_RUN_SHA256, _BRIDGE_LOG_PATH, _BRIDGE_INTERACTION_JSONL_PATH
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
        ts = datetime.datetime.now().isoformat(timespec="milliseconds")
        banner = (
            f"{ts}\tbridge_session_start\trun_sha256={_BRIDGE_RUN_SHA256}\t"
            f"log={_BRIDGE_LOG_PATH.name}\tinteraction={_BRIDGE_INTERACTION_JSONL_PATH.name}\n"
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


def _log_interaction_json(req_id: str, event: str, data: Any) -> None:
    """Append one JSON line per event for full replay / audit (interaction.jsonl)."""
    _ensure_bridge_session_logging()
    if not _BRIDGE_INTERACTION_JSONL_PATH or not _BRIDGE_RUN_SHA256:
        return
    record = {
        "ts": datetime.datetime.now().isoformat(timespec="milliseconds"),
        "run_sha256": _BRIDGE_RUN_SHA256,
        "req_id": req_id,
        "event": event,
        "data": _truncate_for_log(data),
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
    print(console, flush=True)
    if _BRIDGE_LOG_PATH:
        try:
            line = f"{ts}\t[run:{rid}]\t{message}\n"
            with _LOG_FILE_LOCK:
                with open(_BRIDGE_LOG_PATH, "a", encoding="utf-8") as f:
                    f.write(line)
        except Exception:
            pass


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

MAX_TOOL_CALL_ROUNDS = int(os.getenv("CODEX_BRIDGE_MAX_TOOL_CALL_ROUNDS", "40"))


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
            if item_type in ("function_call", "message"):
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
        lines.append(f"data: {json.dumps(payload, ensure_ascii=False)}")
        lines.append("")
    lines.append("data: [DONE]")
    lines.append("")
    return "\n".join(lines)


def _sse_data_line_bytes(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


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

    return StreamingResponse(gen(), status_code=resp.status_code, media_type=media_type)


def _accumulate_chat_completion_chunk(state: dict[str, Any], data_obj: dict[str, Any]) -> None:
    choices = data_obj.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    ch0 = choices[0] if isinstance(choices[0], dict) else {}
    delta = ch0.get("delta") if isinstance(ch0, dict) else {}
    if not isinstance(delta, dict):
        return
    if delta.get("role"):
        state["role"] = delta.get("role")
    if delta.get("content") is not None:
        state["content"] = str(state.get("content") or "") + str(delta.get("content"))
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


def _accumulated_stream_state_to_message(state: dict[str, Any]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": state.get("content") or ""}
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
    """
    from starlette.responses import StreamingResponse

    response_id = f"resp_{uuid.uuid4().hex[:16]}"

    async def gen():
        state: dict[str, Any] = {}
        try:
            async for line in resp.aiter_lines():
                ls = (line or "").strip()
                if not ls.startswith("data:"):
                    continue
                raw = ls[len("data:") :].strip()
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
                delta = ch0.get("delta") if isinstance(ch0, dict) else {}
                if isinstance(delta, dict) and delta.get("content"):
                    ev = {
                        "type": "response.output_text.delta",
                        "response_id": response_id,
                        "delta": str(delta.get("content")),
                    }
                    yield _sse_data_line_bytes(ev)
            msg = _accumulated_stream_state_to_message(state)
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
            yield _sse_data_line_bytes({"type": "response.completed", "response": conv})
            yield b"data: [DONE]\n\n"
        finally:
            await _httpx_streaming_body_close(resp)

    return StreamingResponse(
        gen(),
        status_code=200,
        media_type="text/event-stream; charset=utf-8",
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


def _responses_payload_to_chat_payload(payload: dict[str, Any], model: str) -> dict[str, Any]:
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
    if isinstance(payload.get("temperature"), (int, float)):
        chat_payload["temperature"] = payload.get("temperature")
    if isinstance(payload.get("top_p"), (int, float)):
        chat_payload["top_p"] = payload.get("top_p")
    max_output_tokens = payload.get("max_output_tokens")
    if isinstance(max_output_tokens, int) and max_output_tokens > 0:
        chat_payload["max_tokens"] = max_output_tokens
    return chat_payload


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
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
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

    if not output:
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
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{uuid.uuid4().hex[:8]}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": content}],
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
    if not isinstance(resp, dict):
        return ""
    reasoning_text = _extract_reasoning_text(resp)
    output = resp.get("output")
    assistant_text = ""
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
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
    # Fallback to top-level text fields
    if not assistant_text and "output_text" in resp:
        assistant_text = _extract_text_from_content(resp.get("output_text"))
    if reasoning_text:
        if assistant_text:
            return f"<think>{reasoning_text}</think>\n{assistant_text}"
        return f"<think>{reasoning_text}</think>"
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


def _build_chat_completion(model: str, text: str, usage: dict[str, Any]) -> dict[str, Any]:
    created_at = int(time.time())
    return {
        "id": f"chatcmpl_{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": created_at,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
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


def create_app(upstream_base_url: str, upstream_api_key: str, default_model: str, timeout_sec: float) -> Any:
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
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()}"
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

        model = str(payload.get("model") or app.state.default_model).strip()
        model_source = "payload" if str(payload.get("model") or "").strip() else "default"
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
                return Response(
                    content=sse_body, status_code=200,
                    media_type="text/event-stream; charset=utf-8",
                )
            return JSONResponse(status_code=200, content=error_resp)

        if RESPONSES_COMPAT_CHAT_ENABLED:
            compat_mode = "compat_stream" if is_streaming_request else "compat"
            chat_payload = _responses_payload_to_chat_payload(proxied_payload, model)
            _log(
                f"req={req_id} responses_to_chat mode={compat_mode} tool_count={tools_count} "
                f"message_count={len(chat_payload.get('messages') or [])}"
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
                _log_interaction_json(
                    req_id,
                    "responses.compat_chat_done",
                    {
                        "upstream_chat_completion": chat_data,
                        "converted_responses": converted,
                        "output_lines": _responses_output_full_lines(converted),
                    },
                )
                if is_streaming_request:
                    sse_body = _build_minimal_response_sse(converted)
                    return Response(
                        content=sse_body, status_code=200,
                        media_type="text/event-stream; charset=utf-8",
                    )
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

        if False and (
            RESPONSES_COMPAT_CHAT_ENABLED
            and is_streaming_request
            and RESPONSES_COMPAT_CHAT_STREAM_ENABLED
        ):
            chat_payload = _responses_payload_to_chat_payload(proxied_payload, model)
            chat_payload["stream"] = True
            _log(
                f"req={req_id} responses_to_chat mode=compat_stream_native tool_count={tools_count} "
                f"message_count={len(chat_payload.get('messages') or [])}"
            )
            try:
                creq = app.state.client.build_request(
                    "POST", app.state.upstream_chat_completions_url, json=chat_payload, headers=headers
                )
                compat_r = await app.state.client.send(creq, stream=True)
            except httpx.HTTPError as exc:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(f"req={req_id} compat_stream_upstream_error latency_ms={elapsed_ms} error={_short_text(exc)}")
                raise HTTPException(
                    status_code=502, detail=f"Upstream chat.completions stream failed: {exc}"
                ) from exc
            if compat_r.status_code < 400:
                elapsed_ms = int((time.time() - started_at) * 1000)
                _log(
                    f"req={req_id} compat_stream_ok latency_ms={elapsed_ms} model={model} mode=responses_sse_transcode"
                )
                return await _translate_chat_sse_to_response_sse(
                    compat_r,
                    model,
                    bool(proxied_payload.get("parallel_tool_calls", True)),
                )
            err_raw = await compat_r.aread()
            await _httpx_streaming_body_close(compat_r)
            try:
                err_data = json.loads(err_raw.decode("utf-8")) if err_raw else {}
            except Exception:
                err_data = {"error": {"message": err_raw.decode("utf-8", errors="ignore")}}
            compat_err = _extract_upstream_error_message(err_data, err_raw.decode("utf-8", errors="ignore"))
            _log(
                f"req={req_id} compat_stream_status={compat_r.status_code} "
                f"error={_short_text(compat_err)} fallback=responses_stream"
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
                )
                data, converted_textual_tool = _normalize_response_tool_outputs(data)
                body_text = _build_minimal_response_sse(data)
                if converted_textual_tool:
                    _log(f"req={req_id} coerced_text_tool_call mode=responses_stream model={model}")
                else:
                    _log(f"req={req_id} wrapped_json_to_sse mode=responses_stream model={model}")
                return Response(content=body_text, status_code=r.status_code, media_type="text/event-stream")
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

        model = str(payload.get("model") or app.state.default_model).strip()
        model_source = "payload" if str(payload.get("model") or "").strip() else "default"
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
        temperature = payload.get("temperature")
        if isinstance(temperature, (int, float)):
            responses_payload["temperature"] = temperature
        top_p = payload.get("top_p")
        if isinstance(top_p, (int, float)):
            responses_payload["top_p"] = top_p

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
        usage = _usage_from_responses(resp_data)
        result = _build_chat_completion(model, assistant_text, usage)
        elapsed_ms = int((time.time() - started_at) * 1000)
        _log(
            f"req={req_id} done status=200 latency_ms={elapsed_ms} model={model} "
            f"prompt_tokens={usage.get('prompt_tokens', 0)} completion_tokens={usage.get('completion_tokens', 0)} "
            f"total_tokens={usage.get('total_tokens', 0)} output_chars={len(assistant_text)}"
        )
        _log_interaction_json(
            req_id,
            "chat_completions.done",
            {
                "assistant_text": assistant_text,
                "usage": usage,
                "upstream_responses_json": resp_data,
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


def _ensure_bridge_config(args: argparse.Namespace) -> None:
    """Fill upstream_base_url / api_key / default_model from llm_config.json or ~/.codex."""
    if not args.upstream_base_url or not args.upstream_api_key or not args.default_model:
        plugin_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llm_config.json")
        if os.path.exists(plugin_config_path):
            try:
                with open(plugin_config_path, "r", encoding="utf-8") as f:
                    plugin_cfg = json.load(f)
                if not args.upstream_base_url:
                    args.upstream_base_url = plugin_cfg.get("api_base_url")
                if not args.upstream_api_key:
                    args.upstream_api_key = plugin_cfg.get("api_key", "dummy")
                if not args.default_model:
                    args.default_model = plugin_cfg.get("model_name")
                _log(f"loaded config from plugin llm_config.json: {plugin_config_path}")
            except Exception as e:
                _log(f"failed to load plugin llm_config.json: {e}")

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
        timeout_sec=120.0,
        use_codex_config=False,
    )
    _ensure_bridge_config(args)
    if not args.upstream_base_url or not args.upstream_api_key or not args.default_model:
        raise RuntimeError(
            "Missing upstream settings. Add llm_config.json next to the plugin or set CLI via python chat_bridge.py."
        )
    return create_app(args.upstream_base_url, args.upstream_api_key, args.default_model, args.timeout_sec)


def _run_simple_server(
    host: str,
    port: int,
    upstream_base_url: str,
    upstream_api_key: str,
    default_model: str,
    timeout_sec: float,
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
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()}"
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

            model = str(payload.get("model") or default_model).strip()
            model_source = "payload" if str(payload.get("model") or "").strip() else "default"
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
                    compat_payload = _responses_payload_to_chat_payload(proxied_payload, model)
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
                        if is_streaming_request:
                            sse_body = _build_minimal_response_sse(converted).encode("utf-8")
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
            temperature = payload.get("temperature")
            if isinstance(temperature, (int, float)):
                responses_payload["temperature"] = temperature
            top_p = payload.get("top_p")
            if isinstance(top_p, (int, float)):
                responses_payload["top_p"] = top_p

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
            usage = _usage_from_responses(resp_data)
            result = _build_chat_completion(model, assistant_text, usage)
            elapsed_ms = int((time.time() - started_at) * 1000)
            _log(
                f"req={req_id} done status=200 latency_ms={elapsed_ms} model={model} "
                f"prompt_tokens={usage.get('prompt_tokens', 0)} completion_tokens={usage.get('completion_tokens', 0)} "
                f"total_tokens={usage.get('total_tokens', 0)} output_chars={len(assistant_text)}"
            )
            self._send_json(200, result)

    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


def main() -> None:
    args = _parse_args()
    _ensure_bridge_config(args)

    if not args.upstream_base_url or not args.upstream_api_key or not args.default_model:
        raise RuntimeError("Missing upstream settings (base_url/api_key/model). Provide args or use --use-codex-config or ensure llm_config.json exists.")

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
        f"compat_chat_stream={str(RESPONSES_COMPAT_CHAT_STREAM_ENABLED).lower()}"
    )

    try:
        import uvicorn  # noqa: F401
        app = create_app(args.upstream_base_url, args.upstream_api_key, args.default_model, args.timeout_sec)
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
        )


if __name__ == "__main__":
    main()
