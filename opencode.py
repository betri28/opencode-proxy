"""
Claude Code Proxy → opencode.ai
Convert Anthropic /v1/messages ↔ OpenAI chat/completions
"""

import json
import uuid
import time
import base64
import logging
import os
import sqlite3
import threading
import asyncio
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from config import API_KEY, PROXY, MODELS, ROUTES, get_model_config, HOST, PORT, WEB_PORT, NO_MULTIMODAL

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    _encoding = None

from dashboard import register_dashboard
from dashboard.display import log as _log, RichLogHandler, run_terminal_loop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# SQLite setup
_db_path = os.path.join(LOG_DIR, "requests.db")
_conn = sqlite3.connect(_db_path, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_conn.execute("PRAGMA journal_mode=WAL")
_db_lock = threading.Lock()
_conn.execute("""
    CREATE TABLE IF NOT EXISTS requests (
        id TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        model TEXT NOT NULL,
        original_model TEXT,
        duration_ms INTEGER,
        tokens_input INTEGER,
        tokens_output INTEGER,
        tokens_cache INTEGER,
        success INTEGER,
        error TEXT,
        protocol TEXT,
        is_stream INTEGER,
        thinking TEXT,
        effort TEXT
    )
""")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")
for col, default in [("protocol", "NULL"), ("is_stream", "0"), ("thinking", "NULL"), ("effort", "NULL")]:
    try:
        _conn.execute(f"ALTER TABLE requests ADD COLUMN {col} TEXT DEFAULT {default}")
    except Exception:
        pass
_conn.commit()


def _save_request(req_id, model, original_model, duration_ms,
                  tokens_input, tokens_output, tokens_cache, success=True, error=None,
                  protocol=None, is_stream=False, thinking=None, effort=None):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        _conn.execute("""
            INSERT OR REPLACE INTO requests (id, timestamp, model, original_model, duration_ms,
                tokens_input, tokens_output, tokens_cache, success, error,
                protocol, is_stream, thinking, effort)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (req_id, timestamp, model, original_model, duration_ms,
              tokens_input, tokens_output, tokens_cache, 1 if success else 0, error,
              protocol, 1 if is_stream else 0, thinking, effort))
        _conn.commit()


# Token usage tracking (in-memory, lost on restart)
import collections as _collections
_token_usage = _collections.defaultdict(lambda: {"input": 0, "output": 0, "cache": 0})
for model in MODELS:
    _token_usage[model] = {"input": 0, "output": 0, "cache": 0}
_token_lock = threading.Lock()

# Shared HTTP client (reused across requests)
_transport = httpx.AsyncHTTPTransport(proxy=PROXY) if PROXY else None
_client = httpx.AsyncClient(transport=_transport, timeout=300)


@asynccontextmanager
async def lifespan(app):
    yield
    await _client.aclose()
    _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    _conn.close()

app = FastAPI(lifespan=lifespan)
register_dashboard(app, STATIC_DIR, _conn, _db_lock)


_IMAGE_FORMATS = ("image/jpeg", "image/png", "image/webp", "image/gif")

_DOCUMENT_FORMATS = (
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

_TEXT_FORMATS = (
    "text/plain", "text/csv", "text/html", "text/markdown",
    "application/json", "text/xml",
)


def _is_supported_image(mt: str) -> bool:
    return mt.lower() in _IMAGE_FORMATS


def _is_text_format(mt: str) -> bool:
    return mt.lower() in _TEXT_FORMATS


def _convert_media_block(source: dict, mt: str):
    source_type = source.get("type", "")
    if _is_text_format(mt):
        if source_type == "base64" and source.get("data"):
            try:
                decoded = base64.b64decode(source["data"]).decode("utf-8", errors="replace")
                return {"type": "text", "text": decoded}
            except Exception:
                _log(f"  WARN: failed to decode text document ({mt})")
                return None
        elif source_type == "url" and source.get("url"):
            return {"type": "text", "text": source["url"]}
        return None
    if mt.lower() in _DOCUMENT_FORMATS:
        ext_map = {
            "application/pdf": "document.pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document.docx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "document.xlsx",
        }
        filename = ext_map.get(mt.lower(), "document")
        if source_type == "base64" and source.get("data"):
            return {"type": "file", "file": {"filename": filename, "file_data": f"data:{mt};base64,{source['data']}"}}
        elif source_type == "url" and source.get("url"):
            return {"type": "file", "file": {"filename": filename, "file_data": source["url"]}}
        return None
    return None


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _forward_headers(resp_headers) -> dict:
    """Forward informational upstream headers to the client response."""
    headers = {}
    for key, value in resp_headers.items():
        kl = key.lower()
        if kl.startswith(("x-request-id", "x-ratelimit", "openai-", "anthropic-", "cf-", "x-cache", "x-gg")):
            headers[key] = value
    return headers


def _route_for(model_name: str) -> dict:
    name = model_name.lower().strip()
    if not name:
        return ROUTES["sonnet"]
    for r in ROUTES.values():
        if any(m in name for m in r["match"]):
            return r
    return ROUTES["sonnet"]


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for i in content:
            if isinstance(i, str):
                parts.append(i)
            elif isinstance(i, dict):
                if i.get("type") == "text":
                    parts.append(i.get("text", ""))
                elif i.get("type") == "thinking":
                    parts.append(i.get("thinking", ""))
                elif i.get("type") == "image":
                    parts.append(f"[image:{i.get('source', {}).get('type', 'unknown')}]")
                elif i.get("type") == "document":
                    parts.append(f"[document:{i.get('source', {}).get('media_type', 'unknown')}]")
                elif i.get("type") == "redacted_thinking":
                    parts.append("[redacted]")
                else:
                    parts.append(i.get("text", json.dumps(i, ensure_ascii=False)))
        return "\n".join(parts)
    return str(content) if content else ""


def anthropic_to_openai(body: dict, model: str) -> dict:
    thinking = isinstance(body.get("thinking"), dict) and body["thinking"].get("type") in ("enabled", "adaptive")

    messages = []

    # System prompt
    system = body.get("system", "")
    if isinstance(system, str):
        if system:
            messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        sys_blocks = []
        for block in system:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    sys_blocks.append({"type": "text", "text": block.get("text", "")})
                elif btype == "image":
                    source = block.get("source", {})
                    mt = source.get("media_type", "")
                    if _is_supported_image(mt):
                        if source.get("type") == "base64":
                            img_url = f"data:{mt};base64,{source['data']}"
                        else:
                            img_url = source.get("url", "")
                        sys_blocks.append({"type": "image_url", "image_url": {"url": img_url}})
                elif btype == "document":
                    source = block.get("source", {})
                    mt = source.get("media_type", "")
                    if _is_supported_image(mt):
                        if source.get("type") == "base64":
                            doc_url = f"data:{mt};base64,{source['data']}"
                        else:
                            doc_url = source.get("url", "")
                        sys_blocks.append({"type": "image_url", "image_url": {"url": doc_url}})
                    else:
                        converted = _convert_media_block(source, mt)
                        if converted:
                            sys_blocks.append(converted)
                        elif mt:
                            _log(f"  WARN: dropped unsupported file type in system: {mt}")
        if sys_blocks:
            # If only text blocks (no media), join as plain string for maximum compatibility
            has_sys_media = any(b.get("type") not in ("text",) for b in sys_blocks)
            if has_sys_media and model not in NO_MULTIMODAL:
                messages.append({"role": "system", "content": sys_blocks})
            else:
                sys_texts = [b.get("text", "") for b in sys_blocks if b.get("type") == "text"]
                messages.append({"role": "system", "content": "\n".join(sys_texts)})
                if has_sys_media:
                    _log(f"  WARN: model {model} does not support multimodal system content, media dropped")

    for msg in body.get("messages", []):
        role, content = msg["role"], msg.get("content", "")
        is_asst = role == "assistant"

        # Simple string content
        if isinstance(content, str):
            out = {"role": role, "content": content}
            if thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
            continue

        if not isinstance(content, list):
            continue

        text_parts, content_blocks, tool_calls, thinking_parts, tool_results = [], [], [], [], []
        has_media = False

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                content_blocks.append({"type": "text", "text": block})
                continue
            if not isinstance(block, dict):
                continue

            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
                content_blocks.append({"type": "text", "text": block.get("text", "")})
            elif btype == "image":
                source = block.get("source", {})
                mt = source.get("media_type", "")
                if _is_supported_image(mt):
                    has_media = True
                    if source.get("type") == "base64":
                        img_url = f"data:{mt};base64,{source['data']}"
                    else:
                        img_url = source.get("url", "")
                    content_blocks.append({"type": "image_url", "image_url": {"url": img_url}})
            elif btype == "document":
                source = block.get("source", {})
                mt = source.get("media_type", "")
                if _is_supported_image(mt):
                    has_media = True
                    if source.get("type") == "base64":
                        doc_url = f"data:{mt};base64,{source['data']}"
                    else:
                        doc_url = source.get("url", "")
                    content_blocks.append({"type": "image_url", "image_url": {"url": doc_url}})
                else:
                    converted = _convert_media_block(source, mt)
                    if converted:
                        has_media = True
                        content_blocks.append(converted)
                    elif mt:
                        _log(f"  WARN: dropped unsupported file type in message: {mt}")
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                })
            elif btype == "tool_result":
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id", ""),
                    "content": _extract_text(block.get("content", "")),
                })

        # Emit tool_result messages first (must immediately follow assistant's tool_calls)
        messages.extend(tool_results)

        # Then emit the main message (text + images + tool_calls + thinking)
        joined_thinking = "\n".join(thinking_parts) if thinking_parts else ""
        force_string = model in NO_MULTIMODAL
        if force_string and has_media:
            _log(f"  WARN: model {model} does not support multimodal content, media dropped")
        if tool_calls:
            out = {
                "role": role,
                "content": None,
                "tool_calls": tool_calls,
            }
            if joined_thinking:
                out["reasoning_content"] = joined_thinking
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
        elif has_media and not force_string:
            out = {"role": role, "content": content_blocks}
            if joined_thinking:
                out["reasoning_content"] = joined_thinking
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
        elif text_parts or thinking_parts or (thinking and is_asst) or has_media:
            out = {"role": role, "content": "\n".join(text_parts) if text_parts else ""}
            if joined_thinking:
                out["reasoning_content"] = joined_thinking
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)

    # Build request
    oai = {"model": model, "messages": messages,
           "max_tokens": body.get("max_tokens", 16384),
           "stream": body.get("stream", False)}

    if thinking and isinstance(body.get("thinking"), dict) and body["thinking"].get("budget_tokens"):
        oai["max_completion_tokens"] = body["thinking"]["budget_tokens"]

    for key, oai_key in [("temperature", "temperature"), ("top_p", "top_p"), ("stop_sequences", "stop")]:
        if key in body:
            oai[oai_key] = body[key]

    if "tools" in body:
        oai["tools"] = [{"type": "function", "function": {
            "name": t["name"], "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }} for t in body["tools"]]
        tc = body.get("tool_choice", "auto")
        if isinstance(tc, dict):
            tc_type = tc.get("type", "auto")
            if tc_type == "tool":
                oai["tool_choice"] = {"type": "function", "function": {"name": tc.get("name", "")}}
            elif tc_type == "any":
                oai["tool_choice"] = "required"
            else:
                oai["tool_choice"] = "auto"
        else:
            oai["tool_choice"] = tc

    return oai


def openai_to_anthropic(resp: dict, model: str) -> dict:
    choice = resp.get("choices", [{}])[0]
    msg = choice.get("message", {})
    usage = resp.get("usage", {})

    blocks = []
    if reasoning := msg.get("reasoning_content"):
        blocks.append({"type": "thinking", "thinking": reasoning})
    content = msg.get("content")
    if content:
        text = content if isinstance(content, str) else "".join(
            p.get("text", "") for p in content if isinstance(p, dict)
        )
        blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls", []):
        fn = tc.get("function", {})
        try:
            inp = json.loads(fn.get("arguments", "{}"))
        except Exception:
            inp = {}
        blocks.append({"type": "tool_use", "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}"),
                        "name": fn.get("name", ""), "input": inp})

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use", "content_filter": "content_filter"}
    stop = stop_reason_map.get(choice.get("finish_reason", ""), "end_turn")
    if msg.get("tool_calls"):
        stop = "tool_use"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "content": blocks, "model": model, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": _get_output_tokens(usage),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0) or usage.get("prompt_tokens_details", {}).get("cache_creation", 0),
                "cache_read_input_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)},
    }


def _estimate_tokens(text: str) -> int:
    if _encoding:
        return len(_encoding.encode(text))
    return max(1, len(text) // 3)


def _estimate_input_tokens(body: dict) -> int:
    """Estimate input tokens from message content, tools, and tool_results."""
    chunks = []

    # System prompt
    system = body.get("system", "")
    if isinstance(system, str):
        chunks.append(system)
    elif isinstance(system, list):
        for s in system:
            if isinstance(s, str):
                chunks.append(s)
            elif isinstance(s, dict):
                if s.get("type") == "image":
                    source = s.get("source", {})
                    chunks.append(source.get("data", "") or source.get("url", ""))
                elif s.get("type") == "document":
                    source = s.get("source", {})
                    chunks.append(source.get("data", "") or source.get("url", ""))
                else:
                    chunks.append(s.get("text", ""))

    # Tools definitions
    for tool in body.get("tools", []):
        chunks.append(tool.get("name", ""))
        chunks.append(tool.get("description", ""))
        chunks.append(str(tool.get("input_schema", {})))

    # Messages
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, str):
                    chunks.append(block)
                elif isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "tool_result":
                        chunks.append(_extract_text(block.get("content", "")))
                    elif btype == "thinking":
                        chunks.append(block.get("thinking", ""))
                    elif btype == "image":
                        source = block.get("source", {})
                        chunks.append(source.get("data", "") or source.get("url", ""))
                    elif btype == "document":
                        source = block.get("source", {})
                        chunks.append(source.get("data", "") or source.get("url", ""))
                    else:
                        chunks.append(block.get("text", ""))
                        chunks.append(str(block.get("input", "")))

    combined = "\n".join(chunks)
    if _encoding:
        return len(_encoding.encode(combined))
    return max(1, len(combined) // 3)


def _extract_cache_tokens(usage: dict) -> int:
    """Return total cached tokens (read + creation) from usage."""
    total = 0
    details = usage.get("prompt_tokens_details") or {}
    # Cache read tokens (multiple possible field names across APIs)
    read = (details.get("cached_tokens") or usage.get("cached_tokens") or
            usage.get("cache_read_input_tokens") or 0)
    total += read
    # Cache creation tokens
    creation = details.get("cache_creation") or usage.get("cache_creation_input_tokens") or 0
    total += creation
    return total


def _get_output_tokens(usage: dict) -> int:
    """Get output tokens including reasoning_tokens from completion_tokens_details."""
    output = usage.get("completion_tokens") or 0
    details = usage.get("completion_tokens_details") or {}
    output += details.get("reasoning_tokens") or 0
    return output


def _elapsed_ms(start_time: float) -> int:
    return max(1, int((time.time() - start_time) * 1000))


@app.api_route("/v1/messages", methods=["POST"])
@app.api_route("/anthropic/v1/messages", methods=["POST"])
async def messages(request: Request):
    req_id = f"msg_{uuid.uuid4().hex[:24]}"
    start_time = time.time()

    try:
        body = json.loads(await request.body())
    except Exception:
        return Response(content='{"error":"invalid json"}', status_code=400)

    original_model = body.get("model", "")
    route = _route_for(original_model)
    model_id = route["model"]
    cfg = get_model_config(model_id)
    endpoint = cfg["endpoint"]
    protocol = cfg["protocol"]

    body = dict(body)
    body["model"] = model_id

    # Extract thinking for logging
    thinking = body.get("thinking", {})
    thinking_type = thinking.get("type", "none") if isinstance(thinking, dict) else "none"
    effort = (body.get("effort")
              or (thinking.get("effort") if isinstance(thinking, dict) else None)
              or (body.get("output_config", {}).get("effort") if isinstance(body.get("output_config"), dict) else None)
              or "none")

    _log(f"→ {original_model!r} → {model_id} | {protocol} | stream={body.get('stream', False)} | thinking={thinking_type} | effort={effort}")

    # ── Anthropic pass-through ──────────────────────────────────
    if protocol == "anthropic":
        a_headers = {"x-api-key": API_KEY, "Content-Type": "application/json",
                     "anthropic-version": "2023-06-01"}
        is_stream = body.get("stream", False)

        if not is_stream:
            resp = await _client.post(endpoint, json=body, headers=a_headers)
            if resp.status_code != 200:
                _log(f"  ERROR {resp.status_code}: {resp.text[:300]}")
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             0, 0, 0, success=False, error=f"HTTP {resp.status_code}",
                             protocol=protocol, is_stream=is_stream, thinking=thinking_type, effort=effort)
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            usage = data.get("usage", {})
            req_in = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
            req_out = usage.get("output_tokens") or _get_output_tokens(usage) or 0
            req_cache = _extract_cache_tokens(usage)
            with _token_lock:
                _token_usage[model_id]["input"] += req_in
                _token_usage[model_id]["output"] += req_out
                _token_usage[model_id]["cache"] += req_cache
            _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{req_cache} cache")
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         req_in, req_out, req_cache, success=True,
                         protocol=protocol, is_stream=is_stream, thinking=thinking_type, effort=effort)
            return Response(content=resp.content, media_type="application/json", headers=_forward_headers(resp.headers))

        # Estimate input tokens for Anthropic streaming
        est_input = _estimate_input_tokens(body)
        with _token_lock:
            _token_usage[model_id]["input"] += est_input

        async def anthropic_stream():
            stream_in = None
            stream_out = stream_cache = 0
            _line_buf = ""
            _sent_data = False
            try:
                async with _client.stream("POST", endpoint, json=body, headers=a_headers) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        _log(f"  ERROR {resp.status_code}: {err[:300]}")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     0, 0, 0, success=False, error=f"HTTP {resp.status_code}",
                                     protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)
                        error_payload = {"type": "error", "error": {"type": "api_error",
                                       "message": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='replace')[:200]}"}}
                        yield _sse("error", error_payload)
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
                        _sent_data = True
                        _line_buf += chunk.decode("utf-8", errors="replace")
                        while "\n" in _line_buf:
                            line, _line_buf = _line_buf.split("\n", 1)
                            line = line.strip()
                            if not line.startswith("data:"):
                                continue
                            data_str = line[5:].strip()
                            if data_str == "[DONE]":
                                continue
                            try:
                                event = json.loads(data_str)
                            except Exception:
                                continue
                            etype = event.get("type", "")
                            if etype == "message_start":
                                msg_usage = event.get("message", {}).get("usage", {})
                                stream_in = msg_usage.get("input_tokens")
                                if stream_in is not None:
                                    with _token_lock:
                                        _token_usage[model_id]["input"] -= est_input
                                        _token_usage[model_id]["input"] += stream_in
                                stream_cache = _extract_cache_tokens(msg_usage)
                                if stream_cache:
                                    with _token_lock:
                                        _token_usage[model_id]["cache"] += stream_cache
                            elif etype == "message_delta":
                                usage = event.get("usage", {})
                                stream_out = usage.get("output_tokens", 0)
                # After stream ends, apply final output token count
                if stream_out:
                    with _token_lock:
                        _token_usage[model_id]["output"] += stream_out
            except GeneratorExit:
                raise
            except BaseException as e:
                is_cancelled = isinstance(e, asyncio.CancelledError)
                if not is_cancelled:
                    _log(f"  ERROR stream: {e}")
                if stream_in is None:
                    with _token_lock:
                        _token_usage[model_id]["input"] -= est_input
                if stream_out:
                    with _token_lock:
                        _token_usage[model_id]["output"] += stream_out
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             stream_in if stream_in is not None else est_input, stream_out, stream_cache, success=False,
                             error="cancelled" if is_cancelled else str(e),
                             protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)
                if not is_cancelled and not _sent_data:
                    error_payload = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                    yield _sse("error", error_payload)
                return
            logged_in = stream_in if stream_in is not None else est_input
            if stream_in is not None or stream_out:
                _log(f"  ← {model_id} | +{logged_in} in | +{stream_out} out | +{stream_cache} cache")
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             logged_in, stream_out, stream_cache, success=True,
                             protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)

        return StreamingResponse(anthropic_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})

    # ── OpenAI-protocol ─────────────────────────────────────────
    oai_body = anthropic_to_openai(body, model_id)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    is_stream = oai_body["stream"]

    if not is_stream:
        resp = await _client.post(endpoint, json=oai_body, headers=headers)
        if resp.status_code != 200:
            _log(f"  ERROR {resp.status_code}: {resp.text[:300]}")
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         0, 0, 0, success=False, error=f"HTTP {resp.status_code}",
                         protocol=protocol, is_stream=is_stream, thinking=thinking_type, effort=effort)
            try:
                err_data = resp.json()
                err_msg = err_data.get("error", {})
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", resp.text[:200])
            except Exception:
                err_msg = resp.text[:200]
            anthro_err = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"HTTP {resp.status_code}: {err_msg}"}},
                                    ensure_ascii=False)
            return Response(content=anthro_err, status_code=resp.status_code, media_type="application/json", headers=_forward_headers(resp.headers))
        data = resp.json()
        usage = data.get("usage", {})
        req_in = usage.get("prompt_tokens", 0)
        req_out = _get_output_tokens(usage)
        cache = _extract_cache_tokens(usage)
        with _token_lock:
            _token_usage[model_id]["input"] += req_in
            _token_usage[model_id]["output"] += req_out
            _token_usage[model_id]["cache"] += cache
        _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{cache} cache")
        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                     req_in, req_out, cache, success=True,
                     protocol=protocol, is_stream=False, thinking=thinking_type, effort=effort)
        return Response(content=json.dumps(openai_to_anthropic(data, original_model), ensure_ascii=False),
                        media_type="application/json", headers=_forward_headers(resp.headers))

    # Streaming
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    oai_body["stream_options"] = {"include_usage": True}

    stream_in_est = _estimate_input_tokens(body)
    with _token_lock:
        _token_usage[model_id]["input"] += stream_in_est

    async def stream_gen():
        started = False
        open_blocks = []
        text_block_idx = None
        reasoning_block_idx = None
        tool_block_idx = {}
        next_block_idx = 0
        stream_out_tokens = 0
        actual_usage = None
        finish_reason = None

        try:
            async with _client.stream("POST", endpoint, json=oai_body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    _log(f"  ERROR {resp.status_code}: {err[:300]}")
                    _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                 0, 0, 0, success=False, error=f"HTTP {resp.status_code}",
                                 protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)
                    error_payload = {"type": "error", "error": {"type": "api_error",
                                   "message": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='replace')[:200]}"}}
                    yield _sse("error", error_payload)
                    return

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()

                    if data == "[DONE]":
                        final_in = stream_in_est
                        final_out = stream_out_tokens
                        final_cache = 0
                        final_cache_creation = 0
                        with _token_lock:
                            if actual_usage:
                                final_in = actual_usage.get("prompt_tokens")
                                if final_in is None:
                                    final_in = stream_in_est
                                final_out = _get_output_tokens(actual_usage)
                                if final_out is None:
                                    total = actual_usage.get("total_tokens")
                                    prompt = actual_usage.get("prompt_tokens")
                                    if total is not None and prompt is not None:
                                        final_out = total - prompt
                                if final_out is None:
                                    final_out = stream_out_tokens
                                final_cache = _extract_cache_tokens(actual_usage)
                                final_cache_creation = actual_usage.get("cache_creation_input_tokens", 0) or actual_usage.get("prompt_tokens_details", {}).get("cache_creation", 0)
                                _token_usage[model_id]["input"] -= stream_in_est
                                _token_usage[model_id]["input"] += final_in
                                _token_usage[model_id]["output"] += final_out
                                if final_cache:
                                    _token_usage[model_id]["cache"] += final_cache
                            else:
                                _token_usage[model_id]["output"] += stream_out_tokens
                        if not started:
                            yield _sse("message_start", {"type": "message_start", "message": {
                                "id": msg_id, "type": "message", "role": "assistant", "content": [],
                                "model": original_model, "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": final_in, "output_tokens": 0,
                                          "cache_creation_input_tokens": final_cache_creation,
                                          "cache_read_input_tokens": final_cache}}})
                        started = True
                        for idx in open_blocks:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                        has_tools = bool(tool_block_idx)
                        _stop_reason_map = {"stop": "end_turn", "length": "max_tokens", "content_filter": "content_filter"}
                        stop_reason = _stop_reason_map.get(finish_reason, "end_turn")
                        if has_tools:
                            stop_reason = "tool_use"
                        yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": final_out, "input_tokens": final_in, "cache_read_input_tokens": final_cache}})
                        yield _sse("message_stop", {"type": "message_stop"})
                        log_tag = "" if actual_usage else " (est)"
                        _log(f"  ← {model_id} | +{final_in} in{log_tag} | +{final_out} out{log_tag} | +{final_cache} cache")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     final_in, final_out, final_cache, success=True,
                                     protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)
                        break

                    try:
                        chunk = json.loads(data)
                    except Exception:
                        continue

                    chunk_usage = chunk.get("usage")
                    if chunk_usage and isinstance(chunk_usage, dict):
                        actual_usage = chunk_usage

                    choices = chunk.get("choices", [])
                    if not choices or not isinstance(choices, list):
                        continue
                    first_choice = choices[0] if choices else {}
                    if isinstance(first_choice, dict):
                        if first_choice.get("finish_reason"):
                            finish_reason = first_choice["finish_reason"]
                    delta = first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
                    if not delta or not isinstance(delta, dict):
                        delta = {}

                    if not started:
                        yield _sse("message_start", {"type": "message_start", "message": {
                            "id": msg_id, "type": "message", "role": "assistant", "content": [],
                            "model": original_model, "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": stream_in_est, "output_tokens": 0,
                                      "cache_creation_input_tokens": 0,
                                      "cache_read_input_tokens": 0}}})
                        started = True

                    # Text
                    text = ""
                    c = delta.get("content")
                    if isinstance(c, str):
                        text = c
                    elif isinstance(c, list):
                        text = "".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")

                    if text:
                        if text_block_idx is None:
                            text_block_idx = next_block_idx
                            next_block_idx += 1
                            yield _sse("content_block_start", {"type": "content_block_start", "index": text_block_idx,
                                       "content_block": {"type": "text", "text": ""}})
                            open_blocks.append(text_block_idx)
                        stream_out_tokens += _estimate_tokens(text)
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx,
                                   "delta": {"type": "text_delta", "text": text}})

                    # Reasoning content
                    reasoning = delta.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        if reasoning_block_idx is None:
                            reasoning_block_idx = next_block_idx
                            next_block_idx += 1
                            yield _sse("content_block_start", {"type": "content_block_start", "index": reasoning_block_idx,
                                       "content_block": {"type": "thinking", "thinking": ""}})
                            open_blocks.append(reasoning_block_idx)
                        stream_out_tokens += _estimate_tokens(reasoning)
                        yield _sse("content_block_delta", {"type": "content_block_delta", "index": reasoning_block_idx,
                                   "delta": {"type": "thinking_delta", "thinking": reasoning}})

                    # Tool calls
                    for tc in (delta.get("tool_calls") or []):
                        api_idx = tc.get("index", 0)
                        if api_idx not in tool_block_idx:
                            block_idx = next_block_idx
                            next_block_idx += 1
                            tool_block_idx[api_idx] = block_idx
                            tc_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:8]}")
                            yield _sse("content_block_start", {"type": "content_block_start", "index": block_idx,
                                       "content_block": {"type": "tool_use", "id": tc_id,
                                       "name": tc.get("function", {}).get("name", ""), "input": {}}})
                            open_blocks.append(block_idx)
                        if args := tc.get("function", {}).get("arguments", ""):
                            stream_out_tokens += _estimate_tokens(args)
                            yield _sse("content_block_delta", {"type": "content_block_delta", "index": tool_block_idx[api_idx],
                                       "delta": {"type": "input_json_delta", "partial_json": args}})
        except GeneratorExit:
            raise
        except BaseException as e:
            is_cancelled = isinstance(e, asyncio.CancelledError)
            if not is_cancelled:
                _log(f"  ERROR stream: {e}")
            with _token_lock:
                adj_in = stream_in_est
                adj_out = stream_out_tokens
                adj_cache = 0
                if actual_usage:
                    adj_in = actual_usage.get("prompt_tokens", stream_in_est)
                    if adj_in is None:
                        adj_in = stream_in_est
                    adj_out = _get_output_tokens(actual_usage)
                    if adj_out is None:
                        adj_out = stream_out_tokens
                    adj_cache = _extract_cache_tokens(actual_usage)
                _token_usage[model_id]["input"] -= stream_in_est
                _token_usage[model_id]["input"] += adj_in
                _token_usage[model_id]["output"] += adj_out
                if adj_cache:
                    _token_usage[model_id]["cache"] += adj_cache
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         adj_in, adj_out, adj_cache, success=False,
                         error="cancelled" if is_cancelled else str(e),
                         protocol=protocol, is_stream=True, thinking=thinking_type, effort=effort)
            if not is_cancelled:
                if not started:
                    error_payload = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                    yield _sse("error", error_payload)
                else:
                    for idx in open_blocks:
                        yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                    yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "error"}, "usage": {"output_tokens": stream_out_tokens}})
                    yield _sse("message_stop", {"type": "message_stop"})

    return StreamingResponse(stream_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})


@app.get("/health")
async def health():
    with _db_lock:
        _conn.execute("SELECT 1")
    with _token_lock:
        usage = {model: {"input": d["input"], "output": d["output"], "cache": d["cache"]}
                 for model, d in _token_usage.items()}
    return {"status": "ok", "usage": usage}


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request):
    try:
        body = json.loads(await request.body())
    except Exception:
        return Response(content='{"error":"invalid json"}', status_code=400)
    tokens = _estimate_input_tokens(body)
    return {"input_tokens": tokens}


if __name__ == "__main__":
    import threading as th
    from uvicorn import Config, Server

    h = RichLogHandler()
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = [h]
        lg.propagate = False

    config = Config(app, host=HOST, port=PORT, log_level="info", log_config=None)
    server = Server(config)

    thread = th.Thread(target=server.run, daemon=True)
    thread.start()

    if WEB_PORT != PORT:
        web_config = Config(app, host=HOST, port=WEB_PORT, log_level="info", log_config=None)
        web_server = Server(web_config)
        web_thread = th.Thread(target=web_server.run, daemon=True)
        web_thread.start()

    time.sleep(0.5)
    _log(f"🔌 API: http://localhost:{PORT}")
    if WEB_PORT != PORT:
        _log(f"🌐 Web UI: http://localhost:{WEB_PORT}")

    run_terminal_loop(ROUTES, _token_usage, _token_lock)
