"""
Claude Code Proxy → opencode.ai
Convert Anthropic /v1/messages ↔ OpenAI chat/completions
"""

import json
import uuid
import time
import logging
import os
import sqlite3
import threading
from contextlib import asynccontextmanager
import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

from config import API_KEY, PROXY, MODELS, ROUTES, get_model_config, HOST, PORT, WEB_PORT

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
        error TEXT
    )
""")
_conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON requests(timestamp)")


def _save_request(req_id, model, original_model, duration_ms,
                  tokens_input, tokens_output, tokens_cache, success=True, error=None):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    with _db_lock:
        _conn.execute("""
            INSERT OR REPLACE INTO requests (id, timestamp, model, original_model, duration_ms,
                tokens_input, tokens_output, tokens_cache, success, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (req_id, timestamp, model, original_model, duration_ms,
              tokens_input, tokens_output, tokens_cache, 1 if success else 0, error))
        _conn.commit()


# Token usage tracking (in-memory, lost on restart)
_token_usage = {model: {"input": 0, "output": 0, "cache": 0} for model in MODELS}
_token_lock = threading.Lock()

# Shared HTTP client (reused across requests)
_transport = httpx.AsyncHTTPTransport(proxy=PROXY) if PROXY else None
_client = httpx.AsyncClient(transport=_transport, timeout=300)


@asynccontextmanager
async def lifespan(app):
    yield
    await _client.aclose()

app = FastAPI(lifespan=lifespan)
register_dashboard(app, STATIC_DIR, _conn, _db_lock)


def _sse(event: str, payload: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


def _route_for(model_name: str) -> dict:
    name = model_name.lower()
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
                else:
                    parts.append(i.get("text", str(i)))
        return "\n".join(parts)
    return str(content) if content else ""


def anthropic_to_openai(body: dict, model: str) -> dict:
    thinking = isinstance(body.get("thinking"), dict) and body["thinking"].get("type") in ("enabled", "adaptive")

    messages = []

    # System prompt
    if system_text := _extract_text(body.get("system", "")):
        messages.append({"role": "system", "content": system_text})

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

        text_parts, tool_calls, thinking_parts, tool_results = [], [], [], []

        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
                continue
            if not isinstance(block, dict):
                continue

            btype = block.get("type")
            if btype == "text":
                text_parts.append(block.get("text", ""))
            elif btype == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif btype == "tool_use":
                tool_calls.append({
                    "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
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

        # Then emit the main message (text + tool_calls + thinking)
        joined_thinking = "\n".join(thinking_parts) if thinking_parts else ""
        if tool_calls:
            out = {
                "role": role,
                "content": "\n".join(text_parts) if text_parts else "",
                "tool_calls": tool_calls,
            }
            if joined_thinking:
                out["reasoning_content"] = joined_thinking
            elif thinking and is_asst:
                out["reasoning_content"] = " "
            messages.append(out)
        elif text_parts or thinking_parts or (thinking and is_asst):
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
    if msg.get("content"):
        blocks.append({"type": "text", "text": msg["content"]})
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

    stop = "tool_use" if msg.get("tool_calls") else "end_turn"
    if choice.get("finish_reason") == "length":
        stop = "max_tokens"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant",
        "content": blocks, "model": model, "stop_reason": stop, "stop_sequence": None,
        "usage": {"input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "cache_creation_input_tokens": 0,
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
                    else:
                        chunks.append(block.get("text", ""))
                        chunks.append(str(block.get("input", "")))

    combined = "\n".join(chunks)
    if _encoding:
        return len(_encoding.encode(combined))
    return max(1, len(combined) // 3)


def _extract_cache_tokens(usage: dict) -> int:
    details = usage.get("prompt_tokens_details") or {}
    if "cached_tokens" in details:
        return details["cached_tokens"]
    if "cached_tokens" in usage:
        return usage["cached_tokens"]
    if "cache_read_input_tokens" in usage:
        return usage["cache_read_input_tokens"]
    return 0


def _elapsed_ms(start_time: float) -> int:
    return int((time.time() - start_time) * 1000)


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
                             0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
                return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
            data = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
            usage = data.get("usage", {})
            req_in = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            req_out = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)
            req_cache = usage.get("cache_read_input_tokens", 0)
            with _token_lock:
                _token_usage[model_id]["input"] += req_in
                _token_usage[model_id]["output"] += req_out
                _token_usage[model_id]["cache"] += req_cache
            _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{req_cache} cache")
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         req_in, req_out, req_cache, success=True)
            return Response(content=resp.content, media_type="application/json")

        # Estimate input tokens for Anthropic streaming
        est_input = _estimate_input_tokens(body)
        with _token_lock:
            _token_usage[model_id]["input"] += est_input

        async def anthropic_stream():
            stream_in = None
            stream_out = stream_cache = 0
            _line_buf = ""
            try:
                async with _client.stream("POST", endpoint, json=body, headers=a_headers) as resp:
                    if resp.status_code != 200:
                        err = await resp.aread()
                        _log(f"  ERROR {resp.status_code}: {err[:300]}")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
                        error_payload = {"type": "error", "error": {"type": "api_error",
                                       "message": f"HTTP {resp.status_code}: {err.decode('utf-8', errors='replace')[:200]}"}}
                        yield _sse("error", error_payload)
                        return
                    async for chunk in resp.aiter_bytes():
                        yield chunk
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
                                usage = event.get("message", {}).get("usage", {})
                                stream_in = usage.get("input_tokens")
                                if stream_in is not None:
                                    with _token_lock:
                                        _token_usage[model_id]["input"] -= est_input
                                        _token_usage[model_id]["input"] += stream_in
                                stream_cache = usage.get("cache_read_input_tokens", 0)
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
            except Exception as e:
                _log(f"  ERROR stream: {e}")
                if stream_in is None:
                    with _token_lock:
                        _token_usage[model_id]["input"] -= est_input
                if stream_out:
                    with _token_lock:
                        _token_usage[model_id]["output"] += stream_out
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             stream_in if stream_in is not None else est_input, stream_out, stream_cache, success=False, error=str(e))
                return
            logged_in = stream_in if stream_in is not None else est_input
            if stream_in is not None or stream_out:
                _log(f"  ← {model_id} | +{logged_in} in | +{stream_out} out | +{stream_cache} cache")
                _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                             logged_in, stream_out, stream_cache, success=True)

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
                         0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
            try:
                err_data = resp.json()
                err_msg = err_data.get("error", {})
                if isinstance(err_msg, dict):
                    err_msg = err_msg.get("message", resp.text[:200])
            except Exception:
                err_msg = resp.text[:200]
            anthro_err = json.dumps({"type": "error", "error": {"type": "api_error", "message": f"HTTP {resp.status_code}: {err_msg}"}},
                                    ensure_ascii=False)
            return Response(content=anthro_err, status_code=resp.status_code, media_type="application/json")
        data = resp.json()
        usage = data.get("usage", {})
        req_in = usage.get("prompt_tokens", 0)
        req_out = usage.get("completion_tokens", 0)
        cache = _extract_cache_tokens(usage)
        with _token_lock:
            _token_usage[model_id]["input"] += req_in
            _token_usage[model_id]["output"] += req_out
            _token_usage[model_id]["cache"] += cache
        _log(f"  ← {model_id} | +{req_in} in | +{req_out} out | +{cache} cache")
        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                     req_in, req_out, cache, success=True)
        return Response(content=json.dumps(openai_to_anthropic(data, original_model), ensure_ascii=False),
                        media_type="application/json")

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

        try:
            async with _client.stream("POST", endpoint, json=oai_body, headers=headers) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    _log(f"  ERROR {resp.status_code}: {err[:300]}")
                    _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                 0, 0, 0, success=False, error=f"HTTP {resp.status_code}")
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
                        with _token_lock:
                            if actual_usage:
                                final_in = actual_usage.get("prompt_tokens")
                                if final_in is None:
                                    final_in = stream_in_est
                                final_out = actual_usage.get("completion_tokens")
                                if final_out is None:
                                    total = actual_usage.get("total_tokens")
                                    prompt = actual_usage.get("prompt_tokens")
                                    if total is not None and prompt is not None:
                                        final_out = total - prompt
                                if final_out is None:
                                    final_out = stream_out_tokens
                                final_cache = _extract_cache_tokens(actual_usage)
                                _token_usage[model_id]["input"] -= stream_in_est
                                _token_usage[model_id]["input"] += final_in
                                _token_usage[model_id]["output"] += final_out
                                if final_cache:
                                    _token_usage[model_id]["cache"] += final_cache
                            else:
                                _token_usage[model_id]["output"] += stream_out_tokens
                        if not started:
                            started = True
                            yield _sse("message_start", {"type": "message_start", "message": {
                                "id": msg_id, "type": "message", "role": "assistant", "content": [],
                                "model": original_model, "stop_reason": None, "stop_sequence": None,
                                "usage": {"input_tokens": 0, "output_tokens": 0}}})
                        for idx in open_blocks:
                            yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                        has_tools = bool(tool_block_idx)
                        yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "tool_use" if has_tools else "end_turn"}, "usage": {"output_tokens": 0}})
                        yield _sse("message_stop", {"type": "message_stop"})
                        log_tag = "" if actual_usage else " (est)"
                        _log(f"  ← {model_id} | +{final_in} in{log_tag} | +{final_out} out{log_tag} | +{final_cache} cache")
                        _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                                     final_in, final_out, final_cache, success=True)
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
                    delta = first_choice.get("delta", {}) if isinstance(first_choice, dict) else {}
                    if not delta or not isinstance(delta, dict):
                        delta = {}

                    if not started:
                        started = True
                        yield _sse("message_start", {"type": "message_start", "message": {
                            "id": msg_id, "type": "message", "role": "assistant", "content": [],
                            "model": original_model, "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}}})

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
        except Exception as e:
            _log(f"  ERROR stream: {e}")
            with _token_lock:
                _token_usage[model_id]["input"] -= stream_in_est
            _save_request(req_id, model_id, original_model, _elapsed_ms(start_time),
                         stream_in_est, stream_out_tokens, 0, success=False, error=str(e))
            if started:
                for idx in open_blocks:
                    yield _sse("content_block_stop", {"type": "content_block_stop", "index": idx})
                yield _sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "error"}, "usage": {"output_tokens": 0}})
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
