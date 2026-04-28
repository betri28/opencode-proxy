# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the proxy server
python opencode.py
```

Server starts on:
- **API**: http://localhost:4000
- **Web Dashboard**: http://localhost:8082

## Configuration

Configuration is managed through `.env` file (create from `.env.example`):

```bash
cp .env.example .env
```

Key environment variables:
- `OPENCODE_PROXY` - Proxy server URL
- `OPENCODE_API_KEY` - API key for OpenCode service
- `OPUS_MAP_MODEL` - Model for opus route (default: `kimi-k2.6`)
- `SONNET_MAP_MODEL` - Model for sonnet route (default: `glm-5.1`)
- `HAIKU_MAP_MODEL` - Model for haiku route (default: `minimax-m2.5`)

Optional server overrides:
- `OPENCODE_HOST` - Bind address (default: `0.0.0.0`)
- `OPENCODE_PORT` - API port (default: `4000`)
- `OPENCODE_WEB_PORT` - Web UI port (default: `8082`)

## Architecture

### Entry Point
- `opencode.py` - Main FastAPI server with request routing and token tracking

### Configuration Module (`config/`)
- `__init__.py` - Exports: `API_BASE_OPENAI`, `API_BASE_ANTHROPIC`, `HOST`, `PORT`, `WEB_PORT`, `MODELS`, `ROUTES`, `get_model_config`, `API_KEY`, `PROXY`
- `settings.py` - Loads environment variables from `.env`, defines:
  - `PROXY` and `API_KEY` (secrets)
  - `MODELS` (model endpoints and protocols)
  - `ROUTES` (model name mappings, built from env vars)
  - `get_model_config()` - Returns merged config for a model

### Dashboard Module (`dashboard/`)
- `__init__.py` - Exports: `register_dashboard`, `log`, `RichLogHandler`, `build_display`, `start_input_thread`
- `api.py` - Dashboard API endpoints: stats, logs, history, static file serving
- `display.py` - Rich terminal display: token usage table, log panel, keyboard input (j/k/g/G/arrows)

### Request Flow
1. Request comes to `/v1/messages` or `/anthropic/v1/messages`
2. `_route_for()` maps model name to route config using `ROUTES`
3. `get_model_config()` gets endpoint and protocol for the model
4. Request is forwarded to the appropriate endpoint (OpenAI or Anthropic protocol)
5. Response is converted between Anthropic and OpenAI formats
6. `/v1/messages/count_tokens` estimates token count without forwarding

### Token Tracking
- Non-streaming: Uses actual `usage` from API response
- Streaming: Estimates input tokens, reads actual usage from SSE events when available
- Stored in SQLite database (`logs/requests.db`)
- Terminal display via Rich (`dashboard/display.py`)
- Web dashboard via API endpoints (`dashboard/api.py`)

### Web Dashboard
- Static files in `static/` directory
- Token usage stats and request history via API endpoints
