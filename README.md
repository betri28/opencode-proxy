# OpenCode-Proxy

A python proxy that lets you use [OpenCode Go](https://opencode.ai/docs/go/) subscription with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

Claude Code CLI (Anthropic format)
        │
        ▼
  ┌─────────────────────┐
  │   OpenCode Proxy    │  ← opencode.py (FastAPI, port 4000)
  │                     │
  │  1. Route matching  │  "claude-sonnet-4" → "glm-5.1"
  │  2. Format convert  │  Anthropic ↔ OpenAI
  │  3. Forward request │  → opencode.ai
  │  4. Convert back    │  OpenAI → Anthropic
  │  5. Track tokens    │  → SQLite
  └─────────────────────┘
        │
        ▼
   opencode.ai (backend AI models)

## Requirements

- Python 3.11+

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Edit values in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENCODE_PROXY` | Proxy server | `` |
| `OPENCODE_API_KEY` | API key for OpenCode | `` |
| `OPUS_MAP_MODEL` | Model for opus route | `kimi-k2.6` |
| `SONNET_MAP_MODEL` | Model for sonnet route | `glm-5.1` |
| `HAIKU_MAP_MODEL` | Model for haiku route | `minimax-m2.5` |

## Claude Code Configuration

To use this proxy with Claude Code, add these environment variables to your shell configuration:

```bash
export ANTHROPIC_API_KEY="fake-key"
export ANTHROPIC_AUTH_TOKEN="fake"
export ANTHROPIC_BASE_URL="http://localhost:4000"
```

Or add them to your `.env` file:

```env
ANTHROPIC_API_KEY=fake-key
ANTHROPIC_AUTH_TOKEN=fake
ANTHROPIC_BASE_URL=http://localhost:4000
```

**Note:** The proxy server must be running (`python opencode.py`) before using Claude Code.

## Running

```bash
python opencode.py
```

Server will start:
- **API**: http://localhost:4000
- **Web Dashboard**: http://localhost:8082

## Web Dashboard

The web interface has 2 tabs:

### Token Stats
- Overview: Input, Output, Cache, Success, Failed, Avg Duration
- Charts: Token Distribution, Token % by Model, Requests % by Model
- Detailed table by model
<img width="1187" height="790" alt="image" src="https://github.com/user-attachments/assets/3a8f61c9-9c09-4248-9c6f-fba1d08a566d" />

### Request History
- Request logs with full info: time, model, duration, tokens, status
- Filter by time: Today, 7 Days, 30 Days, Custom
- Delete history: all or by date
<img width="1167" height="865" alt="image" src="https://github.com/user-attachments/assets/24d9f8fb-1029-47cd-a604-35d55be206da" />

### Common Features
- Dark/Light theme (Dark by default)
- Auto-refresh every 5 seconds
- Time filter shared across both tabs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/messages` | Proxy Anthropic format |
| POST | `/anthropic/v1/messages` | Proxy Anthropic format |
| POST | `/v1/messages/count_tokens` | Estimate token count for a request |
| GET | `/health` | Health check |
| GET | `/api/stats` | Token usage stats (supports `from_date`, `to_date`) |
| GET | `/api/logs` | Terminal logs |
| GET | `/api/history` | Request history from DB (supports `from_date`, `to_date`) |
| DELETE | `/api/history` | Delete history (`before` or `all=true`) |

## Advanced Configuration

For other settings, edit `config/settings.py`:

- `PORT`: API port (default 4000)
- `WEB_PORT`: Web UI port (default 8082)
- `MODELS`: Models and endpoints list (hardcoded in code)

## Keyboard shortcuts (Terminal)

- `j`/`↓`: Scroll down log
- `k`/`↑`: Scroll up log
- `g`: Go to top
- `G`: Go to bottom
- `Ctrl+C`: Exit

## Project Structure

```
opencode.py              # Main FastAPI server
config/
  __init__.py            # Package exports
  settings.py            # Configuration (PROXY, API_KEY, MODELS, ROUTES)
dashboard/
  __init__.py            # Package exports
  api.py                 # Dashboard API endpoints (stats, logs, history)
  display.py             # Rich terminal display (token table, log panel)
static/
  index.html             # Dashboard UI
  styles.css             # Styling (dark/light theme)
  app.js                 # JavaScript
requirements.txt         # Python dependencies
.env.example             # Template environment configuration
.env                     # Environment configuration (gitignored)
.gitignore               # Git ignore rules
logs/                    # Runtime directory (auto-created)
  requests.db            # SQLite - request history
```
