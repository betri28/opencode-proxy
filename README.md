# OpenCode-Proxy

A python proxy that lets you use [OpenCode Go](https://opencode.ai/docs/go/) subscription with [Claude Code](https://docs.anthropic.com/en/docs/claude-code).

<img width="449" height="319" alt="image" src="https://github.com/user-attachments/assets/9337be52-4681-414c-bb2b-ad624f079659" />

## Requirements

- Python 3.11+

## Installation

```bash
pip install -r requirements.txt
```

## Docker

Build and run with Docker Compose:

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t opencode-proxy .
docker run --rm -p 4000:4000 -p 8082:8082 --env-file .env -v ${PWD}/logs:/app/logs opencode-proxy
```

The container exposes both the API on `4000` and the dashboard on `8082`, and persists the SQLite history database under `logs/`.

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

## Key Features

- **Extended Thinking & Effort** — Forwards `thinking` and `budget_tokens` parameters as-is to the upstream model. Also supports `effort` (`low` / `medium` / `high`) for models that use effort-based reasoning. Availability depends on the model.

- **Image & Document** — Passes image, PDF, DOCX, and XLSX content blocks through to the upstream model. Whether these are actually processed depends on model support.

- **Web Search** — Forwards web search tool calls to models that natively support it. Results are translated back to Anthropic's `tool_result` format for Claude Code compatibility.

---

## Web Dashboard

The web interface has 2 tabs:

### Token Stats
- Overview: Input, Output, Cache, Success, Failed, Avg Duration
- Charts: Token Distribution, Token % by Model, Requests % by Model
- Detailed table by model
<img width="1172" height="784" alt="image" src="https://github.com/user-attachments/assets/3e29adfe-14c8-4d71-a158-a7cbe0e65174" />


### Request History
- Request logs with full info: time, model, duration, tokens, status
- Filter by time: Today, 7 Days, 30 Days, Custom
- Delete history: all or by date
<img width="1196" height="588" alt="image" src="https://github.com/user-attachments/assets/5e5c2118-79c3-4504-96d9-b14043bfe537" />


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
- `MODELS`: Models and endpoints list, including per-model capability flags (`supports_vision`, `supports_pdf`, `supports_office`, `web_search_tool`, `thinking_style`)

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

## License

[MIT](LICENSE)
