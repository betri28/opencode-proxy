"""
Rich terminal display: token usage table, log panel, keyboard input.
"""

import sys
import time
import collections
import threading
import logging

from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.console import Group
from rich import box

log_lines = collections.deque(maxlen=200)
LOG_VISIBLE = 35
_log_scroll = 0


def log(msg: str):
    global _log_scroll
    if "/api/" in msg or "Uvicorn running" in msg:
        return
    ts = time.strftime("%H:%M:%S")
    log_lines.append(f"[{ts}] {msg}")
    _log_scroll = max(0, len(log_lines) - LOG_VISIBLE)


class RichLogHandler(logging.Handler):
    def emit(self, record):
        msg = record.getMessage()
        if "/api/" in msg or "Uvicorn running" in msg:
            return
        level = record.levelname
        ts = time.strftime("%H:%M:%S")
        log_lines.append(f"[{ts}] [{level}] {msg}")


def build_display(routes, token_usage, token_lock):
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False, expand=False)
    table.add_column("Route", style="bold", width=8)
    table.add_column("Model", style="bold", min_width=14)
    table.add_column("Total", justify="right", min_width=10)
    table.add_column("Input", justify="right", min_width=10)
    table.add_column("Output", justify="right", min_width=10)
    table.add_column("Cache", justify="right", min_width=10)
    table.add_column("%", justify="right", min_width=6)

    with token_lock:
        usage_snapshot = {m: dict(d) for m, d in token_usage.items()}
    _default_usage = {"input": 0, "output": 0, "cache": 0}

    sum_total = 0
    for route_info in routes.values():
        d = usage_snapshot.get(route_info["model"], _default_usage)
        sum_total += d["input"] + d["output"] + d["cache"]

    sum_in = sum_out = sum_cache = 0
    shown = set()
    for route_name, route_info in routes.items():
        model = route_info["model"]
        shown.add(model)
        d = usage_snapshot.get(model, _default_usage)
        total = d["input"] + d["output"] + d["cache"]
        sum_in += d["input"]
        sum_out += d["output"]
        sum_cache += d["cache"]
        pct = f"{total / sum_total * 100:.1f}%" if sum_total else "0%"
        table.add_row(
            route_name, model,
            f"{total:,}", f"{d['input']:,}", f"{d['output']:,}", f"{d['cache']:,}",
            pct,
        )

    for model, d in usage_snapshot.items():
        if model in shown:
            continue
        total = d["input"] + d["output"] + d["cache"]
        if total == 0:
            continue
        sum_in += d["input"]
        sum_out += d["output"]
        sum_cache += d["cache"]
        pct = f"{total / sum_total * 100:.1f}%" if sum_total else "0%"
        table.add_row(
            "-", model,
            f"{total:,}", f"{d['input']:,}", f"{d['output']:,}", f"{d['cache']:,}",
            pct,
        )

    sum_total = sum_in + sum_out + sum_cache
    table.add_row(
        "[bold yellow]ALL[/]", "",
        f"[bold yellow]{sum_total:,}[/]",
        f"[bold yellow]{sum_in:,}[/]",
        f"[bold yellow]{sum_out:,}[/]",
        f"[bold yellow]{sum_cache:,}[/]",
        f"[bold yellow]100%[/]",
    )

    start = max(0, min(_log_scroll, len(log_lines) - LOG_VISIBLE))
    visible = list(log_lines)[start:start + LOG_VISIBLE]
    log_text = "\n".join(visible) if visible else "[dim]waiting for requests...[/]"
    if len(log_lines) > LOG_VISIBLE:
        log_text += f"\n[dim]↑ {start + 1}/{len(log_lines)} logs (scroll with ↑↓ keys)[/]"

    return Group(
        Panel(table, title="[bold green]Token Usage[/]", border_style="green", padding=(0, 1)),
        Panel(log_text, title="[bold]Log[/]", border_style="dim", padding=(0, 1)),
    )


def start_input_thread():
    global _log_scroll
    _running = True

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return lambda: _running

    def _input_thread():
        nonlocal _running
        if sys.platform == "win32":
            import msvcrt
            while _running:
                if msvcrt.kbhit():
                    try:
                        ch = msvcrt.getch()
                        if ch in (b'\xe0', b'\x00'):
                            ch2 = msvcrt.getch()
                            if ch2 == b'H':
                                _log_scroll = max(0, _log_scroll - 1)
                            elif ch2 == b'P':
                                _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + 1)
                            elif ch2 == b'I':
                                _log_scroll = max(0, _log_scroll - LOG_VISIBLE)
                            elif ch2 == b'Q':
                                _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + LOG_VISIBLE)
                            elif ch2 == b'G':
                                _log_scroll = 0
                            elif ch2 == b'O':
                                _log_scroll = max(0, len(log_lines) - LOG_VISIBLE)
                            continue
                        ch = ch.decode("utf-8", errors="ignore")
                        if ch == "\x03":
                            _running = False
                        elif ch == "k":
                            _log_scroll = max(0, _log_scroll - 1)
                        elif ch == "j":
                            _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + 1)
                        elif ch == "G":
                            _log_scroll = max(0, len(log_lines) - LOG_VISIBLE)
                        elif ch == "g":
                            _log_scroll = 0
                    except Exception:
                        pass
                else:
                    time.sleep(0.05)
        else:
            import tty, termios, select
            fd = sys.stdin.fileno()
            old = termios.tcgetattr(fd)
            tty.setraw(fd)
            try:
                while _running:
                    if select.select([sys.stdin], [], [], 0.05)[0]:
                        ch = sys.stdin.read(1)
                        if ch == "\x03":
                            _running = False
                        elif ch == "\x1b":
                            seq = sys.stdin.read(2) if select.select([sys.stdin], [], [], 0.01)[0] else ""
                            if seq == "[A":
                                _log_scroll = max(0, _log_scroll - 1)
                            elif seq == "[B":
                                _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + 1)
                            elif seq == "[5":
                                if sys.stdin.read(1) == "~":
                                    _log_scroll = max(0, _log_scroll - LOG_VISIBLE)
                            elif seq == "[6":
                                if sys.stdin.read(1) == "~":
                                    _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + LOG_VISIBLE)
                            elif seq == "[H":
                                _log_scroll = 0
                            elif seq == "[F":
                                _log_scroll = max(0, len(log_lines) - LOG_VISIBLE)
                        elif ch in ("k",):
                            _log_scroll = max(0, _log_scroll - 1)
                        elif ch in ("j",):
                            _log_scroll = min(max(0, len(log_lines) - LOG_VISIBLE), _log_scroll + 1)
                        elif ch == "G":
                            _log_scroll = max(0, len(log_lines) - LOG_VISIBLE)
                        elif ch == "g":
                            _log_scroll = 0
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)

    threading.Thread(target=_input_thread, daemon=True).start()
    return lambda: _running


def run_terminal_loop(routes, token_usage, token_lock):
    stop = start_input_thread()
    with Live(build_display(routes, token_usage, token_lock), refresh_per_second=1, screen=sys.stdout.isatty()) as live:
        while stop():
            live.update(build_display(routes, token_usage, token_lock))
            time.sleep(1)
