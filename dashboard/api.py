"""
Dashboard API endpoints: stats, logs, history, static files.
"""

import os
from fastapi import Request, Response
from fastapi.staticfiles import StaticFiles

from .display import log_lines


def _build_where(from_date=None, to_date=None):
    conditions, params = [], []
    if from_date:
        conditions.append("timestamp >= ?")
        params.append(from_date)
    if to_date:
        conditions.append("timestamp <= ?")
        params.append(to_date + "T23:59:59")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    return where, params


def register_dashboard(app, static_dir, conn, db_lock):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    async def root():
        from fastapi.responses import FileResponse
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/api/stats")
    async def get_stats(from_date: str = None, to_date: str = None):
        where, params = _build_where(from_date, to_date)

        with db_lock:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),"
                "       COALESCE(SUM(tokens_cache), 0), COUNT(*),"
                "       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),"
                "       SUM(CASE WHEN success = 0 OR success IS NULL THEN 1 ELSE 0 END),"
                "       COALESCE(AVG(duration_ms), 0)"
                " FROM requests " + where,
                params,
            ).fetchone()

            totals = {
                "input": row[0], "output": row[1], "cache": row[2],
                "total": row[0] + row[1] + row[2],
                "count": row[3],
                "success_count": row[4], "fail_count": row[5],
                "avg_duration_ms": int(row[6])
            }

            rows = conn.execute(
                "SELECT model, COALESCE(SUM(tokens_input), 0), COALESCE(SUM(tokens_output), 0),"
                "       COALESCE(SUM(tokens_cache), 0), COUNT(*),"
                "       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END),"
                "       SUM(CASE WHEN success = 0 OR success IS NULL THEN 1 ELSE 0 END),"
                "       COALESCE(AVG(duration_ms), 0)"
                " FROM requests " + where +
                " GROUP BY model",
                params,
            ).fetchall()

        sum_total = totals["total"]
        models = {}
        for r in rows:
            t = r[1] + r[2] + r[3]
            models[r[0]] = {
                "input": r[1], "output": r[2], "cache": r[3], "total": t,
                "pct": f"{t/sum_total*100:.1f}%" if sum_total else "0%",
                "count": r[4], "success_count": r[5], "fail_count": r[6],
                "avg_duration_ms": int(r[7])
            }

        return {"models": models, "totals": totals}

    @app.get("/api/logs")
    async def get_logs(limit: int = 100, offset: int = 0):
        lines = list(log_lines)
        return {
            "logs": lines[offset:offset+limit],
            "total": len(lines),
            "has_more": offset + limit < len(lines)
        }

    @app.get("/api/history")
    async def get_history(from_date: str = None, to_date: str = None, limit: int = 20, offset: int = 0):
        where, params = _build_where(from_date, to_date)
        query = "SELECT * FROM requests " + where + " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with db_lock:
            rows = conn.execute(query, params).fetchall()
            count_query = "SELECT COUNT(*) FROM requests " + where
            total_row = conn.execute(count_query, params[:-2]).fetchone()
            total_count = total_row[0] if total_row else 0

        return {
            "logs": [
                {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "model": r["model"],
                    "original_model": r["original_model"],
                    "duration_ms": r["duration_ms"],
                    "tokens_input": r["tokens_input"],
                    "tokens_output": r["tokens_output"],
                    "tokens_cache": r["tokens_cache"],
                    "success": bool(r["success"]),
                    "error": r["error"],
                    "protocol": r["protocol"] if "protocol" in r.keys() else None,
                    "is_stream": bool(r["is_stream"]) if "is_stream" in r.keys() else False,
                    "thinking": r["thinking"] if "thinking" in r.keys() else None,
                    "effort": r["effort"] if "effort" in r.keys() else None,
                }
                for r in rows
            ],
            "total": total_count,
            "page": offset // limit + 1,
            "per_page": limit,
            "has_more": offset + limit < total_count
        }

    @app.delete("/api/history")
    async def delete_history(before: str = None, all: bool = False):
        with db_lock:
            if all:
                conn.execute("DELETE FROM requests")
            elif before:
                conn.execute("DELETE FROM requests WHERE timestamp < ?", (before + "T23:59:59",))
            else:
                return {"error": "Specify 'before' date or 'all=true'"}
            conn.commit()
        return {"status": "deleted"}
