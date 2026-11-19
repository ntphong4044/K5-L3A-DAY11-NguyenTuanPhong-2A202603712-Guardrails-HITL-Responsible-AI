"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}  # request_id -> perf_counter start
        self._pending: dict[str, dict] = {}  # request_id -> partial entry

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None) -> str:
        """Store input + start time; return the request_id to pass to record_output."""
        request_id = request_id or uuid.uuid4().hex[:12]
        self._open[request_id] = time.perf_counter()
        self._pending[request_id] = {
            "request_id": request_id,
            "user_id": user_id,
            "timestamp": utc_now_iso(),
            "input": text,
        }
        return request_id

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append the full entry to self.logs."""
        start = self._open.pop(request_id, None) if request_id else None
        entry = self._pending.pop(request_id, None) if request_id else None
        if entry is None:
            # Output without a matching input (should not happen) — still log it.
            entry = {
                "request_id": request_id,
                "user_id": user_id,
                "timestamp": utc_now_iso(),
                "input": None,
            }
        entry.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "latency_ms": (
                    round((time.perf_counter() - start) * 1000, 1)
                    if start is not None
                    else None
                ),
            }
        )
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None) -> str:
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
