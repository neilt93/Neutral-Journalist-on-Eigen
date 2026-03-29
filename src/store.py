"""
Article store — persists generated articles and pipeline events to disk.

Simple JSON-file store. No database dependency.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def _default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")


class ArticleStore:
    def __init__(self, path: str = "data/articles.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write([])

    def _read(self) -> list[dict]:
        with _LOCK:
            return json.loads(self.path.read_text())

    def _write(self, data: list[dict]) -> None:
        with _LOCK:
            self.path.write_text(
                json.dumps(data, default=_default_serializer, indent=2)
            )

    def add(self, article: dict) -> None:
        articles = self._read()
        article["stored_at"] = datetime.now(timezone.utc).isoformat()
        articles.insert(0, article)  # newest first
        self._write(articles)

    def list_all(self, limit: int = 50) -> list[dict]:
        return self._read()[:limit]

    def count(self) -> int:
        return len(self._read())


class PipelineLog:
    """Ring buffer of pipeline events for the live activity feed."""

    def __init__(self, max_events: int = 200):
        self._events: list[dict] = []
        self._max = max_events
        self._lock = threading.Lock()

    def log(self, phase: str, message: str, **kwargs: Any) -> None:
        event = {
            "time": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "message": message,
            **kwargs,
        }
        with self._lock:
            self._events.append(event)
            if len(self._events) > self._max:
                self._events = self._events[-self._max:]

    def recent(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(reversed(self._events[-limit:]))
