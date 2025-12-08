from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, List


class _RingBufferHandler(logging.Handler):
    def __init__(self, max_entries: int = 400) -> None:
        super().__init__()
        self._buffer: Deque[str] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - log plumbing
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001 - best effort logging
            message = record.getMessage()
        with self._lock:
            self._buffer.append(message)

    def dump(self, limit: int | None = None) -> List[str]:
        with self._lock:
            if limit is None or limit >= len(self._buffer):
                return list(self._buffer)
            return list(self._buffer)[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


_BUFFER_HANDLER: _RingBufferHandler | None = None


def install_log_buffer(max_entries: int = 400) -> None:
    """Attach the in-memory buffer to the root logger once."""
    global _BUFFER_HANDLER
    if _BUFFER_HANDLER is not None:
        return

    handler = _RingBufferHandler(max_entries=max_entries)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s - %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(handler)
    _BUFFER_HANDLER = handler


def get_recent_logs(limit: int = 200) -> List[str]:
    if _BUFFER_HANDLER is None:
        return []
    return _BUFFER_HANDLER.dump(limit)


def clear_logs() -> None:
    if _BUFFER_HANDLER is None:
        return
    _BUFFER_HANDLER.clear()
