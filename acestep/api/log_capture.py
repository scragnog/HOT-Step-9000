"""Log capture utilities for API progress/status endpoints."""

from __future__ import annotations

from typing import Any


class LogBuffer:
    """Thread-safe ring buffer storing recent log lines with cursor-based polling."""

    MAX_LINES = 500

    def __init__(self):
        """Initialize buffer."""
        import threading
        self._lines: list[tuple[int, str]] = []  # (seq_id, text)
        self._next_id: int = 0
        self._lock = threading.Lock()

    def write(self, message: str) -> None:
        """Capture non-empty log lines."""
        msg = message.strip()
        if not msg:
            return
        with self._lock:
            self._lines.append((self._next_id, msg))
            self._next_id += 1
            # Keep ring buffer bounded
            if len(self._lines) > self.MAX_LINES:
                self._lines = self._lines[-self.MAX_LINES:]

    def flush(self) -> None:
        """No-op flush to satisfy file-like API expectations."""
        return None

    def get_lines_after(self, after: int) -> tuple[list[tuple[int, str]], int]:
        """Return all lines with seq_id > after, and the new cursor.

        Args:
            after: Last seq_id the caller has seen. Pass -1 to get all lines.

        Returns:
            Tuple of (lines, new_cursor) where lines is a list of (seq_id, text).
        """
        with self._lock:
            result = [(seq_id, text) for seq_id, text in self._lines if seq_id > after]
            new_cursor = self._next_id - 1 if self._lines else after
        return result, new_cursor


class StderrLogger:
    """Stderr proxy forwarding writes to original stderr and log buffer."""

    def __init__(self, original_stderr: Any, buffer: LogBuffer):
        """Initialize stderr proxy references."""

        self.original_stderr = original_stderr
        self.buffer = buffer

    def write(self, message: str) -> None:
        """Write to terminal stderr and update in-memory buffer."""

        self.original_stderr.write(message)
        self.buffer.write(message)

    def flush(self) -> None:
        """Flush original stderr stream."""

        self.original_stderr.flush()


def install_log_capture(logger_obj: Any, stderr_obj: Any) -> tuple[LogBuffer, StderrLogger]:
    """Install log sink and stderr proxy for API status polling.

    Args:
        logger_obj: Logger exposing ``add`` method (for example loguru logger).
        stderr_obj: Existing stderr stream object.

    Returns:
        Tuple of ``(log_buffer, stderr_proxy)``.
    """

    log_buffer = LogBuffer()
    logger_obj.add(
        lambda msg: log_buffer.write(str(msg)),
        format="{time:HH:mm:ss} | {level} | {message}",
    )
    stderr_proxy = StderrLogger(stderr_obj, log_buffer)
    return log_buffer, stderr_proxy
