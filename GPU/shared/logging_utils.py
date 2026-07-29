"""
Structured JSON logging + trace_id correlation, shared across all 6 Python
services (frame_extractor, OCR, YOLO, sync_engine, correlation, report_generator).

Previously each service logged plain text via `logging.basicConfig(format="%(asctime)s
%(levelname)s %(message)s")` with session_id embedded ad-hoc inside message
strings ("session=%s ..."). Debugging a failed session meant grepping 4+
terminal windows by eyeball for a UUID substring. This module fixes that two ways:

1. JSON output — every log line is one JSON object (timestamp, level, service,
   message, trace_id, ...) so logs from every service can be ingested/filtered
   the same way (and Node's pino logger already emits JSON, so this brings the
   Python side to parity).
2. trace_id via contextvar — call `set_trace_id(session_id)` once at the top of
   a request handler; every subsequent `logger.info(...)` call in that request
   (including ones already in the codebase that just do `logger.info("text %s", x)`)
   automatically gets `trace_id` attached, no need to touch every call site.

Usage in a service's server.py:
    from logging_utils import configure_logging, set_trace_id
    configure_logging("sync_engine")          # replaces logging.basicConfig(...)
    ...
    @app.post("/sync")
    def sync(req: SyncRequest):
        set_trace_id(req.session_id)           # everything logged below this line
        logger.info("Sync starting")            # is now correlated to this session
"""
import json
import logging
import contextvars
import datetime

_trace_id_var = contextvars.ContextVar("trace_id", default=None)


def set_trace_id(value):
    _trace_id_var.set(str(value) if value is not None else None)


def get_trace_id():
    return _trace_id_var.get()


class _TraceIdFilter(logging.Filter):
    def filter(self, record):
        record.trace_id = _trace_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def __init__(self, service_name: str):
        super().__init__()
        self.service_name = service_name

    def format(self, record):
        payload = {
            "timestamp": datetime.datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "trace_id": getattr(record, "trace_id", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(service_name: str, level=logging.INFO):
    """Replaces logging.basicConfig(...) — call once at module import time."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter(service_name))
    handler.addFilter(_TraceIdFilter())
    root.addHandler(handler)
