"""Structured JSON-line logging, one logger per manager (stage-tagged)."""
from __future__ import annotations

import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "stage": getattr(record, "stage", record.name),
            "msg": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root.addHandler(handler)


class StageLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        merged = {"stage": self.extra["stage"]}
        merged.update(extra)
        kwargs["extra"] = merged
        return msg, kwargs


def get_stage_logger(stage: str) -> logging.LoggerAdapter:
    logger = logging.getLogger(stage)
    return StageLoggerAdapter(logger, {"stage": stage})
