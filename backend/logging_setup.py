"""
logging_setup.py — structured JSON logging with request/job/lead correlation.

Call configure_logging() once at process startup (backend.py, worker.py).
Use request_context()/job_context()/lead_context() to bind an ID that then
appears on every log line emitted from any module while inside the block,
without threading extra parameters through function signatures.

Scope: this instruments our own logger.info/warning/exception calls in
backend.py/pipeline.py/worker.py. CrewAI's own verbose=True console output
(agent "thinking" traces) goes through CrewAI's internal printer, not this
logging pipeline, and is out of scope here.
"""

import contextvars
import json
import logging
from contextlib import contextmanager
from typing import Optional

_job_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("job_id", default=None)
_lead_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("lead_id", default=None)
_request_id_var: "contextvars.ContextVar[Optional[str]]" = contextvars.ContextVar("request_id", default=None)


class _CorrelationFilter(logging.Filter):
    """Attaches whichever correlation IDs are active on the current context."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.job_id = _job_id_var.get()
        record.lead_id = _lead_id_var.get()
        record.request_id = _request_id_var.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field in ("job_id", "lead_id", "request_id"):
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Idempotent — safe to call from both backend.py and worker.py in one process."""
    root = logging.getLogger()
    if getattr(root, "_correlation_configured", False):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(_JsonFormatter())
    handler.addFilter(_CorrelationFilter())
    root.handlers = [handler]
    root.setLevel(level)
    root._correlation_configured = True


@contextmanager
def job_context(job_id):
    token = _job_id_var.set(str(job_id) if job_id is not None else None)
    try:
        yield
    finally:
        _job_id_var.reset(token)


@contextmanager
def lead_context(lead_id):
    token = _lead_id_var.set(str(lead_id) if lead_id is not None else None)
    try:
        yield
    finally:
        _lead_id_var.reset(token)


@contextmanager
def request_context(request_id):
    token = _request_id_var.set(str(request_id) if request_id is not None else None)
    try:
        yield
    finally:
        _request_id_var.reset(token)
