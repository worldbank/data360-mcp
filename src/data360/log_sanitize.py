"""Centralized log-output sanitization (CWE-117 log forging defense).

Veracode flags CWE-117 "Improper Output Neutralization for Logs" wherever
untrusted data (request parameters, upstream API payloads, exception text)
reaches a log file. Rather than auditing every ``logger.*`` call site, this
module provides:

- :func:`sanitize_for_log` — strip CR/LF characters and cap length so a
  single log call can never span multiple log lines.
- :func:`install_crlf_sanitizer` — wrap the global :mod:`logging` record
  factory so **every** ``LogRecord`` created process-wide is sanitized at
  construction time, regardless of logger name or handler (console, file,
  Azure App Insights, uvicorn). Called once from
  :func:`data360.config.setup_logging`.
- :class:`CrlfSanitizingFilter` — a :class:`logging.Filter` with the same
  effect for code that prefers attaching a filter to a specific handler.

CR and LF are replaced with a space (not removed) to keep token boundaries
readable, e.g. ``"a\\nb"`` becomes ``"a b"``.
"""

from __future__ import annotations

import logging

MAX_LOG_FIELD_LENGTH = 2000

_CRLF_TABLE = str.maketrans({"\r": " ", "\n": " ", "\x00": " "})


def sanitize_for_log(value: object, max_length: int = MAX_LOG_FIELD_LENGTH) -> str:
    """Return a single-line, length-capped string safe for log output."""
    text = value if isinstance(value, str) else repr(value)
    text = text.translate(_CRLF_TABLE)
    if len(text) > max_length:
        text = text[:max_length] + "...[truncated]"
    return text


def _sanitize_arg(arg: object) -> object:
    return sanitize_for_log(arg) if isinstance(arg, str) else arg


def _sanitize_record_args(
    args: object,
) -> object:
    if args is None:
        return args
    if isinstance(args, dict):
        return {k: _sanitize_arg(v) for k, v in args.items()}
    if isinstance(args, (tuple, list)):
        return tuple(_sanitize_arg(a) for a in args)
    return _sanitize_arg(args)


class CrlfSanitizingFilter(logging.Filter):
    """Logging filter that strips CR/LF from the record message and args."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = sanitize_for_log(record.msg)
            record.args = _sanitize_record_args(record.args)
        except Exception:
            pass
        return True


def install_crlf_sanitizer() -> None:
    """Wrap the global log record factory to sanitize every record at creation.

    Safe to call multiple times — re-wrapping an already wrapped factory is a
    no-op.
    """
    old_factory = logging.getLogRecordFactory()
    if getattr(old_factory, "_data360_crlf_sanitized", False):
        return

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = old_factory(*args, **kwargs)
        try:
            if isinstance(record.msg, str):
                record.msg = sanitize_for_log(record.msg)
            record.args = _sanitize_record_args(record.args)
        except Exception:
            pass
        return record

    factory._data360_crlf_sanitized = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(factory)
