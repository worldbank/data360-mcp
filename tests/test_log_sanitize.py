"""Unit tests for centralized log sanitization (CWE-117)."""

import logging

from data360.log_sanitize import (
    CrlfSanitizingFilter,
    install_crlf_sanitizer,
    sanitize_for_log,
)


class TestSanitizeForLog:
    def test_strips_crlf_to_single_line(self):
        out = sanitize_for_log("line1\r\nline2\nline3\rlin4")
        assert "\r" not in out and "\n" not in out
        assert out == "line1  line2 line3 lin4"

    def test_truncates_long_input(self):
        out = sanitize_for_log("x" * 3000)
        assert len(out) <= 2000 + len("...[truncated]")
        assert out.endswith("...[truncated]")

    def test_non_string_uses_repr(self):
        assert sanitize_for_log({"a": 1}) == "{'a': 1}"

    def test_clean_input_unchanged(self):
        assert sanitize_for_log("WB_WDI_NY_GDP_PCAP_KD") == "WB_WDI_NY_GDP_PCAP_KD"


class TestCrlfSanitizingFilter:
    def test_filter_rewrites_record(self):
        record = logging.LogRecord(
            "test", logging.INFO, __file__, 1,
            "user=%s", ("evil\r\nINJECTED: x",), None,
        )
        assert CrlfSanitizingFilter().filter(record) is True
        assert record.getMessage() == "user=evil  INJECTED: x"


class TestInstallCrlfSanitizer:
    def _make_record(self, msg, args):
        logger = logging.getLogger("test_crlf_sanitizer")
        return logger.makeRecord(
            logger.name, logging.INFO, __file__, 1, msg, args, None
        )

    def test_records_sanitized_at_creation(self):
        install_crlf_sanitizer()
        record = self._make_record("q=%s", ("a\r\nb",))
        assert record.getMessage() == "q=a  b"

    def test_install_is_idempotent(self):
        install_crlf_sanitizer()
        install_crlf_sanitizer()
        record = self._make_record("ok", ())
        assert record.getMessage() == "ok"
