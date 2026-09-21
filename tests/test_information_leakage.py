"""Tests for the CWE-201 remediation (sensitive information in sent data).

Veracode flagged `static/index.html:606` (the chart demo page, which collected an
OpenAI API key in a form field and put it in the body of an outbound request) and
`data360/server.py:347` (the viz-spec endpoint, which returned the backend's raw
error text to the caller). Both are about what leaves the process, so the tests
assert on what a client receives or sends.
"""

from __future__ import annotations

import importlib.util
import pathlib
import re
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_PAGE = REPO_ROOT / "static" / "index.html"
RATE_LIMIT_SCRIPT = REPO_ROOT / "tests" / "test_rate_limits.py"

BACKEND_DETAIL = "Indicator 'WB_SECRET_IDNO' not found: HTTP 404 {'trace': 'backend-internal'}"
GENERIC_MESSAGE = "Visualization could not be generated for this request."


@pytest.fixture(scope="module")
def client() -> TestClient:
    from data360.server import app

    return TestClient(app)


class TestVizSpecErrorIsGeneric:
    """The HTTP endpoint must not relay backend error text to the caller."""

    def test_backend_error_detail_is_not_sent_to_the_caller(self, client):
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"error": BACKEND_DETAIL},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 400
        assert response.json() == {"error": GENERIC_MESSAGE}
        assert "WB_SECRET_IDNO" not in response.text
        assert "backend-internal" not in response.text

    def test_backend_error_is_logged_server_side(self, client, caplog):
        import logging

        with (
            patch(
                "data360.visualization.get_viz_spec",
                new_callable=AsyncMock,
                return_value={"error": BACKEND_DETAIL},
            ),
            caplog.at_level(logging.WARNING, logger="data360.server"),
        ):
            client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert any("WB_SECRET_IDNO" in record.getMessage() for record in caplog.records)

    def test_unexpected_exception_response_is_generic(self, client):
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            side_effect=RuntimeError("internal stack detail"),
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 500
        assert "internal stack detail" not in response.text


class TestVizSpecResponseIsClosed:
    """CWE-201: the success response is a closed contract, not a relay of `res`."""

    def test_unexpected_spec_keys_and_injected_detail_are_not_sent(self, client):
        hostile = {
            "spec": {
                "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                "mark": "line",
                "encoding": {},
                "debug": {"env": "INTERNAL_SECRET"},
                "url": "http://169.254.169.254/latest/meta-data/",
            },
            "strategy": "temporal_single",
            "reason": "Single indicator, 5 years, 3 economies → line chart",
        }
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value=hostile,
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 200
        assert sorted(response.json()) == ["reason", "spec", "strategy"]
        assert "INTERNAL_SECRET" not in response.text
        assert "169.254" not in response.text
        assert "debug" not in response.json()["spec"]
        assert "url" not in response.json()["spec"]

    def test_unknown_strategy_is_coerced_to_a_known_value(self, client):
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"spec": {"mark": "line"}, "strategy": "not-a-strategy", "reason": None},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.json()["strategy"] == "unknown"

    def test_reason_control_characters_are_stripped_and_length_clipped(self, client):
        hostile_reason = "line one\r\nforged: entry" + "A" * 400
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={
                "spec": {"mark": "line"},
                "strategy": "temporal_single",
                "reason": hostile_reason,
            },
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        reason = response.json()["reason"]
        assert "\r" not in reason and "\n" not in reason
        assert len(reason) <= 300

    def test_non_object_spec_is_rejected_generically(self, client):
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"spec": "not-an-object", "strategy": "temporal_single"},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 500
        assert response.json() == {"error": "Vega-Lite spec was not generated."}

    def test_vega_lite_keys_used_by_this_repo_survive_the_allow_list(self):
        from data360.server import _VEGA_LITE_TOP_LEVEL_KEYS

        used = {
            "$schema", "config", "data", "encoding", "height", "layer", "mark",
            "title", "width", "vconcat", "hconcat", "facet", "concat", "resolve",
            "spec",
        }
        assert used <= _VEGA_LITE_TOP_LEVEL_KEYS

    def test_spec_with_only_unknown_keys_is_rejected_generically(self, client):
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"spec": {"debug": 1, "internal": "x"}, "strategy": "temporal_single"},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 500
        assert response.json() == {"error": "Vega-Lite spec was not generated."}

    def test_nested_spec_content_is_relayed_by_design(self, client):
        """Only top-level keys are allow-listed; nested content is the chart payload.

        `data.url` and `data.values` are legitimate Vega-Lite fields, so the filter
        deliberately stops at the top level. This pins that boundary so the
        documented guarantee matches the code.
        """
        nested_spec = {
            "mark": "line",
            "data": {"url": "https://example.org/chart.json", "values": [{"y": 1}]},
        }
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"spec": nested_spec, "strategy": "temporal_single", "reason": "r"},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 200
        assert response.json()["spec"]["data"]["url"] == "https://example.org/chart.json"


class TestRateLimitScriptLogging:
    """CWE-532: the load-test script's log output must not carry the endpoint's secrets."""

    @pytest.fixture(scope="module")
    def script(self):
        # Loaded by path (it is a standalone CLI, not a pytest module); register before
        # module uses dataclasses.
        module_spec = importlib.util.spec_from_file_location(
            "rate_limits_under_test", RATE_LIMIT_SCRIPT
        )
        assert module_spec is not None and module_spec.loader is not None
        module = importlib.util.module_from_spec(module_spec)
        sys.modules["rate_limits_under_test"] = module
        module_spec.loader.exec_module(module)
        return module

    def test_query_secret_is_masked(self, script):
        url = "https://wbplatformmcpqa.azure-api.net/mcp?subscription-key=SECRET123"

        logged = script.redacted_url(url)

        assert "SECRET123" not in logged
        assert logged == "https://wbplatformmcpqa.azure-api.net/mcp?subscription-key=REDACTED"

    def test_userinfo_is_dropped(self, script):
        logged = script.redacted_url("https://user:pw@qa.internal/mcp")

        assert "pw" not in logged
        assert "user" not in logged
        assert logged == "https://qa.internal/mcp"

    def test_all_query_values_are_masked_and_fragment_dropped(self, script):
        logged = script.redacted_url("https://host:8443/mcp?code=abc&x=1#frag")

        assert "abc" not in logged and "#" not in logged
        assert logged == "https://host:8443/mcp?code=REDACTED&x=REDACTED"

    def test_encoded_control_characters_in_a_query_name_are_re_encoded(self, script):
        """parse_qsl decodes %0A into a newline; the logged name must not carry it."""
        logged = script.redacted_url("https://host/mcp?%0Aforged=1&x=2")

        assert "\n" not in logged
        assert "%0Aforged=REDACTED" in logged
        assert logged.endswith("x=REDACTED")

    def test_plain_endpoint_is_kept_for_the_operator(self, script):
        url = "http://localhost:8021/mcp"

        assert script.redacted_url(url) == url

    def test_flagged_line_no_longer_prints_the_raw_url(self, script):
        """The reported print must go through the redactor.

        Source-level on purpose: exercising it otherwise means running a load test
        against a live endpoint. The reported line was
        `print(f"MCP Server URL: {mcp_url}")`.
        """
        source = RATE_LIMIT_SCRIPT.read_text()

        assert "print(f\"MCP Server URL: {mcp_url}\")" not in source
        assert 'print(f"MCP Server URL: {redacted_url(mcp_url)}")' in source
        assert '"mcp_url": redacted_url(mcp_url),' in source


class TestDemoPageDoesNotHandleCredentials:
    """The served page must not collect a credential and send it outbound."""

    def test_no_credential_input_elements(self):
        page = DEMO_PAGE.read_text()

        # An <input type="password"> element (the CSS selector of the same name is fine).
        assert not re.search(r"<input[^>]*password", page), "a credential input was reintroduced"
        for needle in ("api-key", "apiKey", "api_key", "openai", "OpenAI"):
            assert needle not in page, f"{needle!r} is back in the demo page"

    def test_request_bodies_carry_only_chart_inputs(self):
        page = DEMO_PAGE.read_text()

        bodies = re.findall(r"body:\s*JSON\.stringify\(\{(.*?)\}\)", page, re.S)
        assert bodies, "no request bodies found — did the page change shape?"

        fields = {
            field.split(":")[0].strip()
            for body in bodies
            for field in body.split(",")
            if field.strip()
        }
        # The property that matters: no request body carries a credential, whatever
        # else the page chooses to send.
        secretish = ("key", "token", "secret", "password", "credential", "auth", "bearer")
        offenders = sorted(
            field for field in fields if any(word in field.lower() for word in secretish)
        )
        assert not offenders, f"credential-looking request fields: {offenders}"
