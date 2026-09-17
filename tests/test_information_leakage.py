"""Tests for the CWE-201 remediation (sensitive information in sent data).

Veracode flagged `static/index.html:606` (the chart demo page, which collected an
OpenAI API key in a form field and put it in the body of an outbound request) and
`data360/server.py:347` (the viz-spec endpoint, which returned the backend's raw
error text to the caller). Both are about what leaves the process, so the tests
assert on what a client receives or sends.
"""

from __future__ import annotations

import pathlib
import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_PAGE = REPO_ROOT / "static" / "index.html"

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


class TestInternalSpecFieldsAreNotSent:
    """Internal helper columns must not ship unless the spec needs them."""

    @pytest.fixture
    def strip(self):
        from data360.server import _strip_unreferenced_internal_fields

        return _strip_unreferenced_internal_fields

    def test_unreferenced_helper_column_is_removed(self, strip):
        spec = {"data": {"values": [{"year": "2020", "value": 1.0, "_label_y": 1.0}]}}

        result = strip(spec)

        assert result["data"]["values"] == [{"year": "2020", "value": 1.0}]

    def test_referenced_helper_column_is_kept(self, strip):
        """End-label and line-gap charts reference their helper columns in Vega."""
        spec = {
            "data": {"values": [{"year": "2020", "value": 1.0, "_label_y": 1.0}]},
            "layer": [{"transform": [{"calculate": "datum._last._label_y"}]}],
        }

        result = strip(spec)

        assert result["data"]["values"] == [{"year": "2020", "value": 1.0, "_label_y": 1.0}]

    def test_public_fields_are_untouched(self, strip):
        spec = {"data": {"values": [{"year": "2020", "value": 1.0, "country": "Kenya"}]}}

        assert strip(spec) == spec

    def test_spec_without_data_is_returned_as_is(self, strip):
        assert strip({"mark": "bar"}) == {"mark": "bar"}

    def test_endpoint_response_carries_no_internal_helper_column(self, client):
        spec = {
            "data": {"values": [{"year": "2020", "value": 1.0, "_label_y": 1.0}]},
            "mark": "line",
        }
        with patch(
            "data360.visualization.get_viz_spec",
            new_callable=AsyncMock,
            return_value={"spec": spec, "reason": "one series", "strategy": "temporal_single"},
        ):
            response = client.post(
                "/api/viz-spec",
                json={"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["spec"]["data"]["values"] == [{"year": "2020", "value": 1.0}]
        assert "_label_y" not in response.text
        # The UI-facing fields the demo page renders stay.
        assert body["strategy"] == "temporal_single"


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
