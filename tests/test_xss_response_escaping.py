"""CWE-80 regression tests: untrusted data must never reach a response unescaped.

Two surfaces are covered:

* the HTML MCP Apps resources, where a client-supplied ``spec`` resource
  parameter was interpolated into a ``<script>`` block (``| safe`` on a template
  whose ``.jinja2`` extension was not autoescaped) and ``server_base`` landed in
  an attribute and a JS string;
* the HTTP JSON responses that echo request data back (the security
  middleware's 403 envelope and the viz-spec endpoint's error body).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from fastapi.testclient import TestClient

PAYLOAD = "<img src=x onerror=alert(1)>"
SCRIPT_BREAKOUT = "</script><img src=x onerror=alert(1)>"


@pytest.fixture(scope="module")
def client() -> TestClient:
    # Imported lazily, as the other test modules do: importing the server app (and
    # the ASGI test client) shifts the suite's heap enough to disturb the
    # wall-clock assertions in tests/test_group_lookup_performance.py.
    from fastapi.testclient import TestClient

    from data360.server import app

    return TestClient(app)


class TestVegaLiteRendererResource:
    """`ui://data360/vega-lite-renderer.html{?spec}` — spec comes from the URI."""

    async def test_non_json_spec_is_never_reflected(self):
        from data360.mcp_server.resources import vega_lite_renderer

        html = await vega_lite_renderer(spec=SCRIPT_BREAKOUT)

        assert SCRIPT_BREAKOUT not in html
        assert "PRE_LOADED_SPEC = " not in html  # block omitted entirely

    async def test_json_spec_with_script_breakout_is_escaped(self):
        from data360.mcp_server.resources import vega_lite_renderer

        spec = json.dumps({"mark": "bar", "title": SCRIPT_BREAKOUT})
        html = await vega_lite_renderer(spec=spec)

        assert SCRIPT_BREAKOUT not in html
        assert "\\u003c/script\\u003e" in html
        assert "window.PRE_LOADED_SPEC = " in html

    async def test_legitimate_spec_still_round_trips(self):
        from data360.mcp_server.resources import vega_lite_renderer

        spec = json.dumps({"mark": "line", "data": {"values": [{"x": 1}]}})
        html = await vega_lite_renderer(spec=spec)

        # Parsed and re-serialised as an object, so vegaEmbed receives a spec.
        assert '"mark": "line"' in html
        assert '"values"' in html

    def test_server_base_cannot_break_out_of_markup(self):
        from data360.templates.render import render_template

        hostile = 'x"><img src=x onerror=alert(1)>'
        html = render_template(
            "vega_lite_renderer.jinja2", server_base=hostile, pre_loaded_spec=None
        )

        assert '"' + PAYLOAD not in html
        assert "&#34;&gt;&lt;img src=x onerror=alert(1)&gt;" in html  # attribute
        assert '\\"\\u003e\\u003cimg src=x onerror=alert(1)\\u003e' in html  # JS string

    def test_vendored_scripts_are_still_inlined(self):
        """`| safe` for the bundled Vega libraries must keep working."""
        from data360.templates.render import render_template

        html = render_template(
            "data360_chart.jinja2",
            vega_js="var vega=1;",
            vega_lite_js="var vegaLite=1;",
            vega_embed_js="var vegaEmbed=1;",
            vega_interpreter_js="var vegaInterpreter=1;",
        )

        assert "<script>var vega=1;</script>" in html


class TestReflectedJsonResponses:
    def test_security_violation_message_is_escaped(self, client):
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": PAYLOAD, "arguments": {}},
            },
        )

        assert response.status_code == 403
        assert PAYLOAD not in response.text
        assert "&lt;img src=x onerror=alert(1)&gt;" in response.text

    def test_jsonrpc_id_is_escaped(self, client):
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": PAYLOAD,
                "method": "tools/call",
                "params": {"name": PAYLOAD, "arguments": {}},
            },
        )

        assert response.json()["id"] == "&lt;img src=x onerror=alert(1)&gt;"

    def test_responses_are_marked_non_sniffable(self, client):
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": PAYLOAD, "arguments": {}},
            },
        )

        assert response.status_code == 403
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["content-type"].startswith("application/json")
        assert client.get("/").headers["x-content-type-options"] == "nosniff"
