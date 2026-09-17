"""Regression tests for CWE-80 XSS hardening of HTML templates."""

import json

import pytest

from data360.mcp_server.resources import data360_choice_html, vega_lite_renderer
from data360.mcp_server.tools import data360_chart_html
from data360.templates.render import render_template


class TestJinjaAutoescape:
    def test_server_base_is_html_escaped(self):
        html = render_template(
            "vega_lite_renderer.jinja2",
            server_base='"><script>alert(1)</script>',
            pre_loaded_spec=None,
        )
        assert '"><script>alert(1)</script>' not in html
        assert "&lt;script&gt;" in html

    def test_rendered_templates_contain_no_innerhtml(self):
        for html in (
            render_template(
                "vega_lite_renderer.jinja2",
                server_base="http://localhost:8021",
                pre_loaded_spec=None,
            ),
            render_template("data360_choice.jinja2"),
        ):
            assert "innerHTML" not in html


class TestPreloadedSpec:
    @pytest.mark.asyncio
    async def test_script_breakout_payload_is_rejected(self):
        html = await vega_lite_renderer('</script><script>alert(1)</script>')
        assert "window.PRE_LOADED_SPEC =" not in html
        assert "</script><script>alert(1)" not in html

    @pytest.mark.asyncio
    async def test_non_object_json_is_rejected(self):
        html = await vega_lite_renderer('[1, 2, 3]')
        assert "window.PRE_LOADED_SPEC =" not in html

    @pytest.mark.asyncio
    async def test_valid_spec_embedded_as_json(self):
        spec = {"mark": "bar</script><script>alert(1)</script>"}
        html = await vega_lite_renderer(json.dumps(spec))
        assert "window.PRE_LOADED_SPEC" in html
        # tojson escapes angle brackets so the string cannot break out
        assert "</script><script>" not in html
        assert "\\u003c/script\\u003e" in html

    def test_chart_template_has_no_innerhtml(self):
        html = data360_chart_html()
        assert "innerHTML" not in html
        assert "showChartError" in html

    @pytest.mark.asyncio
    async def test_choice_template_has_no_innerhtml(self):
        html = await data360_choice_html()
        assert "innerHTML" not in html
