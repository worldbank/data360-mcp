"""Live smoke tests: drive a real Data360 MCP server process.

The rest of the suite runs in-process (`TestClient`, mocked HTTP). These tests are
the opposite on purpose: the fixture boots `data360.server:app` as a subprocess on
a local port, waits for readiness and captures the process output, so the CWE-117
checks can assert what the server *actually wrote* to its log.

Selected explicitly — a plain `pytest` run must stay offline and deterministic:

    uv run poe live-smoke
    DATA360_LIVE_SMOKE=1 uv run pytest -m live tests/test_live_smoke.py

Covers: readiness, tools/list, CWE-117 log forging (tool name + data path),
CWE-80 reflected values (blocked-call envelope, renderer resource), CWE-73
malformed country code, and a real search against the Data360 API.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
import time
import urllib.parse
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not os.environ.get("DATA360_LIVE_SMOKE"),
        reason="live test: set DATA360_LIVE_SMOKE=1 or run `uv run poe live-smoke`",
    ),
]

CRLF = "\r\n"
FORGED_ENTRY = "2026-01-01 00:00:00 - data360 - ERROR - admin login succeeded"
XSS_PAYLOAD = "</script><img src=x onerror=alert(1)>"
NOSNIFF = "nosniff"
HTTP_OK = 200
HTTP_FORBIDDEN = 403
DEFAULT_PORT = 8021

RPC_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


class Server:
    """Boot ``uvicorn data360.server:app`` and capture everything it logs."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._lines: list[str] = []
        self._proc: subprocess.Popen[str] | None = None

    def start(self, timeout: float = 90.0) -> None:
        # conftest points every DATA360_* URL at an unreachable test host for the
        # offline suite; the booted server would inherit those and fail to reach the
        # API. Live runs start from a clean DATA360_* slate so the package's own
        # documented defaults (the real endpoints) apply.
        env = {k: v for k, v in os.environ.items() if not k.startswith("DATA360_")}
        env.update(
            {
                "DATA360_API_BASE_URL": os.environ.get(
                    "DATA360_LIVE_API_BASE_URL", "https://data360api.worldbank.org"
                ),
                "MCP_PORT": str(self.port),
                # DEBUG surfaces the data-path sinks the CWE-117 fix touches.
                "MCP_LOG_LEVEL": os.environ.get("MCP_LOG_LEVEL", "DEBUG"),
                "MCP_ENV": os.environ.get("MCP_ENV", "local"),
            }
        )
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "data360.server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        threading.Thread(target=self._drain, daemon=True).start()

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                pytest.fail(
                    "server exited during startup:\n" + "\n".join(self._lines[-15:])
                )
            try:
                if httpx.get(f"{self.base}/mcp/health", timeout=2).status_code == HTTP_OK:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        pytest.fail(f"server not healthy within {timeout:.0f}s")

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            self._lines.append(line.rstrip("\n"))

    def logs(self) -> list[str]:
        return list(self._lines)

    def stop(self) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    port = int(os.environ.get("DATA360_LIVE_SMOKE_PORT", str(DEFAULT_PORT)))
    instance = Server(port)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop()


def rpc(server: Server, body: dict[str, Any], timeout: float = 120.0) -> tuple[int, Any, Any]:
    """POST a JSON-RPC message; unwrap SSE framing when present."""
    response = httpx.post(
        f"{server.base}/mcp", json=body, headers=RPC_HEADERS, timeout=timeout
    )
    text = response.text
    if "data:" in text:
        text = text.split("data:", 1)[1].strip()
    try:
        payload: Any = json.loads(text)
    except ValueError:
        payload = text
    return response.status_code, payload, response.headers


def tool_text(payload: Any) -> str:
    try:
        return payload["result"]["content"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return ""


def wait_for(lines, predicate, timeout: float = 6.0) -> list[str]:
    """Poll the (live) server output until a matching line shows up."""
    deadline = time.monotonic() + timeout
    while True:
        matched = [line for line in lines() if predicate(line)]
        if matched or time.monotonic() >= deadline:
            return matched
        time.sleep(0.15)


def test_health_and_readiness(server: Server) -> None:
    health = httpx.get(f"{server.base}/mcp/health", timeout=30)
    ready = httpx.get(f"{server.base}/mcp/ready", timeout=60).json()

    assert health.status_code == HTTP_OK
    assert health.json()["status"] == "ok"
    assert ready["status"] == "ready", ready


def test_tools_list_exposes_the_registered_tools(server: Server) -> None:
    _, payload, _ = rpc(
        server, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    )

    names = [tool["name"] for tool in payload["result"]["tools"]]
    assert names, payload
    assert "data360_search_indicators" in names


def test_blocked_call_envelope_is_escaped_and_nosniff(server: Server) -> None:
    """CWE-80: request values echoed in the JSON envelope must be escaped."""
    status, payload, headers = rpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": XSS_PAYLOAD,
            "method": "tools/call",
            "params": {"name": XSS_PAYLOAD, "arguments": {}},
        },
    )

    assert status == HTTP_FORBIDDEN
    # The id is echoed verbatim (JSON-RPC correlation); the reflected message is escaped.
    assert payload["id"] == XSS_PAYLOAD
    message = payload["error"]["message"]
    assert XSS_PAYLOAD not in message, message
    assert "&lt;" in message, message
    assert headers.get("x-content-type-options") == NOSNIFF


def test_renderer_resource_keeps_a_hostile_spec_escaped(server: Server) -> None:
    """CWE-80: the spec resource parameter must not break out of <script>."""
    spec = json.dumps({"mark": "bar", "title": XSS_PAYLOAD})
    uri = "ui://data360/vega-lite-renderer.html?spec=" + urllib.parse.quote(spec)

    _, payload, _ = rpc(
        server, {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": uri}}
    )

    html = payload["result"]["contents"][0]["text"]
    assert XSS_PAYLOAD not in html
    assert "\\u003c/script\\u003e" in html


def test_crlf_in_tool_name_stays_on_one_log_line(server: Server) -> None:
    """CWE-117: a CR/LF payload must not forge a second log entry."""
    status, _, _ = rpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "evil" + CRLF + FORGED_ENTRY, "arguments": {}},
        },
    )
    assert status == HTTP_FORBIDDEN

    escaped = wait_for(
        server.logs, lambda line: "Security violation" in line and "\\r\\n" in line
    )
    forged = [line for line in server.logs() if line.startswith(FORGED_ENTRY)]

    assert not forged, f"forged log entry appeared: {forged[0]!r}"
    assert escaped, "no escaped 'Security violation' line in the server output"


def test_crlf_in_indicator_id_stays_on_one_log_line(server: Server) -> None:
    """CWE-117 on the data path: an indicator id with CR/LF reaches a debug sink."""
    rpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "data360_get_data",
                "arguments": {
                    "database_id": "WB_WDI",
                    "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG" + CRLF + FORGED_ENTRY,
                    "limit": 2,
                },
            },
        },
    )

    assert not [
        line for line in server.logs() if line.startswith(FORGED_ENTRY)
    ], "forged log entry appeared on the data path"
    assert wait_for(
        server.logs, lambda line: "Fetching data from" in line and "\\r\\n" in line
    ), "data-path sink did not emit an escaped line (MCP_LOG_LEVEL=DEBUG?)"


def test_malformed_country_code_is_dropped_before_the_request(server: Server) -> None:
    """CWE-73: a traversal-looking country code must not reach the API request."""
    _, payload, _ = rpc(
        server,
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "data360_get_viz_spec",
                "arguments": {
                    "database_id": "WB_WDI",
                    "indicator_id": "WB_WDI_NY_GDP_MKTP_KD_ZG",
                    "country_code": "USA;../../etc/passwd",
                },
            },
        },
        timeout=180,
    )

    body = json.dumps(payload)
    assert "etc/passwd" not in body.upper()
    assert "Traceback" not in body
    text = tool_text(payload)
    assert "Error" not in text[:200], (
        f"request failed instead of dropping the token: {text[:200]}"
    )


def test_live_search_returns_indicators(server: Server) -> None:
    """End-to-end: a real search against the Data360 API still works.

    The API occasionally answers slowly, so one retry keeps this gate stable.
    """
    last: str = ""
    for _attempt in range(2):
        _, payload, _ = rpc(
            server,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "data360_search_indicators",
                    "arguments": {"query": "GDP growth", "limit": 3},
                },
            },
        )
        try:
            data = json.loads(tool_text(payload))
        except ValueError as exc:
            last = f"{type(exc).__name__}: {exc}"
            continue
        indicators = data.get("indicators") or []
        if indicators:
            assert indicators[0].get("idno"), indicators[0]
            return
        last = f"no indicators returned (error={data.get('error')})"

    pytest.fail(last or "search returned nothing")
