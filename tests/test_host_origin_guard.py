"""Regression guard for #145: a proxied Host header must not get 421.

FastMCP 3.4.3 (PR #123) enabled Host/Origin validation with empty allow-lists, so every
request whose ``Host`` was not loopback — public domain, pod IP, proxy service name — was
rejected with ``421 Misdirected Request`` on ``/mcp``, ``/mcp/health`` and ``/mcp/ready``.
FastMCP 2.x accepted any Host.

Note: a bare ``TestClient`` request can never show the bug. Starlette derives
``scope["server"]`` from the client base URL and FastMCP implicitly allow-lists that value,
so the explicit external ``Host`` header below is what makes these tests meaningful.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

EXTERNAL_HOST = "data360-mcp.worldbank.org"

_INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "probe", "version": "0"},
    },
}


@pytest.fixture
def client():
    from data360.server import app

    # Context manager runs the ASGI lifespan, which starts the streamable-HTTP session
    # manager that POST /mcp needs.
    with TestClient(app) as test_client:
        yield test_client


def test_health_accepts_external_host(client) -> None:
    """Deployment probes must not depend on the Host the proxy forwards."""
    response = client.get("/mcp/health", headers={"host": EXTERNAL_HOST})
    assert response.status_code == 200, response.text


def test_mcp_accepts_external_host_and_foreign_origin(client) -> None:
    """The MCP endpoint itself, plus web clients that send an unrelated Origin."""
    response = client.post(
        "/mcp",
        headers={
            "host": EXTERNAL_HOST,
            "origin": "https://claude.ai",
            "accept": "application/json, text/event-stream",
        },
        json=_INITIALIZE,
    )
    assert response.status_code == 200, response.text


def test_narrow_allow_list_still_blocks_unknown_host() -> None:
    """The guard is widened by default, not deleted — a configured list must bind."""
    from data360.mcp_server import mcp

    narrow_app = mcp.http_app(path="/mcp", allowed_hosts=[EXTERNAL_HOST])
    with TestClient(narrow_app) as narrow:
        assert narrow.get("/mcp", headers={"host": "rogue.example"}).status_code == 421
        assert narrow.get("/mcp", headers={"host": EXTERNAL_HOST}).status_code != 421


def test_widening_lives_where_every_entry_point_imports_it() -> None:
    """``python -m data360.mcp_server`` builds its own app and never imports data360.server.

    Runs in a subprocess: in this session other tests may have already imported
    ``data360.server``, which would mask the widening sitting in the wrong module.
    """
    probe = (
        "import data360.mcp_server, fastmcp; "
        "print(fastmcp.settings.http_allowed_hosts, "
        "fastmcp.settings.http_allowed_origins)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "['*'] ['*']"
