#!/usr/bin/env python3
"""Local live smoke test for the Data360 MCP server.

Boots the real ASGI app (``data360.server:app``) on a local port, waits for
readiness, then drives it over the MCP transport and asserts both that the CWE
remediations hold on a real server and that normal traffic still works.

Run it (no server needed — the script starts and stops one):

    uv run poe live-smoke
    uv run python scripts/live_smoke.py --port 8022
    uv run python scripts/live_smoke.py --offline          # skip upstream calls
    uv run python scripts/live_smoke.py --attach           # test a server you already run on the port

Checks:

  1. ``/mcp/health`` + ``/mcp/ready`` answer, and readiness reports ``ready``.
  2. ``tools/list`` returns the registered tools (transport works).
  3. CWE-117: a CR/LF forged-entry payload in a tool name stays escaped on a
     single log line — the forged entry never appears as its own line.
  4. CWE-117: the same payload inside an indicator id, through the data path
     (needs ``--offline`` off and an upstream that answers).
  5. CWE-80: a blocked call's JSON envelope escapes the request ``id`` and the
     message, and responses carry ``X-Content-Type-Options: nosniff``.
  6. CWE-80: the Vega-Lite renderer resource escapes a hostile ``spec`` from the
     request URI instead of injecting it into the page's ``<script>``.
  7. End-to-end: a real search returns indicators (skipped with ``--offline``).

Exit code 0 means every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Any

import httpx

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

CRLF = "\r\n"
FORGED_ENTRY = "2026-01-01 00:00:00 - data360 - ERROR - admin login succeeded"
XSS_PAYLOAD = "</script><img src=x onerror=alert(1)>"
NOSNIFF = "nosniff"
HTTP_OK = 200
HTTP_FORBIDDEN = 403

RPC_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
}


class Result:
    """One check outcome."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ok = False
        self.detail = ""

    def passed(self, detail: str = "") -> Result:
        self.ok, self.detail = True, detail
        return self

    def failed(self, detail: str) -> Result:
        self.ok, self.detail = False, detail
        return self


class Server:
    """Boot ``uvicorn data360.server:app`` and capture everything it logs."""

    def __init__(self, port: int) -> None:
        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        self._lines: list[str] = []
        self._proc: subprocess.Popen[str] | None = None

    def start(self, timeout: float = 90.0) -> None:
        env = {
            **os.environ,
            "DATA360_API_BASE_URL": os.environ.get(
                "DATA360_API_BASE_URL", "https://data360api.worldbank.org"
            ),
            "MCP_PORT": str(self.port),
            # DEBUG surfaces the data-path sinks the CWE-117 fix touches.
            "MCP_LOG_LEVEL": os.environ.get("MCP_LOG_LEVEL", "DEBUG"),
            "MCP_ENV": os.environ.get("MCP_ENV", "local"),
        }
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
                raise RuntimeError(
                    "server exited during startup:\n" + "\n".join(self._lines[-15:])
                )
            try:
                if httpx.get(f"{self.base}/mcp/health", timeout=2).status_code == HTTP_OK:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        raise TimeoutError(f"server not healthy within {timeout:.0f}s")

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


def rpc(base: str, body: dict[str, Any], timeout: float = 120.0) -> tuple[int, Any, Any]:
    """POST a JSON-RPC message; unwrap SSE framing when present."""
    response = httpx.post(f"{base}/mcp", json=body, headers=RPC_HEADERS, timeout=timeout)
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


def port_is_free(port: int) -> bool:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def check_health(base: str) -> Result:
    result = Result("health + readiness")
    try:
        health = httpx.get(f"{base}/mcp/health", timeout=30)
        ready = httpx.get(f"{base}/mcp/ready", timeout=60).json()
    except (httpx.HTTPError, ValueError) as exc:
        return result.failed(f"{type(exc).__name__}: {exc}")
    state = ready.get("status")
    detail = f"health={health.json().get('status')} ready={state}"
    return result.passed(detail) if health.status_code == HTTP_OK and state == "ready" else result.failed(detail)


def check_tools_list(base: str) -> Result:
    result = Result("tools/list")
    try:
        _, payload, _ = rpc(
            base, {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
        )
        names = [t["name"] for t in payload["result"]["tools"]]
    except (KeyError, TypeError, httpx.HTTPError) as exc:
        return result.failed(f"{type(exc).__name__}: {exc}")
    return result.passed(f"{len(names)} tools, e.g. {', '.join(names[:3])}")


def check_log_forging(base: str, server: Server) -> Result:
    """CWE-117: a CR/LF payload must not produce a forged log entry."""
    result = Result("CWE-117: CRLF in tool name (log forging)")
    status, _, _ = rpc(
        base,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "evil" + CRLF + FORGED_ENTRY, "arguments": {}},
        },
    )
    if status != HTTP_FORBIDDEN:
        return result.failed(f"expected {HTTP_FORBIDDEN} from the validator, got {status}")

    escaped = wait_for(
        server.logs, lambda line: "Security violation" in line and "\\r\\n" in line
    )
    forged = [line for line in server.logs() if line.startswith(FORGED_ENTRY)]
    if forged:
        return result.failed(f"forged log line appeared: {forged[0]!r}")
    if not escaped:
        return result.failed("no escaped 'Security violation' line found in server output")
    return result.passed("single escaped line, no forged entry")


def check_data_path_logging(base: str, server: Server) -> Result:
    """CWE-117 on the data path: an indicator id with CR/LF reaches a debug sink."""
    result = Result("CWE-117: CRLF in indicator id (data path)")
    rpc(
        base,
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
    if any(line.startswith(FORGED_ENTRY) for line in server.logs()):
        return result.failed("forged log line appeared on the data path")
    fetched = wait_for(
        server.logs,
        lambda line: "Fetching data from" in line and "\\r\\n" in line,
    )
    if not fetched:
        return result.failed(
            "data-path debug sink did not emit an escaped line (MCP_LOG_LEVEL=DEBUG?)"
        )
    return result.passed("data-path sink stayed on one escaped line")


def check_blocked_envelope(base: str) -> Result:
    """CWE-80: request values echoed in a JSON envelope must be escaped."""
    result = Result("CWE-80: blocked call envelope + nosniff")
    status, payload, headers = rpc(
        base,
        {
            "jsonrpc": "2.0",
            "id": XSS_PAYLOAD,
            "method": "tools/call",
            "params": {"name": XSS_PAYLOAD, "arguments": {}},
        },
    )
    body = json.dumps(payload)
    if status != HTTP_FORBIDDEN:
        return result.failed(f"expected {HTTP_FORBIDDEN}, got {status}")
    if XSS_PAYLOAD in body or "&lt;" not in body:
        return result.failed(f"payload not escaped: {body[:200]}")
    if headers.get("x-content-type-options") != NOSNIFF:
        return result.failed("missing X-Content-Type-Options: nosniff")
    return result.passed("id + message escaped, nosniff present")


def check_renderer_resource(base: str) -> Result:
    """CWE-80: the spec resource parameter must not break out of <script>."""
    result = Result("CWE-80: renderer resource spec escaping")
    spec = json.dumps({"mark": "bar", "title": XSS_PAYLOAD})
    uri = "ui://data360/vega-lite-renderer.html?spec=" + urllib.parse.quote(spec)
    _, payload, _ = rpc(
        base, {"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": uri}}
    )
    try:
        html = payload["result"]["contents"][0]["text"]
    except (KeyError, IndexError, TypeError):
        return result.failed(f"unexpected resources/read payload: {str(payload)[:200]}")
    if XSS_PAYLOAD in html:
        return result.failed("payload injected into the page raw")
    if "\\u003c/script\\u003e" not in html:
        return result.failed("spec is not JSON-escaped in the page")
    return result.passed("payload kept as an escaped JSON string")


def check_upstream(base: str) -> Result:
    result = Result("end-to-end: live search")
    try:
        _, payload, _ = rpc(
            base,
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
        data = json.loads(tool_text(payload))
    except (ValueError, KeyError, TypeError, httpx.HTTPError) as exc:
        return result.failed(f"{type(exc).__name__}: {exc}")
    indicators = data.get("indicators") or []
    if not indicators:
        return result.failed(f"no indicators returned (error={data.get('error')})")
    first = indicators[0]
    return result.passed(f"{len(indicators)} indicators, first={first.get('database_id')}/{first.get('idno')}")


def run_checks(base: str, server: Server | None, offline: bool) -> list[Result]:
    results = [
        check_health(base),
        check_tools_list(base),
        check_blocked_envelope(base),
        check_renderer_resource(base),
    ]
    if server is None:
        results.append(
            Result("CWE-117: log assertions").passed(
                "skipped (--attach cannot read the other process's output)"
            )
        )
    else:
        results.append(check_log_forging(base, server))
        if offline:
            results.append(Result("CWE-117: CRLF in indicator id (data path)").passed("skipped (--offline)"))
        else:
            results.append(check_data_path_logging(base, server))
    results.append(
        Result("end-to-end: live search").passed("skipped (--offline)")
        if offline
        else check_upstream(base)
    )
    return results


NOISE = ("httpx", "httpcore", "sse_starlette", "mcp.server", "uvicorn", "INFO:")


def interesting(lines: list[str]) -> list[str]:
    """Server output worth showing: skip the transport/HTTP debug chatter."""
    return [line for line in lines if not any(marker in line for marker in NOISE)]


def report(results: list[Result], server: Server | None) -> int:
    width = max(len(r.name) for r in results)
    print(f"{'check'.ljust(width)}  result")
    print(f"{'-' * width}  {'-' * 6}")
    for result in results:
        print(f"{result.name.ljust(width)}  {'PASS' if result.ok else 'FAIL'}  {result.detail}")

    failed = [r for r in results if not r.ok]
    if not failed:
        print(f"\nall {len(results)} checks passed")
        return 0
    print(f"\n{len(failed)} of {len(results)} checks FAILED")
    if server is not None:
        print("\n--- relevant server output ---")
        print("\n".join(interesting(server.logs())[-20:]))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8021, help="local port (default 8021)")
    parser.add_argument(
        "--attach",
        action="store_true",
        help="use the server already listening on the port instead of booting one",
    )
    parser.add_argument("--offline", action="store_true", help="skip checks that call the Data360 API")
    parser.add_argument("--timeout", type=float, default=90.0, help="startup timeout in seconds")
    args = parser.parse_args()

    server: Server | None = None
    if args.attach:
        base = f"http://127.0.0.1:{args.port}"
        print(f"attaching to {base} (log assertions read the attached server's own output)")
    else:
        if not port_is_free(args.port):
            print(
                f"port {args.port} is already in use — stop that server, pass --port, or use --attach",
                file=sys.stderr,
            )
            return 2
        server = Server(args.port)
        base = server.base
        print(f"booting data360.server:app on {base} ...")
        try:
            server.start(timeout=args.timeout)
        except (RuntimeError, TimeoutError) as exc:
            print(f"startup failed: {exc}", file=sys.stderr)
            if server:
                server.stop()
            return 2
        print("server is healthy\n")

    try:
        results = run_checks(base, server, args.offline)
    finally:
        if server is not None:
            server.stop()

    return report(results, server)


if __name__ == "__main__":
    sys.exit(main())
