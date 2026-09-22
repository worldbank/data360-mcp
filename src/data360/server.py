import hashlib
import json
import logging
import os
import uuid

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from data360.mcp_server.resources import CORSStaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest

from data360.config import get_mcp_server_settings, setup_logging
from data360.health import get_liveness_body, run_readiness
from data360.http_client import aclose_all_httpx_clients
from data360.otel_setup import (
    configure_open_telemetry_for_server,
    instrument_httpx_outbound,
)

_audit_logger = logging.getLogger("audit")
_telemetry_client = None

# Setup logging from configuration
import sys

_logger = logging.getLogger(__name__)
mcp_settings = get_mcp_server_settings()

# Parse port from argv, supporting both '--port 8021' and '--port=8021' forms.
_parsed_port: int | None = None
for _arg in sys.argv:
    if _arg.startswith("--port="):
        try:
            _parsed_port = int(_arg.split("=", 1)[1])
        except ValueError:
            logging.warning("Could not parse port from argv argument '%s'; using default.", _arg)
        break
    if _arg == "--port":
        _idx = sys.argv.index(_arg)
        try:
            _parsed_port = int(sys.argv[_idx + 1])
        except (ValueError, IndexError):
            logging.warning("Could not parse port after '--port' in argv; using default.")
        break
if _parsed_port is not None:
    mcp_settings.port = _parsed_port

setup_logging(
    log_file=mcp_settings.log_file,
    log_level=mcp_settings.log_level,
    env=mcp_settings.env,
    azure_connection_string=mcp_settings.azure_connection_string,
)

# Tracer export (Azure in deployed envs; optional OTLP/console when MCP_ENV=local) and httpx spans
configure_open_telemetry_for_server(mcp_settings)
instrument_httpx_outbound()

# Import MCP after telemetry so the process uses an instrumented httpx from the first request.
from data360.mcp_server import mcp  # noqa: E402

_connection_string = mcp_settings.azure_connection_string or os.environ.get(
    "APPLICATIONINSIGHTS_CONNECTION_STRING"
)

# Initialize OpenCensus TelemetryClient for custom events (forwarded to Splunk)
if mcp_settings.env != "local" and _connection_string:
    try:
        from opencensus.ext.azure.log_exporter import (
            AzureEventHandler,  # type: ignore[import-untyped]
        )

        # Create a dedicated logger for custom events
        event_logger = logging.getLogger("customEvents")
        event_logger.setLevel(logging.INFO)

        # Add Azure handler that sends to customEvents table
        azure_handler = AzureEventHandler(connection_string=_connection_string)
        event_logger.addHandler(azure_handler)

        _telemetry_client = event_logger
    except ImportError:
        pass


class SecurityValidationMiddleware(BaseHTTPMiddleware):
    """Validate MCP tool calls to prevent prompt injection and unauthorized access."""

    async def dispatch(self, request: Request, call_next):
        # Only validate MCP JSON-RPC tool calls (not health probes under /mcp/*)
        if request.url.path in ("/mcp/health", "/mcp/ready"):
            return await call_next(request)
        if not request.url.path.startswith("/mcp"):
            return await call_next(request)

        try:
            # Read and parse body
            body_bytes = await request.body()
            if not body_bytes:
                return await call_next(request)

            body = json.loads(body_bytes)
            method = body.get("method", "")

            # Log tools/list requests for monitoring (allowed but monitored)
            if method == "tools/list":
                client_ip = request.headers.get(
                    "X-Forwarded-For",
                    request.client.host if request.client else "unknown",
                )
                logging.info(f"tools/list called from IP: {client_ip}")

            # Validate tools/call requests
            if method == "tools/call":
                from data360.mcp_server.security_validator import (  # noqa: PLC0415
                    validate_search_arguments,
                    validate_tool_call,
                )

                params = body.get("params", {})
                tool_name = params.get("name", "")
                arguments = params.get("arguments", {})

                # Validate tool call
                is_valid, error_msg = validate_tool_call(tool_name, arguments)
                if not is_valid:
                    logging.warning(
                        f"Security violation: {error_msg} | Tool: {tool_name}"
                    )
                    return JSONResponse(
                        status_code=403,
                        content={
                            "jsonrpc": "2.0",
                            "id": body.get("id"),
                            "error": {"code": -32001, "message": error_msg},
                        },
                    )

                # Additional validation for all search term inputs
                if tool_name == "data360_search_indicators":
                    is_valid, error_msg = validate_search_arguments(arguments)
                    if not is_valid:
                        logging.warning(
                            "Search query blocked: %s",
                            str(arguments)[:200],
                        )
                        return JSONResponse(
                            status_code=403,
                            content={
                                "jsonrpc": "2.0",
                                "id": body.get("id"),
                                "error": {"code": -32001, "message": error_msg},
                            },
                        )

        except json.JSONDecodeError:
            pass  # Let MCP handle invalid JSON
        except Exception:
            logging.error("Security validation error", exc_info=True)
            # Continue on validation errors to avoid blocking legitimate requests

        return await call_next(request)


class AuditLogMiddleware(BaseHTTPMiddleware):
    """Log structured audit entries for every MCP request."""

    async def dispatch(self, request: Request, call_next):
        # Only audit MCP JSON-RPC calls (not health probes)
        if request.url.path in ("/mcp/health", "/mcp/ready"):
            return await call_next(request)
        if not request.url.path.startswith("/mcp"):
            return await call_next(request)
        session_id = str(uuid.uuid4())
        requestor_id = request.headers.get(
            "X-Forwarded-For", request.client.host if request.client else "unknown"
        )
        timestamp = datetime.now(UTC).isoformat()
        # Read and restore body so downstream handlers still receive it
        body_bytes = await request.body()
        prompt = ""
        prompt_hash = ""
        try:
            body = json.loads(body_bytes)
            method = body.get("method", "")
            params = body.get("params", {})
            prompt = json.dumps(
                {"method": method, "params": params}, separators=(",", ":")
            )
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        except Exception:
            pass
        response = await call_next(request)

        properties = {
            "session_id": session_id,
            "requestor_id": requestor_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "status_code": str(response.status_code),
            "path": request.url.path,
        }

        # Log to traces with custom dimensions
        _audit_logger.info("mcp_audit", extra={"custom_dimensions": properties})

        # Also log as custom event for Splunk forwarding
        if _telemetry_client:
            _telemetry_client.info(
                "MCP_Request",
                extra={
                    "custom_dimensions": properties,
                    "event_name": "MCP_Request",
                },
            )

        return response


from fastmcp import settings
settings.stateless_http = True

# NOTE: import to be able to run the server with all definitions loaded
# path="/mcp" means the MCP endpoint lives at /mcp (no trailing slash needed)
mcp_app = mcp.http_app(path="/mcp")


async def health_check(request: StarletteRequest) -> JSONResponse:
    """Liveness probe under the MCP URL prefix (GET /mcp/health)."""
    del request
    return JSONResponse(get_liveness_body())


async def ready_check(request: StarletteRequest) -> JSONResponse:
    """Readiness probe under the MCP URL prefix (GET /mcp/ready)."""
    del request
    status_code, body = await run_readiness()
    return JSONResponse(content=body, status_code=status_code)


# Starlette routes on mcp_app: paths are absolute from mount root (not nested under /mcp).
# Use /mcp/health so probes sit beside the streamable HTTP endpoint at /mcp.
mcp_app.add_route("/mcp/health", health_check, methods=["GET", "HEAD"])
mcp_app.add_route("/mcp/ready", ready_check, methods=["GET", "HEAD"])


@asynccontextmanager
async def _lifespan_with_http_cleanup(app: FastAPI):
    """Run MCP startup/shutdown, then close the shared httpx clients."""
    async with mcp_app.router.lifespan_context(mcp_app):
        yield
    await aclose_all_httpx_clients()


# https://gofastmcp.com/deployment/http#asgi-application
# redirect_slashes=False prevents 308 redirects between /mcp and /mcp/
app = FastAPI(
    title="Data360 MCP Server",
    lifespan=_lifespan_with_http_cleanup,
    redirect_slashes=False,
)  # pyright: ignore[reportUnusedExpression]

app.add_middleware(AuditLogMiddleware)
# SecurityValidationMiddleware is enabled for incoming request validation.
app.add_middleware(SecurityValidationMiddleware)

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI for incoming request tracking
if mcp_settings.env != "local" and _connection_string:
    try:
        from opentelemetry.instrumentation.fastapi import (
            FastAPIInstrumentor,  # type: ignore[import-untyped]
        )

        FastAPIInstrumentor.instrument_app(app)
    except ImportError:
        pass


@app.get("/")
async def root():
    return {
        "service": "data360-mcp",
        "health": "/mcp/health",
        "ready": "/mcp/ready",
        "mcp": "/mcp",
    }



class VizSpecRequest(BaseModel):
    database_id: str
    indicator_id: str
    country_code: str | None = None
    start_year: int | None = None
    end_year: int | None = None
    disaggregation_filters: dict[str, str | None] | None = None
    chart_type: str | None = None
    relevant_fields: list[str] | None = None
    chart_title: str | None = None
    series_labels: dict[str, str] | None = None


@app.post("/api/viz-spec")
async def get_viz_spec_endpoint(req: VizSpecRequest):
    from data360 import visualization as data360_viz

    try:
        # Pass charts_api_url_override=None to force local static file storage,
        # avoiding mutation of the shared @ft.cache singleton (which is a race condition
        # under concurrent requests). The HTML app always wants local specs for direct
        # file access; blob storage routing happens via the MCP tool path, not this endpoint.
        res = await data360_viz.get_viz_spec(
            database_id=req.database_id,
            indicator_id=req.indicator_id,
            country_code=req.country_code,
            start_year=req.start_year,
            end_year=req.end_year,
            disaggregation_filters=req.disaggregation_filters,
            chart_type=req.chart_type,
            relevant_fields=req.relevant_fields,
            chart_title=req.chart_title,
            series_labels=req.series_labels,
            charts_api_url_override=None,
        )
    except Exception as e:
        _logger.exception("Failed to generate viz spec: %s", e)
        return JSONResponse(status_code=500, content={"error": "Failed to generate visualization spec due to an internal error."})

    if res.get("error"):
        return JSONResponse(status_code=400, content={"error": res.get("error")})

    spec = res.get("spec")
    if not spec:
        return JSONResponse(status_code=500, content={"error": "Vega-Lite spec was not generated."})

    return {k: v for k, v in {
        "spec": spec,
        "reason": res.get("reason"),
        "strategy": res.get("strategy"),
    }.items() if v is not None}




if os.environ.get("PYTEST_CURRENT_TEST"):
    static_dir = os.path.join(os.getcwd(), "static")
else:
    server_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(server_dir, "..", ".."))
    static_dir = os.path.join(project_root, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", CORSStaticFiles(directory=static_dir), name="static")
# Mount MCP app at root — the path="/mcp" in http_app() handles the /mcp route
app.mount("/", mcp_app)


# Tools and other resources are automatically registered via imports in mcp_server/__init__.py
# See src/data360/mcp_server/tools.py for tool definitions
