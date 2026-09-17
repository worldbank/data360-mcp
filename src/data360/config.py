import functools as ft
import logging
import os

# Load environment variables from .env file at import time, but only when not
# running under pytest.  Test sessions set env vars explicitly via conftest/fixtures;
# unconditional load_dotenv() would stomp on those values with whatever is in a
# local .env file, making tests environment-dependent.
import os as _os
import sys
from pathlib import Path

from dotenv import load_dotenv

from data360.log_sanitize import install_crlf_sanitizer

# Skip .env during pytest (collection and execution) so tests control DATA360_* URLs.
if not _os.environ.get("PYTEST_CURRENT_TEST") and not _os.environ.get("PYTEST_RUNNING"):
    load_dotenv()
del _os


from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPServerSettings(BaseSettings):
    """Configuration settings for MCP server."""

    port: int = Field(
        default=8000,
        description="Port for the MCP server",
    )
    transport: str = Field(
        default="http",
        description="Transport for the MCP server",
    )
    log_file: str | None = Field(
        default=None,
        description="Path to log file. If None, logs go to stderr/stdout.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    charts_api_url: str | None = Field(
        default=None,
        description="URL for external charts API to store Vega-Lite specs (e.g. https://.../api/v1/charts). When set, viz specs are POSTed here instead of saving to static.",
    )
    charts_api_token: str | None = Field(
        default=None,
        description="Optional bearer token for external charts API (Authorization header).",
    )
    env: str | None = Field(
        default=None,
        description="Deployment environment (e.g. local, dev, staging, prod). Azure App Insights "
        "and OpenTelemetry export are disabled when set to 'local' unless you set "
        "OTEL_EXPORTER_OTLP_ENDPOINT (or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) and/or "
        "MCP_OTEL_CONSOLE=1 for local trace export; see data360.otel_setup.",
    )
    azure_connection_string: str | None = Field(
        default=None,
        description="Azure Application Insights connection string. If unset, falls back to APPLICATIONINSIGHTS_CONNECTION_STRING env var.",
    )
    readiness_enabled: bool = Field(
        default=True,
        description="When false, GET /ready returns 200 with readiness_checks=disabled (no dependency probes).",
    )
    health_check_timeout: float = Field(
        default=5.0,
        description="Per-check timeout in seconds for GET /ready outbound probes.",
    )


    model_config = SettingsConfigDict(env_prefix="MCP_")


class Data360Settings(BaseSettings):
    """Configuration settings for Data360 API integration."""

    api_base_url: str = Field(
        ...,
        description="Base URL for the Data360 API",
    )
    codelist_api_base_url: str = Field(
        default="https://extdataportal.worldbank.org/api/data360/metadata/codelist",
        description=(
            "URL for the unified Data360 codelist API. Returns all dimension codelists "
            "(COMP_BREAKDOWN, UNIT_MEASURE, SEX, AGE, URBANISATION, FREQ, REF_AREA, …) "
            "in a single call. Used by scripts/build_extdataportal_codelists.py to regenerate "
            "the bundled src/data360/extdataportal_codelists.json."
        ),
    )
    search_url: str | None = Field(
        default=None,
        description="URL for search endpoint (defaults to {api_base_url}/data360/portal/v1/public_data360_search)",
    )
    metadata_url: str | None = Field(
        default=None,
        description="URL for metadata endpoint (defaults to {api_base_url}/data360/metadata)",
    )
    disaggregation_url: str | None = Field(
        default=None,
        description="URL for disaggregation endpoint (defaults to {api_base_url}/data360/disaggregation)",
    )
    dimensions_url: str | None = Field(
        default=None,
        description="URL for dimensions endpoint (defaults to {api_base_url}/data360/portal/v1/dimensions)",
    )
    data_url: str | None = Field(
        default=None,
        description="URL for data endpoint (defaults to {api_base_url}/data)",
    )
    metadata_search_fields: list[str] = Field(
        default=[
            "series_description/idno",
            "series_description/name",
            "series_description/database_id",
            "series_description/definition_long",
            "series_description/methodology",
            "series_description/limitation",
            "series_description/relevance",
            "series_description/aggregation_method",
        ]
    )

    # Limit the data observations to the following confidentiality levels
    data_obs_confidentiality_levels: list[str] = Field(
        default=[
            "PU",  # Public
            # "OU",  # Official Use
            # "CO",  # Confidential
            # "SC",  # Strictly Confidential
        ],
        description="Confidentiality levels to limit the data observations to",
    )

    model_config = SettingsConfigDict(env_prefix="DATA360_")

    @property
    def api_url(self) -> str:
        """Base path for Data360 HTTP APIs: ``{api_base_url}/data360`` (no trailing slash)."""
        return f"{self.api_base_url.rstrip('/')}/data360"


@ft.cache
def get_data360_settings() -> Data360Settings:
    """Get cached Data360 settings instance."""
    return Data360Settings()  # pyright: ignore[reportCallIssue]


@ft.cache
def get_mcp_server_settings() -> MCPServerSettings:
    """Get cached MCP server settings instance."""
    return MCPServerSettings()  # pyright: ignore[reportCallIssue]


def setup_logging(
    log_file: str | None = None,
    log_level: str = "INFO",
    env: str | None = None,
    azure_connection_string: str | None = None,
) -> None:
    """Configure logging to write to a file and/or console, and optionally Azure App Insights.

    Args:
        log_file: Path to log file. If None, logs only go to stderr.
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        env: Deployment environment. Azure handler is skipped when 'local'.
        azure_connection_string: Azure App Insights connection string. Falls back to
            APPLICATIONINSIGHTS_CONNECTION_STRING env var if not provided.
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # CWE-117: neutralize CR/LF in every log record process-wide so untrusted
    # request/upstream data can never forge log lines (console, file, Azure).
    install_crlf_sanitizer()

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file is specified)
    if log_file:
        log_path = Path(log_file)
        # Create parent directories if they don't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Azure App Insights handler (skipped for local environment or when no key is configured)
    effective_connection_string = azure_connection_string or os.environ.get(
        "APPLICATIONINSIGHTS_CONNECTION_STRING"
    )
    if env != "local" and effective_connection_string:
        try:
            from opencensus.ext.azure.log_exporter import (
                AzureLogHandler,  # type: ignore[import-untyped]
            )

            def _callback(_):
                return True

            azure_handler = AzureLogHandler(
                connection_string=effective_connection_string
            )
            azure_handler.setLevel(numeric_level)
            azure_handler.add_telemetry_processor(_callback)
            root_logger.addHandler(azure_handler)
            root_logger.info("Azure App Insights logging enabled (env=%s).", env)
        except ImportError:
            root_logger.warning(
                "opencensus-ext-azure not installed; skipping Azure App Insights handler."
            )
