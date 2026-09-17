"""MCP Tools for the Data360 server.

Thin wrapper layer that registers API functions as MCP tools with optimized signatures,
concise docstrings to reduce token context bloat, and validation schemas.
"""

import json
import os
import threading
from typing import Any, Literal, Optional

import pydantic_core
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.tools.tool import Tool
from mcp.types import TextContent

from data360 import api as data360_api
from data360 import providers as data360_providers
from data360 import visualization as data360_viz
from data360 import viz_config as data360_viz_config

from ._server_definition import mcp
from .tool_spans import instrument_mcp_tool

# ---------------------------------------------------------------------------
# Serializer for aggregation tools
# ---------------------------------------------------------------------------


def _compact_aggregation_serializer(data: Any) -> str:
    """Compact serializer for aggregation tool responses.

    Calls ``to_compact()`` on the response model if available, producing a
    token-efficient JSON representation while preserving all PCN claim_ids
    for data provenance verification.
    """
    if hasattr(data, "to_compact"):
        return json.dumps(data.to_compact(), separators=(",", ":"))
    return pydantic_core.to_json(data, fallback=str).decode()


def _normalize_disaggregation_filters(filters: dict[str, Any] | None) -> dict[str, str | None] | None:
    """Normalize user-provided disaggregation filters.
    Converts list values (e.g., ["F", "M"]) to comma-separated strings (e.g., "F,M")
    to conform to the underlying API support while remaining type-flexible for LLM callers.
    """
    if filters is None:
        return None
    normalized = {}
    for k, v in filters.items():
        if v is None:
            normalized[k] = None
        elif isinstance(v, list):
            normalized[k] = ",".join(str(item).strip() for item in v if item is not None)
        else:
            normalized[k] = str(v)
    return normalized


# ---------------------------------------------------------------------------
# Tool Wrapper Functions
# ---------------------------------------------------------------------------


async def _search_indicators(
    query: str | None = None,
    required_country: str | None = None,
    limit: int = 5,
    offset: int = 0,
    queries: list[str] | None = None,
    query_groups: list[dict[str, Any]] | None = None,
    result_layout: str = "merged",
    dedupe: bool = True,
    database: str | None = None,
) -> Any:
    """Search for Data360 indicators with enriched metadata for selection.

    Use when the user asks for data on a development topic (e.g. GDP, poverty, education).
    Default to using the single `query` parameter for any single topic/indicator search. Use the `queries` or `query_groups` parameters ONLY when the request involves multiple topics or scopes (2 or more).
    Provide exactly one of `query`, `queries`, or `query_groups`. One of these is strictly required.

    ### Parameter Selection Decision Tree (CRITICAL):
    1. **Exactly 1 Topic** (e.g., "life expectancy" or "mortality rate") for any number of countries → you MUST use the single `query` parameter + `required_country`. Do NOT use `queries` with only one element, as it will fail. Do NOT combine multiple topics with 'and' or 'or' in `query` (e.g. do NOT use query="GDP and inflation").
    2. **Multiple Topics, Same Country/Countries** (e.g., "life expectancy and GDP per capita" for Japan) → you MUST use the `queries` list parameter (e.g. `queries=["life expectancy", "GDP per capita"]`) + `required_country`. Do NOT make multiple tool calls. Do NOT pass multiple topics as a single query string (e.g., query="life expectancy and GDP per capita" is invalid).
    3. **Different Topics targeting Different Countries** (e.g., "life expectancy for Japan, but GDP and mortality rate for Korea") → you MUST use the `query_groups` parameter. Do NOT use `queries`.

    Args:
        query: Single topic query (e.g. "unemployment"). Use ONLY for a single topic. Do NOT combine multiple topics with 'and' or 'or' (e.g. do NOT use query="population and life expectancy"). Avoid special characters like parentheses () or dollar signs $. Example: 'GDP per capita'.
        required_country: Semicolon-separated ISO country codes (e.g. "KEN;USA"). Shared across all queries in 'query' or 'queries'. Consider calling `data360_expand_country_group` to find country codes in regional/income groups, or `data360_find_codelist_value` to resolve country names.
        limit: Max indicators per query (default 5).
        offset: Offset for pagination.
        queries: List of topics for multi-topic search (must contain at least 2 non-empty search strings). Use ONLY when 2 or more topics target the SAME countries/geographic scope (e.g. ['GDP per capita', 'inflation rate']).
        query_groups: Grouped queries with specific country scopes. Use ONLY when different topics/queries target different country scopes. Example: [{'queries': ['life expectancy'], 'country': 'JPN'}, {'queries': ['GDP per capita'], 'country': 'KOR'}].
        result_layout: Mode to return results: "merged" (flat, deduped list of indicators) or "by_query" (indicators grouped by search query).
        dedupe: De-duplicate indicators across query results.
        database: Optional database name or ID to filter search results (e.g. "wdi", "wgi", "World Development Indicators"). Multiple databases can be queried at once by separating them with a semicolon (e.g. "pip; lpgd; sgi").
    """
    if queries is not None and len(queries) == 1 and not query:
        query = queries[0]
        queries = None

    return await data360_api.search(
        query=query,
        required_country=required_country,
        limit=limit,
        offset=offset,
        queries=queries,
        query_groups=query_groups,
        result_layout=result_layout,
        dedupe=dedupe,
        database=database,
    )


async def _search_datasets(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> Any:
    """Search for Data360 datasets matching a query.

    Use when the user asks for dataset details, catalogs, or source databases (e.g. "Findex", "WDI").

    Args:
        query: Topic or dataset search term (e.g. "findex"). Avoid special characters like parentheses () or dollar signs $ as they cause search failures.
        limit: Max datasets to return (default 10).
        offset: Offset for pagination.
    """
    return await data360_api.search_datasets(
        query=query,
        limit=limit,
        offset=offset,
    )


async def _get_metadata(
    database_id: str,
    indicator_id: str,
    select_fields: list[str] | None = None,
    fetch_disaggregation: bool = True,
    required_country: str | None = None,
) -> Any:
    """Get metadata and disaggregation options for a Data360 indicator.

    Use when you need detailed methodology, source notes, or limitations for an indicator.
    Ensure the database ID and the indicator ID are already in context (e.g., from `data360_search_indicators`) before using this tool. Do not guess or hallucinate these IDs.

    Args:
        database_id: Database identifier (e.g., "WB_WDI").
        indicator_id: Indicator ID (e.g., "WB_WDI_NY_GDP_PCAP_KD").
        select_fields: Optional metadata fields to return (e.g., ["methodology", "relevance"]).
        fetch_disaggregation: Whether to include disaggregation options.
        required_country: Semicolon-separated ISO country codes to check coverage.
    """
    return await data360_api.get_metadata(
        database_id=database_id,
        indicator_id=indicator_id,
        select_fields=select_fields,
        fetch_disaggregation=fetch_disaggregation,
        required_country=required_country,
    )


async def _get_data(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 50,
    offset: int = 0,
    ref_area_filter: Literal["none", "member_economies_only"] = "member_economies_only",
    year: int | None = None,
) -> Any:
    """Retrieve indicator observations from the Data360 API.

    Use when you need actual numeric values (OBS_VALUE) for specific countries and years.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.
    Call `data360_get_disaggregation` first to find available years and breakdowns for the `disaggregation_filters`.

    Args:
        database_id: Database identifier (e.g., "WB_WDI").
        indicator_id: Indicator ID (e.g., "WB_WDI_NY_GDP_PCAP_KD").
        country_code: Semicolon-separated ISO country codes (e.g. "KEN;USA").
        disaggregation_filters: Optional dimension filters. Values must be strings or null. Call `data360_get_disaggregation` first to find valid options.
        start_year: Start year (inclusive). Defaults to last 5 years if both bounds omitted;
            if only end_year is set, defaults to a 5-year window ending at end_year.
        end_year: End year (inclusive). See start_year for partial-bound defaults.
        limit: Max records per page (default 50, max 100).
        offset: Number of records to skip for pagination.
        ref_area_filter: Filter mode: "member_economies_only" (default) or "none".
        year: Specific single year to retrieve data for. Maps internally to start_year and end_year.
    """
    if year is not None:
        if start_year is None:
            start_year = year
        if end_year is None:
            end_year = year


    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    return await data360_api.get_data(
        database_id=database_id,
        indicator_id=indicator_id,
        country_code=country_code,
        disaggregation_filters=norm_filters,
        start_year=start_year,
        end_year=end_year,
        limit=limit,
        offset=offset,
        ref_area_filter=ref_area_filter,
    )


async def _get_disaggregation(
    database_id: str,
    indicator_id: str,
    required_country: str | None = None,
) -> dict[str, Any]:
    """Get valid filter values and disaggregation options for an indicator.

    Use to find available dimensions (e.g., SEX, AGE) and years before querying data or charts.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.

    Args:
        database_id: Database identifier (e.g., "WB_WDI").
        indicator_id: Indicator ID (e.g., "WB_WDI_NY_GDP_PCAP_KD").
        required_country: Semicolon-separated ISO country codes to check coverage.
    """
    return await data360_api.get_disaggregation(
        database_id=database_id,
        indicator_id=indicator_id,
        required_country=required_country,
    )


async def _find_codelist_value(
    codelist_type: str, query: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Resolve user-friendly names to API dimension codes.

    Use when you need to find codes for country names, sex, age, urbanisation, etc.

    Args:
        codelist_type: Dimension name (e.g. "REF_AREA", "SEX", "AGE", "URBANISATION").
        query: Search term (e.g. "Kenya", "female").
        limit: Max results to return (default 5).
    """
    return await data360_providers.find_codelist_value(
        codelist_type=codelist_type, query=query, limit=limit
    )


async def _list_indicators(database_id: str) -> list[str]:
    """Get all indicator IDs for a specific database.

    Use when you need the full list of indicator IDs for a dataset.

    Args:
        database_id: The database identifier (e.g., "WB_WDI").
    """
    return await data360_api.get_indicators(database_id=database_id)


async def _get_data_api_url(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
    year: int | None = None,
) -> str:
    """Generate the raw Data360 API URL for an indicator request.

    Low-level tool: use only when the caller specifically asks for the URL.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.

    Args:
        database_id: Database identifier (e.g. "WB_WDI").
        indicator_id: Indicator ID (e.g. "WB_WDI_NY_GDP_PCAP_KD").
        country_code: Semicolon-separated ISO country codes.
        start_year: Start year (inclusive). Defaults to last 5 years if omitted.
        end_year: End year (inclusive). Defaults to current year if omitted.
        disaggregation_filters: Optional dimension filters.
        year: Specific single year to generate the URL for. Maps internally to start_year and end_year.
    """
    if year is not None:
        if start_year is None:
            start_year = year
        if end_year is None:
            end_year = year


    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    return await data360_api.get_data_api_url(
        database_id=database_id,
        indicator_id=indicator_id,
        country_code=country_code,
        start_year=start_year,
        end_year=end_year,
        disaggregation_filters=norm_filters,
    )





# ---------------------------------------------------------------------------
# Bundled Vega library cache (thread-safe, loaded once on first use)
# ---------------------------------------------------------------------------

_vega_libs_lock = threading.Lock()
_vega_libs_cache: tuple[str, str, str, str] | None = None


def get_cached_vega_libs() -> tuple[str, str, str, str]:
    """Load and cache local Vega library scripts from static/libs (thread-safe)."""
    global _vega_libs_cache
    if _vega_libs_cache is not None:
        return _vega_libs_cache
    with _vega_libs_lock:
        if _vega_libs_cache is not None:  # double-check after acquiring lock
            return _vega_libs_cache
        from pathlib import Path
        import logging

        libs_dir = (
            Path(__file__).resolve().parent.parent.parent.parent / "static" / "libs"
        )
        try:
            vega_js = (libs_dir / "vega.js").read_text(encoding="utf-8")
            vega_lite_js = (libs_dir / "vega-lite.js").read_text(encoding="utf-8")
            vega_embed_js = (libs_dir / "vega-embed.js").read_text(encoding="utf-8")
            vega_interp_js = (libs_dir / "vega-interpreter.js").read_text(
                encoding="utf-8"
            )
        except Exception as e:
            logging.getLogger("data360").warning(
                "Failed to load local Vega library scripts: %s", repr(e)
            )
            vega_js = vega_lite_js = vega_embed_js = vega_interp_js = ""
        res = (vega_js, vega_lite_js, vega_embed_js, vega_interp_js)
        if all(res):
            _vega_libs_cache = res
        return res


# ---------------------------------------------------------------------------
# Markdown summary helper (text content block for viz ToolResults)
# ---------------------------------------------------------------------------


def _make_text_summary(
    spec: "dict[str, Any] | None",
    strategy: str,
    reason: str,
    warning: str | None = None,
    subtitle_line: str | None = None,
    source_line: str | None = None,
    url: str | None = None,
) -> str:
    """Build a markdown summary table from a Vega-Lite spec for the text content block."""
    import pandas as pd

    lines: list[str] = []

    if warning:
        lines.append(f"### Warning\n{warning}\n")

    lines.append(f"### Data Summary ({strategy})")
    lines.append(reason)
    if subtitle_line:
        lines.append(f"*{subtitle_line}*")
    lines.append("")

    data_rows: list[dict] = []
    if isinstance(spec, dict):
        data_rows = spec.get("data", {}).get("values", [])

    if data_rows:
        try:
            df = pd.DataFrame(data_rows)
            # Reorder: put time/area columns first
            cols = list(df.columns)
            for p in reversed(
                ["TIME_PERIOD", "time_period", "REF_AREA", "ref_area"]
            ):
                if p in cols:
                    cols.remove(p)
                    cols.insert(0, p)
            df = df[cols]

            headers = [c.replace("_", " ").title() for c in df.columns]
            lines.append("| " + " | ".join(headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(df.columns)) + " |")
            for _, row in df.iterrows():
                vals = []
                for col in df.columns:
                    v = row[col]
                    if v is None or (isinstance(v, float) and pd.isna(v)):
                        vals.append("")
                    elif isinstance(v, float):
                        vals.append(f"{v:,.2f}")
                    else:
                        vals.append(str(v))
                lines.append("| " + " | ".join(vals) + " |")
        except Exception:
            lines.append("No tabular data available.")
    else:
        lines.append("No data available.")

    if source_line:
        lines.append(f"\n*{source_line}*")
    if url:
        lines.append(f"\n*Vega-Lite Spec URL:* {url}")

    return "\n".join(lines)

async def _get_viz_spec(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
    chart_type: str | None = None,
    relevant_fields: list[str] | None = None,
    custom_constraints: list[str] | None = None,
    use_default_constraints: bool = True,
    chart_title: str | dict | None = None,
    series_labels: dict[str, str] | None = None,
    strategy_override: str | None = None,
    year: int | None = None,
) -> ToolResult:
    """Generate a Vega-Lite chart from a single Data360 indicator.

    Use when the user requests a chart or plot for a single indicator.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.
    Call `data360_get_disaggregation` first to find available years and breakdowns for the `disaggregation_filters`.

    Args:
        database_id: Database identifier (e.g. "WB_WDI").
        indicator_id: Indicator ID (e.g. "WB_WDI_NY_GDP_PCAP_KD").
        country_code: Semicolon-separated ISO country codes (e.g. "KEN;USA").
        start_year: Start year (inclusive). Defaults to last 5 years if omitted.
        end_year: End year (inclusive). Defaults to current year if omitted.
        disaggregation_filters: Optional dimension filters.
        chart_type: Optional chart type suggestion. If omitted (recommended), the routing engine automatically determines the optimal chart type and strategy based on the data profile. Do not specify this argument unless the user explicitly requested a specific chart type.
        relevant_fields: Fields to include in visual encodings.
        custom_constraints: Custom Draco design rules.
        use_default_constraints: Whether to apply default Draco design constraints.
        chart_title: A concise, human-synthesized title summarizing the data insight (e.g. 'Renewable Energy Share in South Asia (2020)'). Prefer clean, natural phrasing instead of raw long indicator names.
        series_labels: Rename dimension codes for legend (e.g. {"WGI_EST": "Estimate"}).
        strategy_override: Explicitly force a chart strategy (e.g. "stacked_bar", "temporal_single").
        year: Specific single year to generate the chart for. Maps internally to start_year and end_year.
    """
    if year is not None:
        if start_year is None:
            start_year = year
        if end_year is None:
            end_year = year

    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    res = await data360_viz.get_viz_spec(
        database_id=database_id,
        indicator_id=indicator_id,
        country_code=country_code,
        start_year=start_year,
        end_year=end_year,
        disaggregation_filters=norm_filters,
        chart_type=chart_type,
        relevant_fields=relevant_fields,
        custom_constraints=custom_constraints,
        use_default_constraints=use_default_constraints,
        chart_title=chart_title,
        series_labels=series_labels,
        strategy_override=strategy_override,
    )

    if res.get("error"):
        raise ToolError(res["error"])

    url = res.get("url")
    strategy = res.get("strategy") or "unknown"
    reason = res.get("reason") or ""
    warning = res.get("warning")
    source_line = res.get("source_line")
    subtitle_line = res.get("subtitle_line")

    # Prefer the spec already carried in the result dict (populated by _ok() in
    # visualization.py). The disk-reload below is a fallback for callers that
    # do not propagate the spec (e.g. when the chart URL points to an external
    # charts API rather than the local static file server).
    spec: dict | None = res.get("spec") or None
    if spec is None and url:
        try:
            spec_id = url.split("/")[-1].replace("_vega.json", "")
            if os.environ.get("PYTEST_CURRENT_TEST"):
                specs_dir = os.path.join(os.getcwd(), "static", "viz_specs")
            else:
                server_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(server_dir, "..", "..", ".."))
                specs_dir = os.path.join(project_root, "static", "viz_specs")
            vega_path = os.path.join(specs_dir, f"{spec_id}_vega.json")
            if os.path.exists(vega_path):
                with open(vega_path, "r") as f:
                    spec = json.load(f)
        except Exception:
            pass

    text_summary = _make_text_summary(
        spec=spec,
        strategy=strategy,
        reason=reason,
        warning=warning,
        source_line=source_line,
        subtitle_line=subtitle_line,
        url=url,
    )

    structured = {
        "spec": spec,
        "strategy": strategy,
        "url": url,
        "error": None,
        "warning": warning,
        "reason": reason,
        "source_line": source_line,
        "subtitle_line": subtitle_line,
    }
    return ToolResult(
        content=[
            TextContent(type="text", text=json.dumps(structured)),
            TextContent(type="text", text=text_summary),
        ],
        structured_content=structured,
    )


async def _get_multi_indicator_viz_spec(
    indicator_ids: list[dict[str, str]] | None = None,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
    chart_type: str | None = None,
    chart_title: str | dict | None = None,
    series_labels: dict[str, str] | None = None,
    strategy_override: str | None = None,
    year: int | None = None,
) -> ToolResult:
    """Generate a Vega-Lite chart comparing multiple Data360 indicators.

    Use when you need to compare 2–4 indicators (e.g. via scatterplot or dual-axis line chart).
    Ensure the database IDs and indicator IDs are already in context before using this tool. Do not guess or hallucinate these IDs.

    Args:
        indicator_ids: List of database/indicator dicts, e.g. [{"database_id": "WB_WDI", "indicator_id": "..."}].
        country_code: Semicolon-separated ISO country codes (e.g. "KEN;USA").
        start_year: Start year (inclusive). Defaults to last 5 years if omitted.
        end_year: End year (inclusive). Defaults to current year if omitted.
        disaggregation_filters: Optional dimension filters.
        chart_type: Optional chart type suggestion. If omitted (recommended), the routing engine automatically determines the optimal chart type and strategy based on the data profile. Do not specify this argument unless the user explicitly requested a specific chart type.
        chart_title: A concise, human-synthesized title summarizing the data insight (e.g. 'Renewable Energy Share in South Asia (2020)'). Prefer clean, natural phrasing instead of raw long indicator names.
        series_labels: Rename dimension codes for legend.
        strategy_override: Explicitly force a chart strategy (e.g. "stacked_bar", "vconcat_panels").
        year: Specific single year to compare indicators for. Maps internally to start_year and end_year.
    """
    if year is not None:
        if start_year is None:
            start_year = year
        if end_year is None:
            end_year = year

    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    res = await data360_viz.get_multi_indicator_viz_spec(
        indicator_ids=indicator_ids,
        country_code=country_code,
        start_year=start_year,
        end_year=end_year,
        disaggregation_filters=norm_filters,
        chart_type=chart_type,
        chart_title=chart_title,
        series_labels=series_labels,
        strategy_override=strategy_override,
    )

    if res.get("error"):
        raise ToolError(res["error"])

    url = res.get("url")
    strategy = res.get("strategy") or "unknown"
    reason = res.get("reason") or ""
    warning = res.get("warning")
    source_line = res.get("source_line")
    subtitle_line = res.get("subtitle_line")

    # Prefer the spec already carried in the result dict (populated by _ok() in
    # visualization.py). The disk-reload below is a fallback for callers that
    # do not propagate the spec (e.g. when the chart URL points to an external
    # charts API rather than the local static file server).
    spec: dict | None = res.get("spec") or None
    if spec is None and url:
        try:
            spec_id = url.split("/")[-1].replace("_vega.json", "")
            if os.environ.get("PYTEST_CURRENT_TEST"):
                specs_dir = os.path.join(os.getcwd(), "static", "viz_specs")
            else:
                server_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.abspath(os.path.join(server_dir, "..", "..", ".."))
                specs_dir = os.path.join(project_root, "static", "viz_specs")
            vega_path = os.path.join(specs_dir, f"{spec_id}_vega.json")
            if os.path.exists(vega_path):
                with open(vega_path, "r") as f:
                    spec = json.load(f)
        except Exception:
            pass

    text_summary = _make_text_summary(
        spec=spec,
        strategy=strategy,
        reason=reason,
        warning=warning,
        source_line=source_line,
        subtitle_line=subtitle_line,
        url=url,
    )

    structured = {
        "spec": spec,
        "strategy": strategy,
        "url": url,
        "error": None,
        "warning": warning,
        "reason": reason,
        "source_line": source_line,
        "subtitle_line": subtitle_line,
    }
    return ToolResult(
        content=[
            TextContent(type="text", text=json.dumps(structured)),
            TextContent(type="text", text=text_summary),
        ],
        structured_content=structured,
    )


def _get_supported_chart_types() -> str:
    """Return supported chart types and their data requirements as JSON.

    **DEPRECATED**: Read the ``data360://viz/chart-grammar`` resource instead.
    This tool is preserved for backward compatibility.
    """
    import json

    result = data360_viz.get_supported_chart_types()
    parsed = json.loads(result)
    parsed["_deprecation_notice"] = (
        "This tool is deprecated. Read the data360://viz/chart-grammar resource "
        "for comprehensive chart strategy rules. The data_profile in every viz "
        "response now includes per-indicator ranges and scale compatibility."
    )
    return json.dumps(parsed, indent=2)



async def _expand_country_group(
    group_code: str,
) -> dict[str, Any]:
    """Expand a REF_AREA group code into its constituent country codes.

    Use when you need individual country codes for a regional or income group code (e.g. "SAS").

    Args:
        group_code: The group code to expand (e.g. "SAS" for South Asia, "LIC" for Low Income).
    """
    return await data360_providers.expand_country_group(group_code=group_code)


async def _summarize_data(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    group_by: list[str] | None = None,
) -> Any:
    """Compute summary statistics for indicator data, grouped by dimensions.

    Use when the user asks about trends, changes over time, or general statistical summaries.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.

    Args:
        database_id: Database identifier (e.g. "WB_WDI").
        indicator_id: Indicator ID (e.g. "WB_WDI_NY_GDP_PCAP_KD").
        country_code: Semicolon-separated ISO country codes (e.g. "KEN;USA").
        disaggregation_filters: Optional dimension filters.
        start_year: Start year (inclusive). Defaults to last 5 years if omitted.
        end_year: End year (inclusive). Defaults to current year if omitted.
        group_by: Dimensions to group by (default is ["ref_area"]).
    """
    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    return await data360_api.summarize_data(
        database_id=database_id,
        indicator_id=indicator_id,
        country_code=country_code,
        disaggregation_filters=norm_filters,
        start_year=start_year,
        end_year=end_year,
        group_by=group_by,
    )


async def _rank_countries(
    database_id: str,
    indicator_id: str,
    country_group: str | None = None,
    country_codes: str | None = None,
    year: int | None = None,
    order: Literal["desc", "asc"] = "desc",
    top_n: int = 10,
    disaggregation_filters: dict[str, Any] | None = None,
    rank_universe: Literal["explicit", "all_member_economies"] = "explicit",
) -> Any:
    """Rank countries by indicator value for a specific year.

    Use when asked to rank countries, find leaderboards, or query top/bottom performing economies.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.

    Args:
        database_id: Database identifier (e.g. "WB_WDI").
        indicator_id: Indicator ID (e.g. "WB_WDI_NY_GDP_PCAP_KD").
        country_group: Code of region/income group (e.g. "SAS").
        country_codes: Semicolon-separated ISO country codes (e.g. "KEN;USA;NGA").
        year: Year for ranking. If omitted, selected automatically based on coverage.
        order: Sort order: "desc" (default, highest first) or "asc" (lowest first).
        top_n: Number of ranked entries to return.
        disaggregation_filters: Optional dimension filters.
        rank_universe: "explicit" (default, uses codes/group) or "all_member_economies" (world).
    """
    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    return await data360_api.rank_countries(
        database_id=database_id,
        indicator_id=indicator_id,
        country_group=country_group,
        country_codes=country_codes,
        year=year,
        order=order,
        top_n=top_n,
        disaggregation_filters=norm_filters,
        rank_universe=rank_universe,
    )


async def _compare_countries(
    database_id: str,
    indicator_id: str,
    country_codes: str,
    year: int | None = None,
    include_time_series: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, Any] | None = None,
) -> Any:
    """Compare an indicator across multiple countries (2 to 8).

    Use when asked to compare specific countries or find gaps/convergence between them.
    Ensure the database ID and the indicator ID are already in context before using this tool. Do not guess or hallucinate these IDs.
    Call `data360_get_disaggregation` first to find available years and breakdowns for the `disaggregation_filters`.

    Args:
        database_id: Database identifier (e.g. "WB_WDI").
        indicator_id: Indicator ID (e.g. "WB_WDI_NY_GDP_PCAP_KD").
        country_codes: Semicolon-separated ISO country codes (e.g. "KEN;NGA;ZAF").
        year: Snapshot comparison year. If omitted, selected automatically.
        include_time_series: Whether to return time-series data for trend comparison.
        start_year: Start year for time-series alignment. Defaults to last 5 years if omitted.
        end_year: End year for time-series alignment. Defaults to current year if omitted.
        disaggregation_filters: Optional dimension filters.
    """
    norm_filters = _normalize_disaggregation_filters(disaggregation_filters)

    return await data360_api.compare_countries(
        database_id=database_id,
        indicator_id=indicator_id,
        country_codes=country_codes,
        year=year,
        include_time_series=include_time_series,
        start_year=start_year,
        end_year=end_year,
        disaggregation_filters=norm_filters,
    )


# ---------------------------------------------------------------------------
# Tool Registration
# ---------------------------------------------------------------------------

search_indicators = mcp.tool(
    instrument_mcp_tool(_search_indicators, tool_name="data360_search_indicators"),
    name="data360_search_indicators",
)

search_datasets = mcp.tool(
    instrument_mcp_tool(_search_datasets, tool_name="data360_search_datasets"),
    name="data360_search_datasets",
)

get_metadata = mcp.tool(
    instrument_mcp_tool(_get_metadata, tool_name="data360_get_metadata"),
    name="data360_get_metadata",
)

get_data = mcp.tool(
    instrument_mcp_tool(_get_data, tool_name="data360_get_data"),
    name="data360_get_data",
)

get_disaggregation = mcp.tool(
    instrument_mcp_tool(_get_disaggregation, tool_name="data360_get_disaggregation"),
    name="data360_get_disaggregation",
)

find_codelist_value = mcp.tool(
    instrument_mcp_tool(_find_codelist_value, tool_name="data360_find_codelist_value"),
    name="data360_find_codelist_value",
)

list_indicators = mcp.tool(
    instrument_mcp_tool(_list_indicators, tool_name="data360_list_indicators"),
    name="data360_list_indicators",
)

get_data_api_url = mcp.tool(
    instrument_mcp_tool(_get_data_api_url, tool_name="data360_get_data_api_url"),
    name="data360_get_data_api_url",
)

get_viz_spec = mcp.tool(
    instrument_mcp_tool(_get_viz_spec, tool_name="data360_get_viz_spec"),
    name="data360_get_viz_spec",
    app=AppConfig(resource_uri="ui://data360-chart/index.html"),
)

get_multi_indicator_viz_spec = mcp.tool(
    instrument_mcp_tool(
        _get_multi_indicator_viz_spec,
        tool_name="data360_get_multi_indicator_viz_spec",
    ),
    name="data360_get_multi_indicator_viz_spec",
    app=AppConfig(resource_uri="ui://data360-chart/index.html"),
)

get_supported_chart_types = mcp.tool(
    instrument_mcp_tool(
        _get_supported_chart_types,
        tool_name="data360_get_supported_chart_types",
    ),
    name="data360_get_supported_chart_types",
)

expand_country_group = mcp.tool(
    instrument_mcp_tool(
        _expand_country_group, tool_name="data360_expand_country_group"
    ),
    name="data360_expand_country_group",
)

# ---------------------------------------------------------------------------
# Data Aggregation Tools (with custom serialization)
# ---------------------------------------------------------------------------

summarize_data = mcp.add_tool(
    Tool.from_function(
        instrument_mcp_tool(_summarize_data, tool_name="data360_summarize_data"),
        name="data360_summarize_data",
        serializer=_compact_aggregation_serializer,
    )
)

rank_countries = mcp.add_tool(
    Tool.from_function(
        instrument_mcp_tool(_rank_countries, tool_name="data360_rank_countries"),
        name="data360_rank_countries",
        serializer=_compact_aggregation_serializer,
    )
)

compare_countries = mcp.add_tool(
    Tool.from_function(
        instrument_mcp_tool(_compare_countries, tool_name="data360_compare_countries"),
        name="data360_compare_countries",
        serializer=_compact_aggregation_serializer,
    )
)



from data360.templates.render import render_template


@mcp.resource("ui://data360-chart/index.html")
def data360_chart_html() -> str:
    """HTML resource for the Data360 self-contained Vega-Lite chart viewer Custom HTML app."""
    vega_js, vega_lite_js, vega_embed_js, vega_interpreter_js = get_cached_vega_libs()
    return render_template(
        "data360_chart.jinja2",
        vega_js=vega_js,
        vega_lite_js=vega_lite_js,
        vega_embed_js=vega_embed_js,
        vega_interpreter_js=vega_interpreter_js,
    )


@mcp.tool(
    name="data360_interactive_choices",
    app=AppConfig(resource_uri="ui://data360-choice/index.html", prefers_border=False),
)
async def data360_interactive_choices(
    prompt: str,
    options: list[str],
    title: Optional[str] = None,
) -> ToolResult:
    """Present the user with a set of options to choose from using a custom HTML renderer.

    Always call this tool to provide follow-ups and elicitations based on the natural flow of the
    conversation and the type of information being discussed. Your goal is to anticipate the
    user's next question or provide an easy way to steer a broad topic.

    Call this tool in the following scenarios:

    1. Single Follow-up (1 choice):
       - The "Obvious Next Step": When there is one highly logical action to take after your response.
         For example, if you explain a mathematical concept, offer a follow-up to walk through a practical example.
       - Deep Dives into Jargon: If your response introduces a complex technical term or a new concept,
         offer a single follow-up to explain that specific term so the main response does not get too cluttered.
       - Launching Interactive Tools: If you mention that you can build a widget or run a simulation,
         provide a single button to let the user trigger that specific interactive element directly.

    2. Multiple Choices (2+ choices):
       - Broad Overviews & Branching Paths: When you give a high-level summary of a massive topic,
         use this to let the user choose exactly which sub-category or "branch" you want to zoom in on next.
       - Disambiguation (Clarifying Intent): If the user's request is open-ended or could be interpreted in
         a few different ways, present options so the user can clarify exactly which direction they meant to take.
         Examples:
         * GDP/Metric variant: "Real GDP per capita (constant 2015 US$)" vs "Nominal GDP per capita (current US$)"
         * Timeframe/Year range: "Latest available year" vs "Historical trend (last 10 years)" vs "Specify a custom range"
         * Breakdown/Disaggregation: "Total economy average" vs "Break down by gender (Male vs Female)" vs "Break down by geographic area (Urban vs Rural)"
       - Menus and Brainstorming: When generating lists of ideas (like different programming frameworks,
         design patterns, or troubleshooting steps), use this to act like a clickable menu, letting the user
         instantly select the one you want to explore.

    3. Non-exhaustive Lists (CRITICAL):
       - If you present a list of choices that is not exhaustive (such as listing a few popular countries,
         specific years, indicator variants, or breakdowns), you MUST always dynamically include a customizable
         option as the last item in the options list.
         Examples:
         * Country list: options=["Kenya", "Nigeria", "South Africa", "United States", "India", "Specify another country..."]
         * Year list: options=["2024 (latest)", "Last 5 years", "Last 10 years", "Specify a custom range"]
         * Breakdowns: options=["Total Average", "Breakdown by Gender", "Other (specify)"]

    Essentially, surface these components whenever you can save the user the effort of typing out the
    logical next prompt, or when the conversation has reached a crossroads and you need the user to choose
    the direction.

    Args:
        prompt: The question or decision to present to the user.
        options: List of options the user can choose from.
        title: Optional heading for the card.
    """
    payload = {
        "prompt": prompt,
        "options": options,
        "title": title or "Choose an Option"
    }
    return ToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structured_content=payload,
    )
