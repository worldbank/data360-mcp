"""Resources for the Data360 MCP Server.

These resources provide static context to help LLMs understand the Data360 system.
Includes ``data360://agent-recipe`` for host integrators (LangGraph / data360-mcp-agent).
"""

import json
from datetime import datetime

from fastmcp.apps import AppConfig, ResourceCSP
from data360.log_sanitize import sanitize_for_log
from data360.providers import get_database_mapping

from ._server_definition import mcp
from .agent_recipe import AGENT_RECIPE_MARKDOWN

# System prompt with chain-of-thought guidance for chatbot integration
from .prompts import SYSTEM_PROMPT


CODELISTS = {
    "auto_resolved": {
        "description": (
            "The pipeline auto-resolves these dimensions from the extdataportal codelist. "
            "series_labels is NOT required for these dimensions."
        ),
        "source": "https://extdataportal.worldbank.org/api/data360/metadata/codelist",
        "dimensions": {
            "COMP_BREAKDOWN_1": "5 191 indicator-subtype codes (e.g. WGI_EST, IPC_IPC_PHASE3, WEF_TTDI_RNK)",
            "COMP_BREAKDOWN_2": "Same pool as COMP_BREAKDOWN_1",
            "UNIT_MEASURE": "769 unit codes auto-resolved in Y-axis labels and subtitles",
            "SEX": "7 codes: F=Female, M=Male, _T=Total, _O=Other, _U=Unknown, _Z=Not applicable",
            "AGE": "173 codes: _T=All ages, Y15T24=15-24 years, Y_GE25=25+ years, etc.",
            "URBANISATION": "16 codes: URB=Urban area, RUR=Rural area, CITY=City, VILL=Village, etc.",
            "FREQ": "34 codes: A=Annual, M=Monthly, Q=Quarterly, etc.",
        },
    },
    "manual_override": {
        "description": (
            "Provide series_labels only to shorten or rename auto-resolved labels, "
            "e.g. to show 'Estimate' instead of 'Governance estimate (approx. -2.5 to +2.5)'."
        ),
        "example": {"WGI_EST": "Estimate", "WGI_SC": "Score", "WGI_SE": "Std. Error"},
    },
    "geographic": {
        "description": "REF_AREA groups resolved via GroupHierarchyManager (FMR H_REF_AREA_GROUPS)",
        "individual_countries": "532 codes — resolved automatically to country names",
        "groups": "147 group codes (REGION, INCOME, LENDING, CONTINENT) — use expand_country_group",
    },
}


METADATA_FIELDS = {
    "fields": {
        "methodology": {
            "description": "How the indicator is calculated/measured",
            "use_when": ["how is it calculated", "calculation method", "methodology"],
        },
        "statistical_concept": {
            "description": "Statistical definition and conceptual framework",
            "use_when": ["statistical concept", "what does it measure", "definition"],
        },
        "definition_long": {
            "description": "Full description of the indicator",
            "use_when": ["what is", "describe", "explanation"],
        },
        "limitation": {
            "description": "Known data limitations and caveats",
            "use_when": ["limitations", "caveats", "data quality", "issues"],
        },
        "relevance": {
            "description": "Policy relevance and why this indicator matters",
            "use_when": ["why important", "relevance", "policy implications"],
        },
        "aggregation_method": {
            "description": "How values are aggregated (Sum, Average, etc.)",
            "use_when": ["aggregation", "how combined", "sum or average"],
        },
        "periodicity": {
            "description": "Data frequency (Annual, Monthly, etc.)",
            "use_when": ["frequency", "how often", "periodicity"],
        },
        "time_periods": {
            "description": "Nominal time range (may have gaps)",
            "use_when": ["time range", "years available", "coverage"],
            "note": "Call get_disaggregation for actual available years",
        },
        "ref_country": {
            "description": "List of countries with data",
            "use_when": ["countries", "coverage", "available for"],
        },
        "sources_note": {
            "description": "Information about data sources",
            "use_when": ["source", "where from", "data provider"],
        },
    }
}


DATA_FILTERS = {
    "workflow": "Call get_disaggregation first to see available values for each filter",
    "supported_filters": {
        "timePeriodFrom": {"description": "Start year", "example": "2020"},
        "timePeriodTo": {"description": "End year", "example": "2023"},
        "REF_AREA": {
            "description": "Country code(s). Use comma-separated for multiple.",
            "example": "KEN,TZA",
        },
        "SEX": {"values": ["F", "M", "_T", "_O", "_U", "_Z"]},
        "AGE": {
            "description": "173 age codes — common ones below; use get_disaggregation for indicator-specific values",
            "common_values": ["_T", "Y15T24", "Y15T29", "Y30T59", "Y_GE25", "Y_GE60", "Y18T65"],
        },
        "URBANISATION": {
            "values": ["_T", "URB", "RUR", "CITY", "VILL", "DTOW", "TSUB", "STOW", "SUBU", "SURB", "LURB", "_O", "_Z"],
        },
    },
    "excluded_filters": {"FREQ": "DO NOT USE - breaks queries"},
    "important": "Check TIME_PERIOD in disaggregation for actual available years (may have gaps)",
}


DATA_SCHEMA = {
    "description": "Data rows are prefiltered to only include relevant fields. Always present: 5 core fields. Conditionally present: disaggregation fields when their values are non-trivial.",
    "core_fields": {
        "obs_value": "The numeric data value.",
        "time_period": "Date or year of the observation (e.g., '2023', '2024-07-01').",
        "ref_area": "Country or region code (e.g., 'KEN').",
        "unit_measure": "Unit of measurement (e.g., 'PT', 'USD_K_2015', 'PS').",
        "claim_id": "Verification hash for data provenance.",
    },
    "conditional_fields": {
        "description": "Included only when values carry real disaggregation (not _T total or _Z not-applicable).",
        "sex": "Gender breakdown ('F', 'M'). Present in WB_HCP, WB_SSGD, WB_GS.",
        "age": "Age group ('Y15T24', 'Y18T65', etc.). Present in WB_SSGD, OECD_IDD.",
        "urbanisation": "Urban/Rural ('URB', 'RUR'). Present in WB_SSGD.",
        "comp_breakdown_1": "Indicator subtype (e.g., IPC phase period, OECD indicator type, WEF rank/value/score).",
        "comp_breakdown_2": "Secondary breakdown (e.g., IPC phase level, OECD income definition).",
    },
    "visualization_guidance": "When calling get_viz_spec(relevant_fields=...), prioritize 'time_period' and 'obs_value'. Include 'ref_area' or dimensions like 'sex' only for comparison/grouping.",
}


SEARCH_USAGE = {
    "basic_search": {
        "example": "data360_search_indicators(query='poverty', limit=10)",
        "note": "Uses default select_fields",
    },
    "enriched_search": {
        "example": "data360_search_indicators(query='poverty', limit=5, select_fields=['idno', 'name', 'database_id', 'definition_long', 'periodicity', 'time_periods', 'dimensions'])",
        "note": "Use when LLM needs to pick best indicator",
    },
    "indicator_selection_workflow": [
        "1. Use enriched search with select_fields for extra coverage info",
        "2. Call get_disaggregation to check TIME_PERIOD and REF_AREA",
        "3. Pick indicator based on coverage, time range, and relevance",
    ],
    "warning": "DO NOT use odata_options - it is deprecated",
}

K360_NARRATIVE_STYLE = {
    "sections": ["Data", "Analysis", "Note", "Sources"],
    "required_behavior": [
        "Ground every statement in tool evidence or content packet fields.",
        "Use concise markdown suitable for analyst and policy audiences.",
        "When chart outputs exist, describe what each chart conveys in 1-2 sentences.",
        "If no data is available, clearly state the gap and suggest a narrower follow-up query.",
    ],
    "optional_claim_tags": {
        "enabled_by": "include_claim_tags=true",
        "format": "<claim id=\"short-id\">numeric statement</claim>",
    },
}


@mcp.resource("data360://system-prompt")
async def system_prompt_resource() -> str:
    """System prompt with chain-of-thought guidance for chatbot integration."""
    return SYSTEM_PROMPT


@mcp.resource("data360://agent-recipe")
async def agent_recipe_resource() -> str:
    """How to compose MCP resources + named prompts for LangGraph / ``data360-mcp-agent``."""
    return AGENT_RECIPE_MARKDOWN


@mcp.resource("data360://context")
async def context_resource() -> str:
    """Runtime context including current date. Read this to know today's date."""
    return json.dumps(
        {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_year": datetime.now().year,
            "note": "Use current_year to calculate 'last N years' queries",
        },
        indent=2,
    )


@mcp.resource("data360://databases")
async def databases_resource() -> str:
    """List of available Data360 databases."""
    db_mapping = await get_database_mapping()
    formatted = {"databases": [{"id": k, "name": v} for k, v in db_mapping.items()]}
    return json.dumps(formatted, indent=2)


@mcp.resource("data360://codelists")
async def codelists_resource() -> str:
    """Codelist reference information (global and indicator-level)."""
    return json.dumps(CODELISTS, indent=2)


@mcp.resource("data360://metadata-fields")
async def metadata_fields_resource() -> str:
    """Metadata field mapping for smart routing based on user questions."""
    return json.dumps(METADATA_FIELDS, indent=2)


@mcp.resource("data360://data-filters")
async def data_filters_resource() -> str:
    """Available data filters and usage guidance."""
    return json.dumps(DATA_FILTERS, indent=2)


@mcp.resource("data360://data-schema")
async def data_schema_resource() -> str:
    """Standard data schema and column definitions for visualization."""
    return json.dumps(DATA_SCHEMA, indent=2)


@mcp.resource("data360://search-usage")
async def search_usage_resource() -> str:
    """Search tool usage guidance."""
    return json.dumps(SEARCH_USAGE, indent=2)


@mcp.resource("data360://k360-narrative-style")
async def k360_narrative_style_resource() -> str:
    """Narrative formatting contract for K360 staged agent hosts."""
    return json.dumps(K360_NARRATIVE_STYLE, indent=2)


# ---------------------------------------------------------------------------
# Chart Grammar Resource — grammar-of-graphics decision rules
# ---------------------------------------------------------------------------

CHART_GRAMMAR = """# Data360 Chart Grammar — Decision Rules for Visualization

This resource teaches you how to reason about data shapes and select the correct
chart strategy. The visualization engine applies these rules automatically, but
understanding them lets you make better upstream decisions (which tool to call,
what chart_type to pass, and how to narrate the result).

## 1. Strategy Selection Rules

The engine selects a strategy based on the **data shape** after fetching:

| Condition | Strategy | Chart type |
|-----------|----------|-----------|
| 1 indicator, temporal, 1–8 countries | `temporal_single` | Line chart (color=country) |
| 1 indicator, temporal, >8 countries, no breakdowns | `heatmap` | Heatmap matrix (country × year) |
| 1 indicator, single year, ≤8 countries | `cross_sectional` | Horizontal bar chart |
| 1 indicator, single year, >8 countries | `distribution` | Strip/beeswarm chart |
| 1 indicator, breakdown dimensions present | `breakdown_comparison` or `small_multiples` | Grouped bar or faceted panels |
| 2+ indicators, temporal, 1 country | `temporal_multi_indicator` | Layered lines or stacked panels |
| 2+ indicators, single year, multiple countries | `scatter` or `cross_sectional` | Scatter or grouped bar |
| Composition data (parts sum to ~100%) | `stacked_area` or `stacked_bar` | Stacked marks |

## 2. Layout Composition Rules (Multi-Indicator)

When comparing 2+ indicators, the engine decides whether to use a **single shared
panel** or **vertically stacked panels with independent Y-axes**.

The decision is based on the `data_profile.scale_compatibility` in the tool response:

| Condition | Layout | Reason |
|-----------|--------|--------|
| Same scale type AND value ratio ≤ 10× | Single panel, shared Y-axis | Values are comparable |
| Same scale type BUT value ratio > 10× | vconcat panels, independent Y-axes | Large magnitude difference distorts one series |
| Different scale types (e.g. % vs USD) | vconcat panels, independent Y-axes | Incomparable units |
| All values are percentages in [0, 100] | Single panel | Natural shared range |

**How to use**: After calling `data360_get_multi_indicator_viz_spec`, read
`data_profile.scale_compatibility.can_share_axis` and `data_profile.indicators`
to understand the layout decision and narrate it to the user.

## 3. Encoding Grammar

The engine maps data dimensions to visual channels:

| Data dimension | Vega-Lite encoding | When used |
|---|---|---|
| year / time_period | `x` (temporal) | Time-series charts |
| country | `color` (nominal) | Multi-country lines; `y` for cross-sectional bars |
| value / obs_value | `y` (quantitative) | Always the measurement axis |
| indicator | `color` (nominal) | Multi-indicator overlays |
| breakdown dim (sex, age, etc.) | `color` or `facet` | Disaggregation present |

## 4. Data Profile Fields

Every viz tool response now includes a `data_profile` with these sections:

- **indicators**: Per-indicator value ranges (min/max/median), unit codes, scale
  types (percentage/currency/persons/index), and whether values are proportions.
- **scale_compatibility** (multi-indicator): Whether indicators can share a Y-axis.
- **structure**: Country list, year range, temporal density (dense/moderate/sparse).
- **breakdowns**: Available disaggregation dimensions with actual values and meanings.
- **composition_hint**: Whether data looks like parts-of-a-whole (suitable for stacked).

Use these fields to:
1. **Narrate accurately**: "GDP ranges from $1,200 to $63,000" instead of guessing.
2. **Assess chart quality**: If `temporal_density` is "sparse", note potential gaps.
3. **Suggest alternatives**: If `composition_hint.suitable_for_stacked` is true,
   suggest a stacked area view.

## 5. When NOT to Pass chart_type

Let the engine auto-select when:
- The data shape is unambiguous (single indicator, clear temporal or cross-sectional)
- You are unsure which chart fits the data

Only override chart_type when:
- The user explicitly asked for a style ("show me a bar chart")
- You need a specific multi-indicator layout ("scatter", "connected_scatter")

## 6. Tool Selection

| Scenario | Tool |
|----------|------|
| 1 indicator | `data360_get_viz_spec` |
| 2–4 indicators to compare | `data360_get_multi_indicator_viz_spec` |
| Need to summarize without a chart | `data360_summarize_data` |
"""


@mcp.resource("data360://viz/chart-grammar")
async def chart_grammar_resource() -> str:
    """Grammar-of-graphics decision rules for Data360 visualization.

    Teaches the LLM how to reason about data shapes, scale compatibility,
    encoding rules, and layout decisions. Read this resource to understand
    how the visualization engine selects strategies and how to interpret
    the data_profile in tool responses.
    """
    return CHART_GRAMMAR


from data360.templates.render import render_template


@mcp.resource(
    "ui://data360/vega-lite-renderer.html{?spec}",
    app=AppConfig(
        csp=ResourceCSP(
            connect_domains=["*"],
            resource_domains=[
                "http://localhost:*",
                "http://127.0.0.1:*",
                "'unsafe-eval'",
            ],
        )
    ),
)
async def vega_lite_renderer(spec: str | None = None) -> str:
    """HTML renderer template for Vega-Lite v6 charts."""
    from data360.config import get_mcp_server_settings

    settings = get_mcp_server_settings()
    port = settings.port or 8021
    server_base = getattr(settings, "server_base_url", None) or f"http://localhost:{port}"
    return render_template("vega_lite_renderer.jinja2", server_base=server_base, pre_loaded_spec=spec)


@mcp.resource(
    "ui://data360-choice/index.html",
    app=AppConfig(
        csp=ResourceCSP(
            connect_domains=["*"],
            resource_domains=[
                "https://fonts.googleapis.com",
                "https://fonts.gstatic.com",
            ],
        )
    ),
)
async def data360_choice_html() -> str:
    """HTML resource for the Data360 self-contained choice Custom HTML app."""
    return render_template("data360_choice.jinja2")



import os
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from starlette.exceptions import HTTPException

class CORSStaticFiles(StaticFiles):
    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await super().__call__(scope, receive, send)
            return

        if scope["method"] == "OPTIONS":
            response = Response(
                "OK",
                status_code=200,
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                    "Access-Control-Allow-Headers": "*",
                }
            )
            await response(scope, receive, send)
            return

        async def cors_send(message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                has_origin = any(h[0].lower() == b"access-control-allow-origin" for h in headers)
                if not has_origin:
                    headers.append((b"access-control-allow-origin", b"*"))
                    headers.append((b"access-control-allow-methods", b"GET, HEAD, OPTIONS"))
                    headers.append((b"access-control-allow-headers", b"*"))
                message["headers"] = headers
            await send(message)

        await super().__call__(scope, receive, cors_send)

    async def get_response(self, path: str, scope) -> Response:
        try:
            return await super().get_response(path, scope)
        except HTTPException as exc:
            return JSONResponse(
                {"detail": exc.detail},
                status_code=exc.status_code,
                headers=exc.headers
            )




@mcp.custom_route("/debug-log", methods=["POST", "OPTIONS"])
async def debug_log(request: Request) -> Response:
    if request.method == "OPTIONS":
        return Response(
            "OK",
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    try:
        body = await request.json()
        print(f"\n[IFRAME DEBUG LOG] {sanitize_for_log(body)}\n", flush=True)
        return JSONResponse(
            {"status": "ok"},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        print(f"Error reading debug log: {e}", flush=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=400,
            headers={"Access-Control-Allow-Origin": "*"}
        )
