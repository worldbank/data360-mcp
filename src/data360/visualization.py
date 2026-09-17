"""
Visualization generation for Data360 data.

Fetches series via the Data360 API (``data360.api.get_data_api_url``), builds
Vega-Lite specifications, then persists them either through the optional Charts API
(``charts_api_url``) or as static JSON under ``static/viz_specs/``.

**Public tools**

- ``get_viz_spec`` — one indicator. Uses ``data360.viz_config`` strategy dispatch
  to build validated Vega-Lite v5 specs directly.
- ``get_multi_indicator_viz_spec`` — two to four indicators; merges frames and
  dispatches multi-series strategies (scatter, layered lines, connected scatter, etc.).

Return shape for both: ``{"url": str|None, "error": str|None, ...}`` plus optional
``database_id``, ``database_name``, ``indicator_id``, ``indicator_name``, optional
``strategy`` / ``reason``, and optional preformatted ``source_line`` /
``subtitle_line`` for clients that only render strings. Before any HTTP
or ``json.dump``, specs are passed through ``_vega_spec_to_json_safe`` so
``data.values`` never contains raw ``pandas.Timestamp`` / numpy scalars that would
break JSON encoding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import uuid
from datetime import date, datetime
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx
import numpy as np
import pandas as pd

from data360 import viz_config
from data360.config import get_mcp_server_settings
from data360.http_client import get_shared_httpx_client
from data360.providers import get_database_mapping

_logger = logging.getLogger(__name__)

VizResult = dict[str, Any]


class VizInsufficientDataError(Exception):
    """Raised when data returned from the API is too sparse to render a meaningful chart.

    Prefer a clear, actionable error over a misleading visualization.
    """

# Disaggregation dimensions considered during viz data cleaning and encoding.
# Mirrors _DISAGG_DIMS_TO_DETECT from api.py but in lowercase (post-column-rename).
# Single source of truth for the viz pipeline: _clean_single_df, color dim
# detection, and multi-indicator join keys all reference this tuple.
#
# unit_measure is included last and uses a separate sentinel check (_UNIT_MEASURE_TRIVIAL)
# because its "total" sentinel is 'U' (Unitless), not '_T'. When unit_measure has
# 2+ distinct non-trivial values (e.g. Persons + Percentage for IPC), it is kept
# as a discriminating column so select_strategy can route correctly instead of
# silently mixing incompatible values on the same Y-axis.
_UNIT_MEASURE_TRIVIAL: frozenset[str] = frozenset({"U", ""})
_VIZ_DISAGG_DIMS: tuple[str, ...] = (
    "sex",
    "age",
    "urbanisation",
    "residence",
    "comp_breakdown_1",
    "comp_breakdown_2",
    "comp_breakdown_3",
    "unit_measure",
)

# Fallback attribution string when neither database nor indicator name is available.
# Mirrors DATA360_CHART_SOURCE_FALLBACK in the TypeScript frontend.
_SOURCE_FALLBACK = "World Bank — Data360"


def _unit_measure_for_formatting(
    raw_unit: str | None, resolved_unit_label: str | None = None
) -> str | None:
    """Return a formatting-friendly unit token for viz_config formatters."""
    raw = (raw_unit or "").strip()
    label = (resolved_unit_label or "").strip()
    raw_norm = raw.upper()
    label_norm = label.upper()

    if raw_norm == "PT" or ("PERCENT" in label_norm and "PERSON" not in label_norm):
        return "%"
    if (
        "$" in raw_norm
        or "USD" in raw_norm
        or "$" in label_norm
        or "USD" in label_norm
        or "DOLLAR" in label_norm
    ):
        return "USD"
    # If the label has descriptive keywords like PROPORTION or SHARE, return the label
    # so downstream rules can detect proportion formatting.
    if "PROPORTION" in label_norm or "SHARE" in label_norm:
        return label
    if raw:
        return raw
    if label:
        return label
    return None


# ============================================================================
# STORAGE HELPERS
# ============================================================================


def save_specs_to_static(vl_spec: dict) -> str:
    """Persist ``vl_spec`` as JSON and return a URL the **browser** can fetch.

    The frontend (or another client) loads this file over HTTP; it is not read
    only inside Docker. ``WEBSITE_HOSTNAME`` should be a host:port the user's browser
    can resolve; when unset we default to ``localhost`` and the MCP server port from
    ``data360.config.get_mcp_server_settings()``.
    """
    spec_id = str(uuid.uuid4())
    server_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(server_dir, "..", ".."))
    specs_dir = os.path.join(project_root, "static", "viz_specs")
    os.makedirs(specs_dir, exist_ok=True)
    vega_path = os.path.join(specs_dir, f"{spec_id}_vega.json")
    with open(vega_path, "w") as f:
        json.dump(vl_spec, f, indent=2)
    base_url = os.environ.get(
        "WEBSITE_HOSTNAME", f"http://localhost:{get_mcp_server_settings().port}"
    )
    return f"{base_url}/static/viz_specs/{spec_id}_vega.json"


async def post_spec_to_charts_api(vl_spec: dict) -> str:
    """POST a Vega-Lite spec to the external Charts API (JSON body).

    The service expects standard Vega-Lite keys (``$schema``, ``data``, ``mark``,
    ``encoding``, …). Many deployments require a non-empty ``title``; we set a default
    if missing. On success, returns a chart URL from ``Location``, response JSON
    ``url`` / ``id``, or the configured API base as a last resort.
    """
    settings = get_mcp_server_settings()
    url = settings.charts_api_url
    if not url:
        raise ValueError("charts_api_url is not configured")
    payload = dict(vl_spec)
    # Charts API often rejects payloads without title
    payload.setdefault("title", vl_spec.get("title") or "Generated Visualization")
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if settings.charts_api_token:
        headers["Authorization"] = f"Bearer {settings.charts_api_token}"
    client = get_shared_httpx_client()
    response = await client.post(
        url,
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    location = response.headers.get("Location")
    if location:
        return (
            location
            if location.startswith("http")
            else f"{url.rsplit('/', 1)[0]}/{location.lstrip('/')}"
        )
    body = response.json() if response.content else {}
    if isinstance(body, dict):
        if body.get("url"):
            return body["url"]
        if body.get("id"):
            return f"{url.rstrip('/')}/{body['id']}"
    return url


def _vega_spec_to_json_safe(obj: object) -> object:
    """Recursively convert a Vega-Lite spec tree to JSON-serializable Python types.

    Covers pandas/numpy scalars and datetimes that can appear in ``data.values`` or
    elsewhere after ``DataFrame.to_dict`` (including merged dtypes and edge cases the
    DataFrame-only sanitizer misses).
    """
    if obj is None:
        return None
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat() if pd.notna(obj) else None
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(pd.Timestamp(obj))
    if isinstance(obj, (np.integer, np.floating)):
        if pd.isna(obj):
            return None
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _vega_spec_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_vega_spec_to_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return _vega_spec_to_json_safe(obj.item())
        except (ValueError, AttributeError, TypeError):
            pass
    raise TypeError(
        f"Vega-Lite spec contains unsupported type for JSON: {type(obj).__name__}"
    )


_UNSET = object()  # sentinel for "not provided" (distinct from None which means "disable")


async def _store_spec(
    vl_spec: dict,
    charts_api_url_override: object = _UNSET,
) -> str:
    """Persist a Vega-Lite dict: Charts API when configured, else static file.

    Always runs ``_vega_spec_to_json_safe`` first so httpx ``json=`` and
    ``json.dump`` cannot fail on non-JSON-native types in embedded data.

    Args:
        vl_spec: The Vega-Lite spec dict to persist.
        charts_api_url_override: When provided, overrides the ``charts_api_url``
            from settings *for this call only*, without mutating the shared singleton.
            Pass ``None`` to force local-only storage even when a Charts API URL is
            configured. Use the sentinel ``_UNSET`` (default) to read from settings.
    """
    safe = _vega_spec_to_json_safe(vl_spec)
    if not isinstance(safe, dict):
        raise TypeError("Vega spec must serialize to a JSON object")

    settings = get_mcp_server_settings()
    if charts_api_url_override is _UNSET:
        charts_url = settings.charts_api_url
    else:
        charts_url = charts_api_url_override  # type: ignore[assignment]
    env = settings.env or "local"
    is_prod = env.lower() in ("prod", "production")

    if charts_url:
        try:
            url = await post_spec_to_charts_api(safe)
            # In production, do not save locally if Charts API is active and succeeded
            if is_prod:
                return url
            # Otherwise (local/dev), still save locally for caching/debugging
            save_specs_to_static(safe)
            return url
        except Exception as e:
            _logger.warning("Charts API store failed, falling back to static local: %s", repr(e))

    # Fallback to local static file
    return save_specs_to_static(safe)


def _ok(
    url: str,
    warning: str | None = None,
    *,
    source_attribution: dict[str, str] | None = None,
    strategy: str | None = None,
    reason: str | None = None,
    dimensions: dict[str, list] | None = None,
    data_summary: dict | None = None,
    data_profile: dict | None = None,
    spec: dict | None = None,
) -> VizResult:
    r: VizResult = {"url": url, "error": None}
    if warning:
        r["warning"] = warning
    if source_attribution:
        for key, val in source_attribution.items():
            if val:
                r[key] = val
    if strategy:
        r["strategy"] = strategy
    if reason:
        r["reason"] = reason
    if dimensions:
        r["dimensions"] = dimensions  # type: ignore[assignment]
    if data_summary:
        r["data_summary"] = data_summary  # type: ignore[assignment]
    if data_profile:
        r["data_profile"] = data_profile  # type: ignore[assignment]
    if spec is not None:
        r["spec"] = spec
    attrib_for_line = {
        k: str(v)
        for k, v in r.items()
        if k
        in (
            "database_id",
            "database_name",
            "indicator_id",
            "indicator_name",
        )
        and v
    }
    r["source_line"] = _format_source_line_from_attribution(attrib_for_line)
    strat_v = r.get("strategy")
    reas_v = r.get("reason")
    sub = _format_subtitle_line(
        warning,
        strat_v if isinstance(strat_v, str) else None,
        reas_v if isinstance(reas_v, str) else None,
    )
    if sub:
        r["subtitle_line"] = sub
    return r


def _err(msg: str) -> VizResult:
    return {"url": None, "error": msg}


# Dimensions surfaced to the LLM in the tool response when they discriminate rows.
_SURFACE_DIMS: list[str] = [
    "unit_measure",
    "sex",
    "age",
    "urbanisation",
    "comp_breakdown_1",
    "comp_breakdown_2",
]


def _extract_dimension_summary(df: pd.DataFrame) -> dict[str, list]:
    """Return dimension -> sorted distinct values for non-trivial dims in *df*.

    Only dimensions with 2+ distinct values after excluding trivial sentinels
    ('_T', '_Z' for totals, 'U' for Unitless, '') are included. Values are
    expected to already be resolved to human-readable labels when called.
    The result is included in the tool response so the LLM knows what is in
    the data without needing to parse the Vega-Lite spec.
    """
    _trivial: frozenset[str] = frozenset({"_T", "_Z", "U", ""})
    result: dict[str, list] = {}
    for dim in _SURFACE_DIMS:
        if dim not in df.columns:
            continue
        vals = sorted(
            str(v) for v in df[dim].dropna().unique() if str(v) not in _trivial
        )
        if len(vals) > 1:
            result[dim] = vals
    return result


def _build_data_summary(df: pd.DataFrame) -> dict:
    """Compact structural summary of the plotted DataFrame for LLM narration.

    Supplements ``_extract_dimension_summary`` (which covers categorical dims)
    with quantitative/shape information: how many rows, what year range, which
    countries, and the numeric value range. All fields are optional — missing
    columns are silently skipped.

    Kept deliberately small: no raw rows, no per-dimension statistics. The goal
    is to give the LLM enough to narrate the chart without overwhelming context.
    """
    out: dict = {"shape": list(df.shape)}
    if "year" in df.columns:
        years = df["year"].dropna().astype(str)
        # Strip time components if ISO timestamps leaked through
        years = years.str[:4]
        unique_years = sorted(years.unique())
        if unique_years:
            out["year_range"] = [unique_years[0], unique_years[-1]]
    if "country" in df.columns:
        countries = sorted(df["country"].dropna().unique().tolist())
        if countries:
            out["countries"] = countries
    if "value" in df.columns:
        vals = pd.to_numeric(df["value"], errors="coerce").dropna()
        if not vals.empty:
            out["value"] = {
                "min": round(float(vals.min()), 4),
                "max": round(float(vals.max()), 4),
                "has_negatives": bool((vals < 0).any()),
            }
    return out


# ---------------------------------------------------------------------------
# Scale-type inference from unit codes / labels
# ---------------------------------------------------------------------------

_PERCENTAGE_TOKENS: frozenset[str] = frozenset({
    "PT", "PC", "PERCENT", "PERCENTAGE", "%", "RATE", "SHARE", "PROPORTION",
    "PC_GDP", "PC_GNI", "PC_TOT",
})

_CURRENCY_TOKENS: frozenset[str] = frozenset({
    "$", "USD", "CURRENCY", "DOLLARS", "LCU", "EUR", "GBP",
})

_PERSONS_TOKENS: frozenset[str] = frozenset({
    "PS", "PEOPLE", "PERSONS", "COUNT", "HEADCOUNT", "NR", "NUMBER",
})


def _infer_scale_type(unit_code: str | None, unit_label: str | None = None) -> str:
    """Infer a semantic scale type from a unit code or label string.

    Returns one of: ``"percentage"``, ``"currency"``, ``"persons"``, ``"index"``.
    """
    tokens: set[str] = set()
    import re
    for raw in (unit_code, unit_label):
        if raw:
            normalised = raw.upper().strip()
            tokens.add(normalised)
            # Remove all punctuation before splitting
            clean_str = re.sub(r'[^\w\s]', ' ', normalised)
            tokens.update(clean_str.split())

    if tokens & _PERSONS_TOKENS:
        return "persons"
    if tokens & _PERCENTAGE_TOKENS:
        return "percentage"
    if tokens & _CURRENCY_TOKENS:
        return "currency"
    return "index"


# ---------------------------------------------------------------------------
# Comprehensive data profile for LLM reasoning
# ---------------------------------------------------------------------------


def _build_data_profile(
    df: pd.DataFrame,
    *,
    scale_type: str | None = None,
    indicator_cols: list[str] | None = None,
    indicator_names: list[str] | None = None,
    units_raw: list[str] | None = None,
    units_label: list[str] | None = None,
    dim_name_labels: dict[str, str] | None = None,
) -> dict:
    """Build a comprehensive data profile for LLM charting reasoning.

    Computed **before routing**, so signals flow into strategy selection and
    spec building — not just post-hoc narration.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned DataFrame (post-_clean_single_df, post-country-mapping).
    scale_type : str | None
        Semantic unit type: ``"percentage"``, ``"currency"``, ``"persons"``,
        ``"index"``.  Derived from unit codes when omitted.
    indicator_cols : list[str] | None
        Wide-format column names for multi-indicator charts.
    indicator_names : list[str] | None
        Human-readable display names aligned with indicator_cols.
    units_raw : list[str] | None
        Raw unit codes, e.g. ``["PT", "USD_K_2015"]``.
    units_label : list[str] | None
        Resolved unit labels, e.g. ``["Percent", "Constant 2015 USD"]``.
    dim_name_labels : dict[str, str] | None
        API-sourced human-readable names for comp_breakdown_* columns.

    Sections returned
    -----------------
    indicators
        Per-indicator: name, unit, scale_type, value_range (with skewness,
        p25/p75, null_count/null_pct), all_positive, is_proportion.
    scale_compatibility   (multi-indicator only)
        Whether indicators can share a Y-axis.
    structure
        Year range, temporal density, gaps.
    coverage_quality
        Completeness pct, sparse countries (< 3 years), missing years.
    temporal_trend   (temporal data, single-indicator)
        Direction, total pct change, monotonicity, per-country when divergent.
    cross_country    (multi-country only)
        Spread, leader/laggard at latest year, convergence signal.
    breakdowns
        Disaggregation dimensions with actual values, cardinality, meaning,
        codelist name, and value_labels dict.
    composition_hint
        Whether data looks like parts-of-a-whole.
    """
    profile: dict[str, Any] = {}

    # ── helpers ────────────────────────────────────────────────────────────────
    def _series_stats(series: pd.Series) -> dict:
        """Full numeric stats including nulls, percentiles, skewness."""
        raw = pd.to_numeric(series, errors="coerce")
        null_count = int(raw.isna().sum())
        total = len(raw)
        null_pct = round(null_count / total * 100, 1) if total else 0.0
        clean = raw.dropna()
        if clean.empty:
            return {
                "min": None, "max": None, "median": None,
                "p25": None, "p75": None, "skewness": None,
                "null_count": null_count, "null_pct": null_pct,
            }
        try:
            skew = float(clean.skew())
            skew = round(skew, 3) if not pd.isna(skew) else None
        except Exception:
            skew = None
        return {
            "min": round(float(clean.min()), 4),
            "max": round(float(clean.max()), 4),
            "median": round(float(clean.median()), 4),
            "p25": round(float(clean.quantile(0.25)), 4),
            "p75": round(float(clean.quantile(0.75)), 4),
            "skewness": skew,
            "null_count": null_count,
            "null_pct": null_pct,
        }

    # ── 1.  Indicator profiles ─────────────────────────────────────────────────
    ind_profiles: list[dict] = []

    if indicator_cols and len(indicator_cols) >= 2:
        # Multi-indicator (wide-format columns)
        for i, col in enumerate(indicator_cols):
            if col not in df.columns:
                continue
            name = (indicator_names[i] if indicator_names and i < len(indicator_names) else col)
            unit_raw = (units_raw[i] if units_raw and i < len(units_raw) else None)
            unit_lbl = (units_label[i] if units_label and i < len(units_label) else None)
            sc = _infer_scale_type(unit_raw, unit_lbl)
            stats = _series_stats(df[col])
            clean = pd.to_numeric(df[col], errors="coerce").dropna()
            ind_profiles.append({
                "name": name,
                "unit_code": unit_raw or None,
                "unit_label": unit_lbl or None,
                "scale_type": sc,
                "value_range": stats,
                "all_positive": bool(clean.ge(0).all()) if not clean.empty else None,
                "is_proportion": sc == "percentage" and stats["max"] is not None and stats["max"] <= 100,
            })
    elif "value" in df.columns:
        # Single-indicator (long-format value column)
        sc = scale_type or _infer_scale_type(
            units_raw[0] if units_raw else None,
            units_label[0] if units_label else None,
        )
        stats = _series_stats(df["value"])
        clean = pd.to_numeric(df["value"], errors="coerce").dropna()
        ind_profiles.append({
            "name": (indicator_names[0] if indicator_names else None),
            "unit_code": (units_raw[0] if units_raw else None),
            "unit_label": (units_label[0] if units_label else None),
            "scale_type": sc,
            "value_range": stats,
            "all_positive": bool(clean.ge(0).all()) if not clean.empty else None,
            "is_proportion": sc == "percentage" and stats["max"] is not None and stats["max"] <= 100,
        })

    if ind_profiles:
        profile["indicators"] = ind_profiles

    # ── 2.  Scale compatibility (multi-indicator only) ─────────────────────────
    if len(ind_profiles) >= 2:
        maxes = [
            p["value_range"]["max"] for p in ind_profiles
            if p["value_range"].get("max") and p["value_range"]["max"] > 0
        ]
        mins_nonzero = [
            p["value_range"]["min"] for p in ind_profiles
            if p["value_range"].get("min") and p["value_range"]["min"] > 0
        ]
        all_same_unit = len(set(p.get("unit_code") for p in ind_profiles if p.get("unit_code"))) <= 1
        all_same_scale = len(set(p["scale_type"] for p in ind_profiles)) == 1

        if maxes and mins_nonzero:
            ratio = max(maxes) / min(mins_nonzero)

            # Bounded percentages relaxation
            is_percentage = (ind_profiles[0]["scale_type"] == "percentage") if ind_profiles else False
            maxes_val = [p["value_range"]["max"] for p in ind_profiles if p["value_range"]["max"] is not None]
            any_gt_1 = any(m > 1.0 for m in maxes_val)
            all_lte_1 = all(m <= 1.0 for m in maxes_val)
            same_numeric_scale = any_gt_1 or all_lte_1

            if is_percentage and same_numeric_scale:
                thresh = 100.0
            else:
                thresh = 10.0

            can_share = all_same_scale and ratio <= thresh
            reason_parts = []
            if all_same_scale:
                reason_parts.append(f"same scale type ({ind_profiles[0]['scale_type']})")
            else:
                types = ", ".join(p["scale_type"] for p in ind_profiles)
                reason_parts.append(f"different scale types ({types})")
            reason_parts.append(f"value ratio {ratio:.1f}x")
            reason_parts.append(f"{'within' if ratio <= thresh else 'exceeds'} {int(thresh)}x threshold")
            profile["scale_compatibility"] = {
                "same_unit": all_same_unit,
                "same_scale_type": all_same_scale,
                "max_min_ratio": round(ratio, 2),
                "can_share_axis": can_share,
                "reason": ", ".join(reason_parts),
            }
        else:
            profile["scale_compatibility"] = {
                "same_unit": all_same_unit,
                "same_scale_type": all_same_scale,
                "max_min_ratio": None,
                "can_share_axis": False,
                "reason": "Could not compute ratio — zero or missing values",
            }

    # ── 3.  Structure (temporal + geographic shape) ────────────────────────────
    structure: dict[str, Any] = {}
    unique_years: list[str] = []
    if "country" in df.columns:
        countries = sorted(df["country"].dropna().unique().tolist())
        structure["countries"] = countries
        structure["country_count"] = len(countries)
    if "year" in df.columns:
        years_str = df["year"].dropna().astype(str).str[:4]
        unique_years = sorted(years_str.unique().tolist())
        if unique_years:
            structure["year_range"] = [unique_years[0], unique_years[-1]]
            structure["year_count"] = len(unique_years)
        if "country" in df.columns and not df.empty:
            try:
                ypc = df.groupby("country")["year"].nunique()
                avg = float(ypc.mean())
                structure["avg_years_per_country"] = round(avg, 1)
                expected = len(unique_years)
                structure["temporal_density"] = (
                    "dense" if avg >= expected * 0.8
                    else "moderate" if avg >= expected * 0.5
                    else "sparse"
                )
                structure["has_gaps"] = avg < expected
            except Exception:
                pass
    if structure:
        profile["structure"] = structure

    if "country" in df.columns and "year" in df.columns and not df.empty:
        try:
            # Use the first available value column for null filtering
            _val_col = indicator_cols[0] if indicator_cols else "value"
            _val_col = _val_col if _val_col in df.columns else None

            _subset = df
            if _val_col:
                _subset = df[pd.to_numeric(df[_val_col], errors="coerce").notna()]

            ypc = _subset.groupby("country")["year"].nunique()
            n_countries = len(ypc)
            expected_years = len(unique_years) if unique_years else (int(ypc.max()) if not ypc.empty else 1)
            _SPARSE_THRESHOLD_YEARS = 3 if expected_years >= 3 else 1

            total_expected = n_countries * expected_years
            total_filled = int(ypc.sum())
            completeness_pct = round(total_filled / total_expected * 100, 1) if total_expected > 0 else 100.0

            sparse = sorted(ypc[ypc < _SPARSE_THRESHOLD_YEARS].index.tolist())

            # Per-country missing years
            all_years_set = set(unique_years)
            per_country_missing: dict[str, list[str]] = {}
            if unique_years:
                for country, grp in _subset.groupby("country"):
                    country_years = set(grp["year"].astype(str).str[:4].unique())
                    missing = sorted(all_years_set - country_years)
                    if missing:
                        per_country_missing[str(country)] = missing

            cq: dict[str, Any] = {
                "total_expected_cells": total_expected,
                "total_filled_cells": total_filled,
                "completeness_pct": completeness_pct,
                "sparse_threshold_years": _SPARSE_THRESHOLD_YEARS,
                "sparse_countries": sparse,
            }
            if per_country_missing:
                cq["missing_years_by_country"] = per_country_missing
            profile["coverage_quality"] = cq
        except Exception:
            pass

    # ── 5.  Temporal trend (single-indicator, temporal data) ───────────────────
    if "year" in df.columns and "value" in df.columns and len(unique_years) >= 2:
        try:
            _df_t = df.copy()
            _df_t["_year_int"] = pd.to_numeric(df["year"].astype(str).str[:4], errors="coerce")
            _df_t["_val"] = pd.to_numeric(df["value"], errors="coerce")
            _df_t = _df_t.dropna(subset=["_year_int", "_val"])

            if not _df_t.empty and _df_t["_year_int"].nunique() >= 2:
                # Aggregate: mean across countries per year
                agg = _df_t.groupby("_year_int")["_val"].mean().sort_index()
                first_yr, last_yr = int(agg.index.min()), int(agg.index.max())
                first_val, last_val = float(agg.iloc[0]), float(agg.iloc[-1])

                pct_change = (
                    round((last_val - first_val) / abs(first_val) * 100, 1)
                    if first_val != 0 else None
                )

                diffs = agg.diff().dropna()
                pos = int((diffs > 0).sum())
                neg = int((diffs < 0).sum())
                if pos == 0 and neg == 0:
                    direction = "flat"
                elif pos >= len(diffs) * 0.75:
                    direction = "increasing"
                elif neg >= len(diffs) * 0.75:
                    direction = "decreasing"
                else:
                    direction = "volatile"

                monotonic = (pos == len(diffs)) or (neg == len(diffs))
                multi_country = "country" in df.columns and df["country"].nunique() > 1

                trend: dict[str, Any] = {
                    "direction": direction,
                    "pct_change_total": pct_change,
                    "monotonic": monotonic,
                }
                if multi_country:
                    trend["note"] = "Averaged across all countries"

                # Per-country directions when countries diverge
                if multi_country:
                    per_country: dict[str, str] = {}
                    for ctry, grp in _df_t.groupby("country"):
                        c_agg = grp.groupby("_year_int")["_val"].mean().sort_index()
                        if c_agg.nunique() < 2:
                            continue
                        c_diffs = c_agg.diff().dropna()
                        c_pos = int((c_diffs > 0).sum())
                        c_neg = int((c_diffs < 0).sum())
                        n = len(c_diffs)
                        if c_pos >= n * 0.75:
                            per_country[str(ctry)] = "increasing"
                        elif c_neg >= n * 0.75:
                            per_country[str(ctry)] = "decreasing"
                        else:
                            per_country[str(ctry)] = "volatile"
                    # Only include per-country if directions differ across countries
                    if len(set(per_country.values())) > 1:
                        trend["per_country"] = per_country

                profile["temporal_trend"] = {k: v for k, v in trend.items() if v is not None}
        except Exception:
            pass

    # ── 6.  Cross-country divergence (multi-country only) ─────────────────────
    if "country" in df.columns and "year" in df.columns and "value" in df.columns:
        try:
            _df_cc = df.copy()
            _df_cc["_val"] = pd.to_numeric(df["value"], errors="coerce")
            _df_cc["_year_int"] = pd.to_numeric(df["year"].astype(str).str[:4], errors="coerce")
            _df_cc = _df_cc.dropna(subset=["country", "_year_int", "_val"])

            if _df_cc["country"].nunique() >= 2 and not _df_cc.empty:
                latest_year = int(_df_cc["_year_int"].max())
                earliest_year = int(_df_cc["_year_int"].min())

                latest_agg = _df_cc[_df_cc["_year_int"] == latest_year].groupby("country")["_val"].mean()
                earliest_agg = _df_cc[_df_cc["_year_int"] == earliest_year].groupby("country")["_val"].mean()

                if len(latest_agg) >= 2:
                    spread_latest = round(float(latest_agg.max() - latest_agg.min()), 4)
                    leader = str(latest_agg.idxmax())
                    laggard = str(latest_agg.idxmin())

                    cc: dict[str, Any] = {
                        "latest_year": str(latest_year),
                        "spread": spread_latest,
                        "leader": {"country": leader, "value": round(float(latest_agg[leader]), 4)},
                        "laggard": {"country": laggard, "value": round(float(latest_agg[laggard]), 4)},
                    }

                    # Convergence: spread narrowing?
                    common = list(set(latest_agg.index) & set(earliest_agg.index))
                    if len(common) >= 2:
                        spread_earliest = float(
                            earliest_agg[common].max() - earliest_agg[common].min()
                        )
                        converging = spread_latest < spread_earliest
                        cc["converging"] = converging
                        cc["spread_note"] = (
                            f"Spread {'narrowed' if converging else 'widened'} from "
                            f"{round(spread_earliest, 2)} ({earliest_year}) to "
                            f"{round(spread_latest, 2)} ({latest_year})"
                        )

                    profile["cross_country"] = cc
        except Exception:
            pass

    # ── 7.  Breakdowns with values, labels, and codelist definitions ───────────
    _trivial: frozenset[str] = frozenset({"_T", "_Z", "U", ""})
    _DIM_MEANINGS: dict[str, str] = {
        "sex": "Gender breakdown",
        "age": "Age group breakdown",
        "urbanisation": "Urban/rural breakdown",
        "residence": "Residence type",
        "comp_breakdown_1": "Indicator subtype",
        "comp_breakdown_2": "Secondary breakdown",
        "comp_breakdown_3": "Tertiary breakdown",
        "unit_measure": "Unit of measurement",
    }
    _DIM_TO_CODELIST: dict[str, str] = {
        "sex": "SEX",
        "age": "AGE",
        "urbanisation": "URBANISATION",
        "residence": "RESIDENCE",
        "comp_breakdown_1": "COMP_BREAKDOWN_1",
        "comp_breakdown_2": "COMP_BREAKDOWN_2",
        "comp_breakdown_3": "COMP_BREAKDOWN_3",
        "unit_measure": "UNIT_MEASURE",
    }
    _cl: Any = None
    try:
        from data360.providers import get_codelist_manager as _get_cl_mgr
        _cl = _get_cl_mgr()
    except Exception:
        pass

    breakdowns: dict[str, dict] = {}
    for dim in _SURFACE_DIMS:
        if dim not in df.columns:
            continue
        vals = sorted(str(v) for v in df[dim].dropna().unique() if str(v) not in _trivial)
        if len(vals) > 1:
            meaning = _DIM_MEANINGS.get(dim, dim.replace("_", " ").title())
            if dim_name_labels and dim in dim_name_labels:
                meaning = dim_name_labels[dim]
            bd_entry: dict[str, Any] = {
                "values": vals,
                "cardinality": len(vals),
                "meaning": meaning,
            }
            if _cl and dim in _DIM_TO_CODELIST:
                try:
                    all_labels = _cl.get_dimension_labels(_DIM_TO_CODELIST[dim])
                    code_vals = [v for v in df[dim].dropna().unique() if str(v) not in _trivial]
                    val_labels = {
                        str(code): all_labels[str(code)]
                        for code in code_vals
                        if str(code) in all_labels and all_labels[str(code)] != str(code)
                    }
                    if val_labels:
                        bd_entry["value_labels"] = val_labels
                        bd_entry["codelist"] = _DIM_TO_CODELIST[dim]
                except Exception:
                    pass
            breakdowns[dim] = bd_entry

    if breakdowns:
        profile["breakdowns"] = breakdowns

    # ── 8.  Composition hint ───────────────────────────────────────────────────
    composition: dict[str, Any] = {"suitable_for_stacked": False}
    if "value" in df.columns:
        val_series = pd.to_numeric(df["value"], errors="coerce").dropna()
        if not val_series.empty:
            composition["all_positive"] = bool((val_series >= 0).all())
            group_cols = [c for c in ("country", "year") if c in df.columns]
            if group_cols and breakdowns:
                try:
                    sums = df.assign(
                        _val=pd.to_numeric(df["value"], errors="coerce")
                    ).groupby(group_cols)["_val"].sum()
                    non_null_sums = sums.dropna()
                    if not non_null_sums.empty:
                        mean_sum = float(non_null_sums.mean())
                        composition["sums_to_100"] = 90 <= mean_sum <= 110
                        if composition["sums_to_100"] and composition["all_positive"]:
                            composition["suitable_for_stacked"] = True
                            composition["reason"] = (
                                f"Values within groups average {mean_sum:.0f} "
                                "— looks like parts summing to ~100%"
                            )
                except Exception:
                    composition["sums_to_100"] = False
            if not composition.get("reason"):
                composition["reason"] = (
                    "Independent values — not parts of a whole"
                    if not composition.get("suitable_for_stacked")
                    else ""
                )
    profile["composition_hint"] = composition
    return profile

def _format_source_line_from_attribution(attrib: dict[str, str]) -> str:
    """One-line \"Source\" string; matches client `formatData360VizSourceLine`."""
    ind_name = (attrib.get("indicator_name") or "").strip()
    ind_id = (attrib.get("indicator_id") or "").strip()

    if ind_name and ind_id and ind_name != ind_id:
        ind = f"{ind_name} ({ind_id})"
    else:
        ind = ind_name or ind_id

    if ind:
        return f"World Bank — {ind}"
    return _SOURCE_FALLBACK


def _format_subtitle_line(
    warning: str | None,
    strategy: str | None,
    reason: str | None,
) -> str | None:
    """Optional subtitle under chart title; matches client `formatData360VizSubtitleLine`."""
    parts: list[str] = []
    if warning:
        parts.append(warning)
    if strategy or reason:
        parts.append(" — ".join(x for x in (strategy or "", reason or "") if x))
    if not parts:
        return None
    return " · ".join(parts)


def _scalar_for_json(value: object) -> object:
    """Coerce pandas/numpy scalars so stdlib json can encode them."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat() if pd.notna(value) else None
    return value


def _sanitize_dataframe_for_json_records(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``DataFrame.to_dict(orient='records')`` is JSON-serializable (no Timestamp)."""
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].map(lambda x: x.isoformat() if pd.notna(x) else None)
        elif out[col].dtype == object:
            out[col] = out[col].map(_scalar_for_json)
    return out


# ============================================================================
# DATA FETCHING
# ============================================================================


async def _fetch_data_internal(url: str) -> pd.DataFrame:
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
    client = get_shared_httpx_client()

    parsed = urlparse(url)
    params = parse_qs(parsed.query)

    all_raw_data = []
    has_more = True
    offset = 0

    while has_more:
        params["offset"] = [str(offset)]
        new_query = urlencode(params, doseq=True)
        page_url = urlunparse(parsed._replace(query=new_query))

        response = await client.get(page_url)
        response.raise_for_status()
        data = response.json()
        raw_data = data.get("value", [])
        if not raw_data:
            break
        all_raw_data.extend(raw_data)

        has_more = data.get("has_more", False)
        next_offset = data.get("next_offset")
        if next_offset is not None and next_offset > offset:
            offset = next_offset
        else:
            offset += len(raw_data)

        if len(all_raw_data) >= 1000:
            _logger.warning(
                "_fetch_data_internal: capped at 1000 rows for %s — some data may be truncated.",
                repr(url),
            )
            break

    if not all_raw_data:
        raise ValueError("No data found at the provided URL.")
    return pd.DataFrame(all_raw_data)



async def _fetch_single_indicator(
    database_id: str,
    indicator_id: str,
    country_code: str | None,
    start_year: int | None,
    end_year: int | None,
    disaggregation_filters: dict | None,
) -> tuple[pd.DataFrame, str | None, str | None]:
    """Fetch one indicator and return (DataFrame, title, unit_label).

    Returns (empty_df, None, None) on error — caller checks df.empty.
    """
    from data360.api import get_data_api_url, get_metadata

    try:
        data_url = await get_data_api_url(
            database_id=database_id,
            indicator_id=indicator_id,
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
            disaggregation_filters=disaggregation_filters,
        )
        df = await _fetch_data_internal(data_url)
        df.columns = [c.lower() for c in df.columns]
        if "urbanisation" in df.columns:
            df = df.rename(columns={"urbanisation": "residence"})
        elif "urbanization" in df.columns:
            df = df.rename(columns={"urbanization": "residence"})

        # Filter out total sentinels if breakdown values exist
        for col in ["sex", "age", "residence", "comp_breakdown_1", "comp_breakdown_2", "comp_breakdown_3"]:
            if col in df.columns:
                uv = df[col].dropna().unique()
                if len(uv) > 1 and "_T" in uv:
                    df = df[df[col] != "_T"].copy()
    except Exception as e:
        _logger.error(f"Failed to fetch {repr(indicator_id)}: {repr(e)}")
        return pd.DataFrame(), None, None

    # Fetch title + unit from metadata
    title, unit = None, None
    try:
        parsed = urlparse(data_url)
        params = parse_qs(parsed.query)
        db_id = params.get("DATABASE_ID", [database_id])[0]
        ind_id = (
            params.get("indicatorId", [None])[0]
            or params.get("INDICATOR", [None])[0]
            or indicator_id
        )
        meta = await get_metadata(db_id, ind_id)
        if meta and meta.indicator_metadata:
            title = meta.indicator_metadata.get("name")
            unit = meta.indicator_metadata.get(
                "measurement_unit"
            ) or meta.indicator_metadata.get("unit_measure")
    except Exception as e:
        _logger.warning(f"Could not fetch metadata for {repr(indicator_id)}: {repr(e)}")

    # Qualify unit using unit_mult if present
    raw_unit = ""
    raw_unit_mult = 0
    if not df.empty:
        if "unit_measure" in df.columns:
            _units = df["unit_measure"].dropna().unique().tolist()
            if len(_units) == 1:
                raw_unit = _units[0]
        if "unit_mult" in df.columns:
            _mults = df["unit_mult"].dropna().unique().tolist()
            if len(_mults) == 1:
                try:
                    raw_unit_mult = int(_mults[0])
                except (ValueError, TypeError):
                    pass

    try:
        from data360.providers import get_codelist_manager
        _cl_mgr = get_codelist_manager()
        await _cl_mgr._ensure_extdataportal_loaded()

        resolved_label = ""
        if raw_unit:
            _resolved = _cl_mgr.get_label("UNIT_MEASURE", raw_unit)
            resolved_label = _resolved if _resolved != raw_unit else ""

        unit_name = resolved_label or unit or raw_unit

        from data360.api import _qualify_unit_name
        has_mapping = bool(resolved_label or unit)
        is_special_ps = bool(raw_unit and raw_unit.upper() == "PS" and raw_unit_mult > 0)
        if has_mapping or is_special_ps:
            unit = _qualify_unit_name(unit_name, raw_unit_mult, raw_unit)
        else:
            unit = None

    except Exception as e:
        _logger.warning(f"Could not qualify unit for {repr(indicator_id)}: {repr(e)}")

    return df, title, unit


# ============================================================================
# DATA CLEANING HELPERS
# ============================================================================


def _clean_single_df(
    data: pd.DataFrame,
    relevant_fields: list[str] | None,
    chart_type: str | None,
    data_frequency: str | None,
) -> tuple[pd.DataFrame, list[str], viz_config.TemporalFreq]:
    if "ref_area" in data.columns:
        try:
            from data360.providers import get_group_hierarchy_manager
            _ghm = get_group_hierarchy_manager()
            has_leaf = data["ref_area"].apply(lambda x: _ghm.is_country(str(x))).any()
            if has_leaf:
                data = data[data["ref_area"].apply(lambda x: not _ghm.is_group(str(x)))]
        except Exception as e:
            _logger.warning(f"Could not filter FMR leaf economies: {repr(e)}")

    # Trivial values for disaggregation dimensions: _T = aggregate total, _Z = not applicable.
    # unit_measure uses a different sentinel: 'U' = Unitless (defined in _UNIT_MEASURE_TRIVIAL).
    _TRIVIAL_DIM_VALUES = ("_T", "_Z")

    def _is_non_trivial(col_name: str, series: pd.Series) -> bool:
        """Return True when a dimension column carries discriminating values."""
        uv = series.unique()
        if col_name == "unit_measure":
            non_trivial = set(uv) - _UNIT_MEASURE_TRIVIAL
            return len(non_trivial) > 1
        return len(uv) > 1 or (len(uv) == 1 and uv[0] not in _TRIVIAL_DIM_VALUES)

    if relevant_fields:
        req = [f.lower() for f in relevant_fields]
        missing = [f for f in req if f not in data.columns]
        if missing:
            raise ValueError(
                f"Requested fields not found: {missing}. Available: {list(data.columns)}"
            )
        valid_cols = req
        for dim in ["ref_area", *_VIZ_DISAGG_DIMS]:
            if dim in data.columns and dim not in valid_cols:
                if _is_non_trivial(dim, data[dim]):
                    valid_cols.append(dim)
        viz_data = data[valid_cols].copy()
        relevant_cols = valid_cols
    else:
        relevant_cols = []
        for col in ["time_period", "obs_value", "ref_area"]:
            if col in data.columns:
                relevant_cols.append(col)
        for dim in _VIZ_DISAGG_DIMS:
            if dim in data.columns:
                if _is_non_trivial(dim, data[dim]):
                    relevant_cols.append(dim)
        viz_data = data[relevant_cols].copy() if relevant_cols else data.copy()

    # Temporal preparation — use API's FREQ code directly; fall back to
    # inference from time_period values only when FREQ is unavailable.
    _FREQ_TO_TEMPORAL: dict[str, viz_config.TemporalFreq] = {
        "A": "annual",
        "M": "monthly",
        "Q": "quarterly",
        "D": "daily",
    }
    temporal_frequency: viz_config.TemporalFreq = "annual"
    if "time_period" in viz_data.columns:
        try:
            # Prefer the FREQ code from the API (already parsed by the caller).
            if data_frequency and data_frequency.upper() in _FREQ_TO_TEMPORAL:
                temporal_frequency = _FREQ_TO_TEMPORAL[data_frequency.upper()]
            else:
                # Fallback: infer from the actual time_period values.
                temporal_frequency = viz_config._detect_temporal_frequency(
                    viz_data["time_period"]
                )
            viz_data["time_period"] = viz_config._format_time_period_series(
                viz_data["time_period"], temporal_frequency
            )
        except Exception as e:
            _logger.warning(f"time_period conversion failed: {repr(e)}")
            try:
                viz_data["time_period"] = pd.to_datetime(
                    viz_data["time_period"]
                ).dt.year.astype(str)
            except Exception:
                pass

    if "obs_value" in viz_data.columns:
        viz_data["obs_value"] = pd.to_numeric(viz_data["obs_value"], errors="coerce")

    # Rename to friendly names
    viz_data = viz_data.rename(
        columns={"time_period": "year", "obs_value": "value", "ref_area": "country"}
    )
    relevant_cols = [
        {"time_period": "year", "obs_value": "value", "ref_area": "country"}.get(c, c)
        for c in relevant_cols
    ]
    if "value" in viz_data.columns:
        viz_data["value"] = pd.to_numeric(viz_data["value"], errors="coerce")
    return viz_data, relevant_cols, temporal_frequency


# Inverse rename map: viz_data friendly column names → raw API column names.
# Used by _resolve_hidden_dimension to identify which raw columns are already
# represented in the cleaned DataFrame.
_VIZ_TO_RAW_NAMES: dict[str, str] = {
    "year": "time_period",
    "value": "obs_value",
    "country": "ref_area",
}


def _resolve_hidden_dimension(
    viz_data: pd.DataFrame,
    raw_data: pd.DataFrame,
    max_cardinality: int = 8,
) -> tuple[pd.DataFrame, str | None]:
    """Detect and resolve duplicate (key → value) rows in viz_data.

    After ``_clean_single_df``, columns present in the raw API data that lie
    outside ``_VIZ_DISAGG_DIMS`` are silently dropped.  When such a column has
    ``n_unique > 1`` it creates rows that share the same visualization key
    (year × country × breakdowns) but carry different values, producing a
    sawtooth / zigzag pattern in line and bar charts.

    Resolution strategy
    -------------------
    1. **Surface**: if a hidden dimension is found with cardinality
       ≤ *max_cardinality*, add it to ``viz_data`` under the first unused
       ``comp_breakdown_N`` slot (N ∈ {1, 2, 3}) so that ``select_strategy``
       naturally routes it as a chart breakdown dimension.
    2. **Aggregate**: if no suitable hidden dimension exists, collapse
       duplicate rows to their row-wise mean and return a warning string that
       the caller appends to the chart subtitle.

    Parameters
    ----------
    viz_data :
        DataFrame produced by ``_clean_single_df`` (columns: year, value,
        country, and any breakdown dims already surfaced).
    raw_data :
        Original API response DataFrame — all columns present before cleaning.
    max_cardinality :
        Maximum number of unique values a hidden dimension may have to be
        surfaced as a chart series.  Higher-cardinality dims are aggregated.

    Returns
    -------
    (resolved_df, warning_message_or_None)
        *warning_message* is ``None`` when no duplicates were found.
    """
    key_cols = [c for c in viz_data.columns if c != "value"]

    # Fast path: no duplicates — nothing to resolve.
    if not viz_data.duplicated(subset=key_cols).any():
        return viz_data, None

    # Build the set of raw column names already represented in viz_data so we
    # can find truly hidden columns (present in raw_data but absent in viz_data).
    raw_names_in_viz: set[str] = {_VIZ_TO_RAW_NAMES.get(c, c) for c in viz_data.columns}

    # Candidate hidden dimensions: in raw_data, not in viz_data, 1 < n_unique ≤ max_cardinality.
    candidates: list[tuple[str, int]] = []
    for col in raw_data.columns:
        if col in raw_names_in_viz:
            continue
        try:
            n_u = raw_data.loc[viz_data.index, col].dropna().nunique()
        except (KeyError, IndexError):
            continue
        if 1 < n_u <= max_cardinality:
            candidates.append((col, n_u))

    # Prefer the dimension with the smallest cardinality (fewer panels / series).
    candidates.sort(key=lambda x: x[1])

    if candidates:
        hidden_col, hidden_n = candidates[0]
        # Assign to the first comp_breakdown_N slot not already used in viz_data.
        target_slot: str | None = None
        for slot in ("comp_breakdown_1", "comp_breakdown_2", "comp_breakdown_3"):
            if slot not in viz_data.columns:
                target_slot = slot
                break

        if target_slot is not None:
            viz_data = viz_data.copy()
            viz_data[target_slot] = raw_data.loc[viz_data.index, hidden_col].values
            _logger.info(
                "Hidden dimension %s (%d values) surfaced as %s.",
                repr(hidden_col),
                hidden_n,
                repr(target_slot),
            )
            return viz_data, (
                f"Additional dimension '{hidden_col}' ({hidden_n} values) "
                f"detected in the source data and surfaced as a chart series."
            )

    # Fallback: aggregate duplicate rows to their mean value.
    n_before = len(viz_data)
    viz_data = viz_data.groupby(key_cols, sort=False)["value"].mean().reset_index()
    n_collapsed = n_before - len(viz_data)
    dim_hint = f" (hidden dimension: '{candidates[0][0]}'" if candidates else ""
    _logger.warning(
        "Collapsed %d duplicate rows to row-wise mean%s.",
        n_collapsed,
        (
            repr(f" — hidden dim '{candidates[0][0]}' had too many values")
            if candidates
            else ""
        ),
    )
    return viz_data, (
        f"Note: {n_collapsed} duplicate data rows were averaged"
        + (f" over hidden dimension '{candidates[0][0]}'." if candidates else ".")
    )


async def _map_country_codes(viz_data: pd.DataFrame) -> pd.DataFrame:
    """Map REF_AREA / country codes to human-readable names."""
    col = "country" if "country" in viz_data.columns else None
    if col is None:
        return viz_data
    try:
        from data360.providers import get_codelist_mapping

        country_map = await get_codelist_mapping("REF_AREA")
        viz_data[col] = viz_data[col].map(lambda x: country_map.get(x, x))
    except Exception as e:
        _logger.warning(f"Could not map country codes: {repr(e)}")
    return viz_data


# Dimensions resolved automatically from the extdataportal bundle.
# Maps DataFrame column name → extdataportal dimension key.
_EXTDATAPORTAL_DIM_MAP: dict[str, str] = {
    "comp_breakdown_1": "COMP_BREAKDOWN_1",
    "comp_breakdown_2": "COMP_BREAKDOWN_2",
    "comp_breakdown_3": "COMP_BREAKDOWN_3",
    "sex": "SEX",
    "age": "AGE",
    "urbanisation": "URBANISATION",
    "residence": "URBANISATION",
    "unit_measure": "UNIT_MEASURE",
}


async def _map_dimension_codes(viz_data: pd.DataFrame) -> pd.DataFrame:
    """Replace raw dimension codes with human-readable labels from extdataportal.

    Resolves COMP_BREAKDOWN_1/2/3, SEX, AGE, URBANISATION, and UNIT_MEASURE
    columns using the extdataportal codelist (fetched lazily on first call,
    same as REF_AREA). Columns absent from the DataFrame are silently skipped.
    Codes not found are left unchanged (graceful fallback).

    Must be called *before* the ``series_labels`` override so that LLM-supplied
    labels can still take precedence over auto-resolved ones.
    """
    try:
        from data360.providers import get_codelist_manager

        manager = get_codelist_manager()
        # Lazy-load the extdataportal mapping on first use (no startup hook needed).
        await manager._ensure_extdataportal_loaded()
        for col, dim in _EXTDATAPORTAL_DIM_MAP.items():
            if col not in viz_data.columns:
                continue
            lookup = manager.get_dimension_labels(dim)
            if lookup:
                viz_data[col] = viz_data[col].map(lambda x, lu=lookup: lu.get(x, x))
    except Exception as exc:
        _logger.warning("Could not auto-resolve dimension codes: %s", repr(exc))
    return viz_data


def _find_common_prefix(strings: list[str]) -> str:
    """Return the longest string that is a prefix of every element in *strings*."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""
    return prefix


def _strip_common_prefix_in_dims(
    df: pd.DataFrame,
    dim_cols: list[str],
    min_prefix_len: int = 15,
) -> pd.DataFrame:
    """Strip a long shared prefix from each comp_breakdown_* column.

    When all values in a column share a long common prefix (e.g.
    ``"Severity Phase of Acute Food Insecurity or Malnutrition : "``), the
    prefix carries no information and Vega-Lite labels become indistinguishable
    after truncation.  This function strips the prefix in-place and leaves only
    the unique suffix (e.g. ``"Phase 1 - Minimal"``).

    The ``series_labels`` override runs *after* this step, so the LLM can
    further customise labels for heterogeneous cases (e.g. WGI breakdowns).

    *min_prefix_len* guards against spurious stripping when a short coincidental
    prefix exists (default 15 chars).
    """
    for col in dim_cols:
        if col not in df.columns:
            continue
        unique_vals = [v for v in df[col].dropna().unique() if isinstance(v, str)]
        if len(unique_vals) < 2:
            continue
        prefix = _find_common_prefix(unique_vals)
        if prefix and prefix[-1].isalnum():
            # Find the last space or punctuation character to avoid cutting a word in half
            match = re.search(r'[^a-zA-Z0-9][a-zA-Z0-9]+$', prefix)
            if match:
                prefix = prefix[:match.start() + 1]
        if len(prefix) < min_prefix_len:
            continue
        # Strip trailing separator characters so suffixes start cleanly.
        stripped_prefix = prefix.rstrip(": -_/\\ ")
        if not stripped_prefix:
            continue
        sep_len = len(prefix) - len(stripped_prefix)
        cut = len(stripped_prefix) + sep_len  # includes trailing separator(s)
        mapping = {v: v[cut:].lstrip(": -_/\\ ") for v in unique_vals}
        df = df.copy()
        df[col] = df[col].map(
            lambda x, m=mapping: m.get(x, x) if isinstance(x, str) else x
        )
        _logger.debug(
            "Stripped common prefix %s from column %s (%d values)",
            repr(stripped_prefix),
            repr(col),
            len(unique_vals),
        )
    return df


def _slugify(name: str) -> str:
    """Convert indicator name to a safe column name."""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s[:40].strip("_") or "indicator"


def _make_unique_col(base: str, existing: set[str]) -> str:
    col, n = base, 1
    while col in existing:
        col = f"{base}_{n}"
        n += 1
    return col


# ============================================================================
# TOOL: get_supported_chart_types
# ============================================================================


def get_supported_chart_types() -> str:
    """Return supported chart types and their data requirements as JSON.

    Call before data360_get_viz_spec or data360_get_multi_indicator_viz_spec
    to choose the right chart_type for the data and user intent.

    Returns:
        JSON string with chart_types list and strategy guidance.
    """
    chart_types = {
        "chart_types": [
            {
                "id": "line",
                "description": "Line chart for temporal trends.",
                "when_to_use": "Single indicator, 1+ countries, multiple years.",
                "data_requirements": "time_period + obs_value. Color-codes countries automatically.",
            },
            {
                "id": "bar",
                "description": "Horizontal bar chart for ranking/comparison.",
                "when_to_use": "Single indicator, multiple countries, typically one year.",
                "data_requirements": "obs_value + country dimension.",
            },
            {
                "id": "scatter",
                "description": "Scatterplot for correlation between two indicators.",
                "when_to_use": "Exactly 2 indicator_ids, multiple countries, single year.",
                "data_requirements": "Requires indicator_ids list with 2 entries in get_multi_indicator_viz_spec.",
            },
            {
                "id": "connected_scatter",
                "description": "Connected scatterplot: 2 indicators over time.",
                "when_to_use": "Exactly 2 indicator_ids, multiple countries, multiple years.",
                "data_requirements": "Same as scatter but multi-year.",
            },
            {
                "id": "layered_lines",
                "description": "Dual/multi-axis line chart for 2-3 indicators in one country.",
                "when_to_use": "2-3 indicator_ids, typically 1 country, multi-year.",
                "data_requirements": "Requires indicator_ids list in get_multi_indicator_viz_spec.",
            },
            {
                "id": "small_multiples",
                "description": "Faceted panel chart for breakdown × country comparisons.",
                "when_to_use": "1 indicator, multiple breakdowns (sex/age) or many countries.",
                "data_requirements": "Disaggregation dimensions with multiple values.",
            },
            {
                "id": "strip",
                "description": "Strip/beeswarm chart for cross-country distribution.",
                "when_to_use": ">8 countries, single year.",
                "data_requirements": "obs_value + many country values.",
            },
            {
                "id": "area",
                "description": "Stacked area chart for part-to-whole composition over time.",
                "when_to_use": "User asks about composition, share, or breakdown over time.",
                "data_requirements": "Requires multiple breakdown series or indicator_ids that sum to a whole.",
            },
            {
                "id": "heatmap",
                "description": "Heatmap matrix for high-cardinality time series.",
                "when_to_use": ">8 countries over multiple years without breakdowns.",
                "data_requirements": "Automatically selected for dense country x year data.",
            },
            {
                "id": "map",
                "description": "Geographic choropleth map.",
                "when_to_use": "User asks for a map or spatial distribution across countries.",
                "data_requirements": "Single indicator across multiple countries.",
            },
        ],
        "multi_indicator_note": (
            "For scatter, connected_scatter, and layered_lines, use "
            "data360_get_multi_indicator_viz_spec with an indicator_ids list."
        ),
    }
    return json.dumps(chart_types, indent=2)


# ============================================================================
# TOOL: get_viz_spec (single indicator — original interface preserved)
# ============================================================================


async def _detect_missing_countries(country_code: str | None, present_countries: set[str]) -> list[str]:
    """Identify which of the requested country codes are missing from the returned set."""
    if not country_code:
        return []
    requested_list = [c.strip().upper() for c in country_code.replace(";", ",").split(",") if c.strip()]
    if not requested_list:
        return []
    try:
        from data360.providers import get_codelist_mapping
        country_map = await get_codelist_mapping("REF_AREA")
    except Exception:
        country_map = {}

    missing_names = []
    present_upper = {str(c).upper() for c in present_countries}
    for code in requested_list:
        name = country_map.get(code, code)
        if code not in present_upper and name.upper() not in present_upper:
            missing_names.append(name)
    return missing_names


def _apply_post_processing_rules(
    spec: dict,
    data_frequency: str | None,
    unit_measure: str | None,
    strategy_result: "viz_config.StrategyResult",
    df: pd.DataFrame
) -> dict:
    """Recursively applies post-processing rules to all sub-views/panels in a Vega-Lite spec."""
    import inspect

    def _apply_rule_recursively(rule, subspec, data_root=None, is_composite=False, **kwargs):
        if not isinstance(subspec, dict):
            return subspec

        # Structural rules must be run at the top-level
        is_structural = rule.name in ("general_error_band", "population_pyramid", "apply_wb_style")
        if is_structural:
            sig = inspect.signature(rule.apply)
            rule_kwargs = dict(kwargs)
            if "is_composite" in sig.parameters:
                rule_kwargs["is_composite"] = is_composite
            return rule.apply(subspec, data_frequency=data_frequency, unit_measure=unit_measure, **rule_kwargs)

        # For rules like line_year_gap_stroke_dash, if we are at a panel root (has layer or spec),
        # apply the rule directly here so it can handle the layer/spec structure and data localization.
        if rule.name == "line_year_gap_stroke_dash" and ("layer" in subspec or "spec" in subspec):
            sig = inspect.signature(rule.apply)
            rule_kwargs = dict(kwargs)
            if "is_composite" in sig.parameters:
                rule_kwargs["is_composite"] = is_composite
            return rule.apply(subspec, data_frequency=data_frequency, unit_measure=unit_measure, **rule_kwargs)

        current_data_root = data_root if data_root is not None else subspec

        # Recurse into composite views
        if "concat" in subspec and isinstance(subspec["concat"], list):
            subspec["concat"] = [
                _apply_rule_recursively(rule, child, current_data_root, is_composite=True, **kwargs)
                for child in subspec["concat"]
            ]
            return subspec
        elif "vconcat" in subspec and isinstance(subspec["vconcat"], list):
            subspec["vconcat"] = [
                _apply_rule_recursively(rule, child, current_data_root, is_composite=True, **kwargs)
                for child in subspec["vconcat"]
            ]
            return subspec
        elif "concat" in subspec and isinstance(subspec["concat"], list):
            subspec["concat"] = [
                _apply_rule_recursively(rule, child, current_data_root, is_composite=True, **kwargs)
                for child in subspec["concat"]
            ]
            return subspec
        elif "hconcat" in subspec and isinstance(subspec["hconcat"], list):
            subspec["hconcat"] = [
                _apply_rule_recursively(rule, child, current_data_root, is_composite=True, **kwargs)
                for child in subspec["hconcat"]
            ]
            return subspec
        elif "layer" in subspec and isinstance(subspec["layer"], list):
            subspec["layer"] = [
                _apply_rule_recursively(rule, child, current_data_root, is_composite=is_composite, **kwargs)
                for child in subspec["layer"]
            ]
            return subspec
        elif "spec" in subspec and isinstance(subspec["spec"], dict):
            subspec["spec"] = _apply_rule_recursively(rule, subspec["spec"], current_data_root, is_composite=is_composite, **kwargs)
            return subspec

        # Leaf view: temporarily inject data/datasets from root if missing
        has_local_data = "data" in subspec
        if not has_local_data and current_data_root and "data" in current_data_root:
            subspec["data"] = current_data_root["data"]
        if not has_local_data and current_data_root and "datasets" in current_data_root:
            subspec["datasets"] = current_data_root["datasets"]

        # Apply the rule
        sig = inspect.signature(rule.apply)
        rule_kwargs = dict(kwargs)
        if "is_composite" in sig.parameters:
            rule_kwargs["is_composite"] = is_composite
        subspec = rule.apply(subspec, data_frequency=data_frequency, unit_measure=unit_measure, **rule_kwargs)

        # Cleanup injected data if it wasn't modified (still references parent data/datasets)
        if not has_local_data:
            if current_data_root and subspec.get("data") is current_data_root.get("data"):
                subspec.pop("data", None)
            if current_data_root and subspec.get("datasets") is current_data_root.get("datasets"):
                subspec.pop("datasets", None)


        return subspec


    for rule in viz_config.POST_PROCESSING_RULES:
        sig = inspect.signature(rule.apply)
        kwargs = {}
        if "scale_type" in sig.parameters:
            kwargs["scale_type"] = strategy_result.scale_type
        if "unit_mult" in sig.parameters:
            kwargs["unit_mult"] = strategy_result.unit_mult
        if "df" in sig.parameters:
            kwargs["df"] = df
        if "raw_hint" in sig.parameters:
            kwargs["raw_hint"] = strategy_result.raw_hint

        spec = _apply_rule_recursively(rule, spec, **kwargs)

    return spec


async def get_viz_spec(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
    chart_type: str | None = None,
    relevant_fields: list[str] | None = None,
    custom_constraints: list[str] | None = None,
    use_default_constraints: bool = True,
    chart_title: str | None = None,
    series_labels: dict[str, str] | None = None,
    strategy_override: str | None = None,
    charts_api_url_override: object = _UNSET,
) -> VizResult:
    """Generate a Vega-Lite chart from a single Data360 indicator.

    Use when the user wants a visualization for ONE indicator. For comparing multiple
    indicators (scatter, dual-axis), use data360_get_multi_indicator_viz_spec instead.

    ## Grammar of Graphics Contract

    The pipeline maps data dimensions to Vega-Lite aesthetic channels automatically
    (color, facet). DO NOT pre-filter a dimension to simplify the chart — that collapses
    multi-series data into a single undifferentiated line and silently discards information.

    ### disaggregation_filters decision rule

    Only pin a dimension when the user has explicitly requested a specific value:

      CORRECT — user asked for females only:
        disaggregation_filters={"SEX": "F"}

      CORRECT — force aggregate totals (suppress breakdown display):
        disaggregation_filters={"SEX": "_T"}

      CORRECT — dimension is not applicable for this indicator:
        disaggregation_filters={"SEX": "_Z"}

      WRONG — do not pre-filter to reduce chart complexity:
        disaggregation_filters={"COMP_BREAKDOWN_1": "WGI_EST"}  # silently drops other series
        disaggregation_filters={"SEX": "_T"}

    When a dimension has multiple meaningful values and the user has not requested a
    specific one, OMIT it from disaggregation_filters entirely. The pipeline will:
      - detect non-trivial values (anything other than "_T" or "_Z")
      - choose the correct Vega-Lite channel based on data shape (see chart strategy table)

    ### UNIT_MEASURE must never be pre-filtered

    UNIT_MEASURE is a first-class dimension. When an indicator has multiple units
    (e.g. Persons + Percentage, or USD + % of GDP), the pipeline automatically
    produces a separate panel for each unit with independent Y-axes. Pre-filtering
    to one unit (e.g. UNIT_MEASURE="PERSONS") silently discards the other unit and
    produces a misleading single-unit chart.

      WRONG — do not pick a "preferred" unit:
        disaggregation_filters={"UNIT_MEASURE": "PERSONS"}   # drops Percentage panel
        disaggregation_filters={"UNIT_MEASURE": "PT"}        # drops Persons panel

      CORRECT — omit UNIT_MEASURE entirely:
        disaggregation_filters={}   # pipeline creates one panel per unit automatically

    Only pin UNIT_MEASURE when the USER explicitly says "show me only percentages" or
    "I only want the persons count". In that case, pass the exact code from the
    disaggregation response (e.g. "PERSONS", "PT", "USD").

    ### Chart strategy → Vega-Lite encoding table

    The pipeline selects a strategy from the shape of the data after filtering. All
    strategies bypass Draco and produce validated Vega-Lite v5 specs directly.

    Strategy              | When selected                                       | Vega-Lite encoding
    ----------------------|-----------------------------------------------------|-------------------------------------------
    TEMPORAL_SINGLE       | multi-year, ≤8 countries, 0-1 breakdowns            | x=year(temporal), y=value(Q), color=country or breakdown(N)
    CROSS_SECTIONAL       | single year, ≤8 countries, no breakdown             | y=country(N) sorted, x=value(Q) (horizontal bar)
    DISTRIBUTION          | single year, >8 countries                           | x=value(Q), y=country(N) (tick/strip chart)
    BREAKDOWN_COMPARISON  | 1 breakdown, single year, ≤4 countries              | x=country(N), xOffset=breakdown(N), y=value(Q) (grouped bar)
    SMALL_MULTIPLES       | breakdown + >1 country, OR 2+ breakdowns            | facet=country(N), color=breakdown(N), x=year(temporal), y=value(Q)
    TEMPORAL_SINGLE*      | 1 breakdown (any dim), multi-year                   | x=year(temporal), y=value(Q), color=breakdown(N) (multi-series line)
    HEATMAP               | >8 countries, multi-year, 0 breakdowns              | x=year(temporal), y=country(N), color=value(Q) (rect grid)
    STACKED_AREA          | chart_type="area"/"stacked_area", composition Q     | x=year(temporal), y=value(Q) stacked, color=breakdown(N)
    FALLBACK_LINE         | unclassified shapes                                 | x=year(temporal), y=value(Q), color=country(N)

    *When comp_breakdown_1/2 has multiple values and year_count > 1, the pipeline routes
    to TEMPORAL_SINGLE with color=comp_breakdown_1 — producing one colored line per
    breakdown series. This is the correct encoding for WGI, sectoral breakdowns, etc.

    ### High-cardinality and high-dimensionality guidance

    DO NOT reduce country_code or disaggregation_filters to simplify the output.
    The pipeline handles high-cardinality data automatically:

    | Data shape                              | Pipeline action                              |
    |-----------------------------------------|----------------------------------------------|
    | >8 countries, multi-year, 0 breakdowns  | Auto-routes to HEATMAP (rect grid)           |
    | >8 countries + breakdown                | SMALL_MULTIPLES, caps at 6 facets            |
    | Single year, >8 countries               | DISTRIBUTION (strip chart), caps at 20 bars  |
    | ≤8 countries + breakdown, multi-year    | TEMPORAL_SINGLE with colored lines per series|

    If the LLM passes 3 countries when the user asked for all Sub-Saharan Africa,
    the pipeline cannot recover the missing data — it will produce a misleading chart.
    Always pass the full country list and let the pipeline decide the correct strategy.

    ### Visual channel hierarchy (perception accuracy)

    Channels are ranked by how accurately viewers read encoded values:

    Channel           | Type        | Used for
    ------------------|-------------|----------------------------------------
    Position (x/y)    | Highest     | Quantitative values, time axis
    Color (hue)       | Medium      | Nominal categories (country, breakdown)
    Color (lightness) | Medium      | Quantitative gradient (heatmap cells)
    Size              | Medium-low  | Not used in current strategies
    Shape             | Low         | Not used in current strategies

    The pipeline always maps `value` → position (y-axis) and `country`/`breakdown` → color
    (hue). Do not override this — a chart that encodes values as colors (instead of y) is
    strictly less readable, except in heatmaps where a 2D spatial grid makes position
    unavailable for both axes simultaneously.

    ### Encoding type rules (Vega-Lite v5)

    - Year fields: always type="temporal", format="%Y". Do not use "ordinal" for years
      — ordinal maps ISO timestamps to raw millisecond integers on the axis.
    - Country / breakdown fields: type="nominal"
    - Numeric values: type="quantitative"
    - Legend titles: derived from _TOOLTIP_SPECS labels (e.g. "Breakdown", not
      "Comp_Breakdown_1").
    - Color scale: WB categorical palette (9 colors). Gender data uses WB_GENDER_COLORS.
    - Heatmap color scale: sequential ("yellowgreenblue") for positive-only data;
      divergent ("redblue") when data contains negative values.

    Args:
        database_id: Database identifier (e.g., WB_HNP, WB_WDI).
        indicator_id: Indicator ID (e.g., WB_HNP_SP_POP_TOTL).
        country_code: Optional ISO code(s) for REF_AREA — one code, or several separated
            by semicolons (e.g. KEN or CHN;USA, matching required_country style) or commas
            (also accepted). Normalized to comma-separated for the Data API.
        start_year: Optional start year (inclusive).
        end_year: Optional end year (inclusive).
        disaggregation_filters: Optional dimension filters; each value is str or null, not a list.
            Only pin a dimension when the user explicitly requested a specific value. Follow
            these exact rules for pre-filtering:

            1. COMP_BREAKDOWN_1/2/3: NEVER PRE-FILTER. These contain structural components
               (like estimates vs standard errors). Pass them entirely unfiltered and
               let the visualization engine map them to the correct panels/colors.
            2. Demographic Dimensions (SEX, AGE, URBANISATION, EDUCATION): Pre-filter to
               the aggregate total ("_T") by default unless the user explicitly asks for
               a breakdown (e.g., "by sex"). If no "_T" exists, leave unfiltered.
            3. UNIT_MEASURE: NEVER PRE-FILTER. The pipeline automatically creates
               vertically-stacked panels for each unit of measure.

            For REF_AREA use comma-separated ISO codes (e.g. 'KEN,TZA');
            semicolons in REF_AREA are normalized to commas.
        chart_type: Optional hint — "line", "bar", "scatter", "strip", "small_multiples",
            "map", "choropleth", "heatmap", "area", "stacked_area". Use to force a specific chart family.

            ## chart_type decision rule

            Only pass "area" or "stacked_area" when the user's question implies
            **composition or share over time** — i.e., the series together add up to a
            meaningful whole. Typical signal words: "breakdown of", "composition of",
            "share of", "proportion", "what portion", "structure of".

            CORRECT — user asked about composition:
              chart_type="stacked_area"   # "Show the age group breakdown in Japan"
              chart_type="area"           # "What is the composition of electricity by source?"

            WRONG — user asked about a trend:
              chart_type="area"           # "How has GDP changed in Africa?"   ← use default
              chart_type="stacked_area"   # "What is the trend of poverty?"    ← use default

            For trend questions ("how has X changed?", "show the evolution of Y"), omit
            chart_type entirely and let the pipeline choose the correct strategy
            (line chart, heatmap, etc.) based on data shape.
        relevant_fields: Optional list of column names to include in the chart.
        custom_constraints: Deprecated legacy field; ignored by strategy-based specs.
        use_default_constraints: If True (default), apply standard encoding heuristics.
        chart_title: You MUST provide a custom, human-readable chart title here (e.g., 'Male vs. Female Unemployment'). Do not include chart types (e.g. 'Heatmap', 'Line Chart') in the title, as the strategy engine may override the requested chart type based on data cardinality. Do not leave this blank.
        series_labels: Optional. A dictionary mapping raw dimension codes or
            auto-resolved labels to short, human-readable names for legends and
            panel titles (e.g., {"WGI_EST": "Estimate", "WGI_SC": "Score"}).

            The pipeline already handles two cases automatically:
            - Code → label resolution: COMP_BREAKDOWN, SEX, AGE, URBANISATION,
              UNIT_MEASURE codes are resolved to full labels via the extdataportal
              codelist (e.g. "IPC_IPC_PHASE1" → full phase name).
            - Common-prefix stripping: when all values in a dimension share a
              long common prefix, the prefix is stripped automatically, leaving
              only the unique suffix (e.g. all IPC phase labels share
              "Severity Phase of Acute Food Insecurity or Malnutrition : " —
              it is stripped so the legend shows "Phase 1 - Minimal" etc.).

            Use series_labels only when the auto-resolved labels are still too
            long or unclear after prefix stripping — typically for heterogeneous
            dimensions like WGI breakdowns where each label describes a
            structurally different metric (e.g. "Standard error of the governance
            estimate" → "Std Error", "Governance score (0-100)" → "Score").

    Returns:
        Dict with the following fields:

        url (str | None): Browser-accessible URL to the rendered Vega-Lite spec. None on error.
        error (str | None): Error message. None on success.
        strategy (str): Strategy selected by the pipeline (e.g. "small_multiples", "temporal_single").
        reason (str): Human-readable explanation of why that strategy was chosen.
        dimensions (dict[str, list[str]] | None): Non-trivial categorical dimensions present
            in the plotted data, each mapped to its distinct resolved values (human-readable labels,
            not raw codes). Example: {"unit_measure": ["Persons", "Percentage"],
            "comp_breakdown_2": ["Phase 1 - Minimal", "Phase 2 - Stressed"]}.

            When present, narrate what was plotted and how each dimension was encoded
            (color, facet, or filtered). Then offer the user the option to re-call with
            disaggregation_filters pinned to a specific value if they want to focus on
            one breakdown (e.g. {"UNIT_MEASURE": "PT"} for percentage only).

            The pipeline handles encoding automatically per Grammar of Graphics rules —
            dimensions here are informational so you can explain the chart, not instructions
            to re-encode manually.
        data_summary (dict | None): Structural summary of the plotted data:
            - shape: [n_rows, n_cols] — total observations and columns in the chart data.
            - year_range: [first_year, last_year] as strings.
            - countries: sorted list of country names in the chart.
            - value.min / value.max: numeric range of the plotted values (4 decimal places).
            - value.has_negatives: True if the value axis includes negative numbers.
            Use this to narrate scale, scope, and coverage (e.g. "data spans 2000–2022
            across 3 countries, values ranging from 0.2% to 87.4%").
        source_line (str): Formatted attribution string.
        subtitle_line (str | None): Subtitle with strategy and warning info for the chart.
    """
    from data360.api import get_data_api_url, get_disaggregation, get_metadata

    # ── Detect "expand" dimensions ─────────────────────────────────────────────
    # If the caller passes disaggregation_filters={"SEX": None}, it means
    # "fetch all values for SEX". The Data360 API only accepts a single scalar
    # per dimension, so a None value is our internal sentinel for "expand me".
    # We resolve the valid non-trivial codes via get_disaggregation, then fire
    # one fetch per value concurrently and concat the results.
    _TRIVIAL_CODES = {"_T", "_Z"}
    expand_dims: dict[str, list[str]] = {}  # dim → [valid non-trivial codes]
    if disaggregation_filters:
        for dim, val in disaggregation_filters.items():
            if val is None:
                try:
                    disagg = await get_disaggregation(database_id, indicator_id)
                    for d in disagg.get("dimensions") or []:
                        if d.get("field_name", "").upper() == dim.upper():
                            codes = [
                                c
                                for c in (d.get("field_value") or [])
                                if c not in _TRIVIAL_CODES
                            ]
                            if codes:
                                expand_dims[dim] = codes
                            break
                except Exception as e:
                    _logger.warning(f"Could not resolve expand dim {repr(dim)}: {repr(e)}")

    # ── Fetch ──────────────────────────────────────────────────────────────────
    if expand_dims:
        # Build one filter dict per value combination (only handles single expand dim for now).
        # Multi-dim expansion (e.g. SEX × AGE) would be a combinatorial explosion — skip it.
        first_dim, codes = next(iter(expand_dims.items()))
        base_filters = {
            k: v for k, v in (disaggregation_filters or {}).items() if k != first_dim
        }

        async def _fetch_one(code: str) -> pd.DataFrame:
            filters = {**base_filters, first_dim: code}
            try:
                url = await get_data_api_url(
                    database_id=database_id,
                    indicator_id=indicator_id,
                    country_code=country_code,
                    start_year=start_year,
                    end_year=end_year,
                    disaggregation_filters=filters,
                )
                df = await _fetch_data_internal(url)
                df.columns = [c.lower() for c in df.columns]
                return df
            except Exception as e:
                _logger.warning(f"Expand fetch failed for {repr(first_dim)}={repr(code)}: {repr(e)}")
                return pd.DataFrame()

        frames = await asyncio.gather(*[_fetch_one(c) for c in codes])
        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return _err("Error: No data returned for any disaggregation value.")
        data = pd.concat(non_empty, ignore_index=True)
        # data.columns already lowercased inside _fetch_one

        # Build a canonical data_url for the expand-dims case so that
        # frequency detection (step 3) and metadata extraction (step 4)
        # can parse it the same way as the single-fetch path.
        try:
            data_url = await get_data_api_url(
                database_id=database_id,
                indicator_id=indicator_id,
                country_code=country_code,
                start_year=start_year,
                end_year=end_year,
                disaggregation_filters={
                    k: v
                    for k, v in (disaggregation_filters or {}).items()
                    if v is not None
                },
            )
        except Exception:
            # Fall back to a simple constructed URL so metadata steps degrade
            # gracefully rather than crashing on UnboundLocalError.
            data_url = f"?DATABASE_ID={database_id}&INDICATOR={indicator_id}"
    else:
        # 1. Build URL
        try:
            data_url = await get_data_api_url(
                database_id=database_id,
                indicator_id=indicator_id,
                country_code=country_code,
                start_year=start_year,
                end_year=end_year,
                disaggregation_filters=disaggregation_filters,
            )
        except ValueError as e:
            return _err(f"Error: {e}")

        # 2. Fetch data
        try:
            data = await _fetch_data_internal(data_url)
        except ValueError as e:
            return _err(f"Error: {e}")
        except httpx.HTTPStatusError as e:
            return _err(f"Error fetching data: {e.response.status_code}")
        except Exception as e:
            _logger.exception("Failed to fetch data")
            return _err(f"Error fetching data: {e}")

        data.columns = [c.lower() for c in data.columns]

    # 3. Detect frequency
    data_frequency = None
    try:
        parsed = urlparse(data_url)
        params = parse_qs(parsed.query)
        db_id = params.get("DATABASE_ID", [database_id])[0]
        ind_id_param = (
            params.get("indicatorId", [None])[0]
            or params.get("INDICATOR", [None])[0]
            or indicator_id
        )
        if db_id and ind_id_param:
            meta = await get_metadata(db_id, ind_id_param)
            if meta:
                if meta.disaggregation_options:
                    for d in meta.disaggregation_options:
                        if d.get("field_name") == "FREQ" and d.get("field_value"):
                            data_frequency = d["field_value"][0]
                            break
                if not data_frequency and meta.indicator_metadata:
                    data_frequency = viz_config.infer_frequency_from_periodicity(
                        meta.indicator_metadata.get("periodicity", "")
                    )
    except Exception as e:
        _logger.warning(f"Could not detect frequency: {repr(e)}")

    # 4. Fetch title and unit
    chart_title_auto = "Generated Visualization"
    raw_unit = ""
    try:
        parsed = urlparse(data_url)
        params = parse_qs(parsed.query)
        db_id = params.get("DATABASE_ID", [database_id])[0]
        ind_id_param = (
            params.get("indicatorId", [None])[0]
            or params.get("INDICATOR", [None])[0]
            or indicator_id
        )
        if db_id and ind_id_param:
            meta = await get_metadata(db_id, ind_id_param)
            if meta and meta.indicator_metadata:
                chart_title_auto = meta.indicator_metadata.get("name", chart_title_auto)
                raw_unit = (
                    meta.indicator_metadata.get("measurement_unit")
                    or meta.indicator_metadata.get("unit_measure")
                    or ""
                )
    except Exception as e:
        _logger.warning(f"Could not fetch metadata for title: {repr(e)}")

    # Prefer the unit code from the actual data column over the metadata freeform string.
    # Metadata APIs often return display labels (e.g. "Unit") rather than codelist codes
    # (e.g. "PS" for Persons). The SDMX data column is authoritative.
    _TRIVIAL_UNIT_CODES = {"", "_T", "_Z", "U"}  # Unitless / catch-all / not meaningful
    if "unit_measure" in data.columns:
        _data_units = data["unit_measure"].dropna().unique().tolist()
        if len(_data_units) == 1 and _data_units[0] not in _TRIVIAL_UNIT_CODES:
            raw_unit = _data_units[0]  # e.g. "PS" → will resolve to "Persons"
        elif len(_data_units) > 1:
            # Multi-unit datasets: keep the metadata value; _clean_single_df will
            # retain unit_measure as a dimension and the chart will facet on it.
            pass

    try:
        db_map = await get_database_mapping()
    except Exception as e:
        _logger.warning(f"Could not load database mapping for source attribution: {repr(e)}")
        db_map = {}
    database_display = db_map.get(database_id, database_id)
    chart_title_auto = viz_config._clean_label_generic(chart_title_auto)
    indicator_display = (
        chart_title_auto
        if chart_title_auto != "Generated Visualization"
        else indicator_id
    )
    indicator_display = viz_config._clean_label_generic(indicator_display)
    source_attribution: dict[str, str] = {
        "database_id": database_id,
        "database_name": database_display,
        "indicator_id": indicator_id,
        "indicator_name": indicator_display,
    }

    # 5. Clean data — column selection, bar-vs-temporal time handling, renames (→ year/value/country)
    if "obs_value" in data.columns:
        data["obs_value"] = pd.to_numeric(data["obs_value"], errors="coerce")

    # Auto-filter unit_measure for population pyramid candidates (containing age and sex dimensions)
    # when multiple units are present to avoid faceting on unit_measure and collapsing age groups.
    cols = {c.lower() for c in data.columns}
    if "sex" in cols and "age" in cols and "unit_measure" in cols:
        unit_col = [c for c in data.columns if c.lower() == "unit_measure"][0]
        unique_units = data[unit_col].dropna().unique()
        if len(unique_units) > 1:
            count_units = [u for u in unique_units if str(u).upper() in ("COUNT", "VAL", "NUMBER", "VALUE")]
            selected_unit = count_units[0] if count_units else unique_units[0]
            data = data[data[unit_col] == selected_unit].copy()
            _logger.info("Multi-unit population data detected. Auto-filtered unit_measure to %s to preserve age/sex breakdown structure.", repr(selected_unit))

    try:
        viz_data, relevant_cols, temporal_frequency = _clean_single_df(
            data, relevant_fields, chart_type, data_frequency
        )
    except ValueError as e:
        return _err(str(e))

    if viz_data.empty:
        return _err("Error: No data available for visualization after cleaning.")

    # 5.1 Detect and resolve hidden dimensions that create duplicate (key → value)
    # rows after cleaning.  Surface as the next unused comp_breakdown_N slot when
    # cardinality is small, or collapse to mean with a subtitle warning.
    viz_data, _hidden_dim_warning = _resolve_hidden_dimension(viz_data, data)

    # 6. Map country codes
    viz_data = await _map_country_codes(viz_data)

    # 6.1 Auto-resolve dimension codes from extdataportal codelist
    # (COMP_BREAKDOWN_1/2/3, SEX, AGE, URBANISATION → human-readable labels).
    viz_data = await _map_dimension_codes(viz_data)

    # 6.1b Strip long shared prefixes from comp_breakdown_* columns so Vega-Lite
    # legend/panel labels are unique and readable without truncation.
    # e.g. "Severity Phase of Acute Food Insecurity or Malnutrition : Phase 1 - Minimal"
    # → "Phase 1 - Minimal".  The series_labels override below still takes priority.
    _CB_DIMS = [
        c
        for c in ["comp_breakdown_1", "comp_breakdown_2", "comp_breakdown_3"]
        if c in viz_data.columns
    ]
    viz_data = _strip_common_prefix_in_dims(viz_data, _CB_DIMS)

    # 6.2 Resolve raw_unit code to human-readable label for axis/subtitle.
    # get_label returns the raw input unchanged when the code is not found in the
    # extdataportal mapping. We check: if the resolved label equals the raw code,
    # no mapping exists and we suppress the label (show nothing rather than a raw
    # code like "PS" or a freeform metadata string like "Unit").
    raw_unit_label: str = ""
    raw_unit_mult = 0
    if "unit_mult" in data.columns:
        _data_mults = data["unit_mult"].dropna().unique().tolist()
        if len(_data_mults) == 1:
            try:
                raw_unit_mult = int(_data_mults[0])
            except (ValueError, TypeError):
                pass

    try:
        from data360.providers import get_codelist_manager

        _cl_mgr = get_codelist_manager()
        if raw_unit:
            _resolved = _cl_mgr.get_label("UNIT_MEASURE", raw_unit)
            # Only use the resolved label if get_label actually found a mapping.
            has_mapping = _resolved != raw_unit
            raw_unit_label = _resolved if has_mapping else ""

            from data360.api import _qualify_unit_name
            is_special_ps = bool(raw_unit and raw_unit.upper() == "PS" and raw_unit_mult > 0)
            if has_mapping or is_special_ps:
                unit_name = raw_unit_label if raw_unit_label else raw_unit
                raw_unit_label = _qualify_unit_name(unit_name, raw_unit_mult, raw_unit) or ""

    except Exception:
        pass  # No label; y-axis will have no title rather than a raw code

    # 6.5 Apply custom series labels (override auto-resolved labels).
    # series_labels is now optional — the pipeline auto-resolves the dimensions above.
    if series_labels and isinstance(series_labels, dict):
        for col in _VIZ_DISAGG_DIMS:
            if col in viz_data.columns:
                viz_data[col] = viz_data[col].replace(series_labels)

    # 6.5b Pre-filter for confidence interval error bands if present
    from data360.viz_config import _filter_df_for_error_band
    viz_data = _filter_df_for_error_band(viz_data)

    # 6.6 If a cross-sectional chart type is explicitly requested (bar, map, tick) but years are omitted
    # or a multi-year/multi-country dataset is passed, default to the latest available year to prevent
    # a cluttered "bar chart for time series".
    # Exclusions:
    #   - "stacked_bar" is always temporal — never strip its years.
    #   - Any hint when start_year AND end_year were both explicitly provided — the caller
    #     requested a time range; routing (not this pre-filter) decides what to do with it.
    is_cross_sectional_hint = chart_type and any(
        kw in chart_type.lower() and not (kw == "map" and "heatmap" in chart_type.lower())
        for kw in ("bar", "column", "ranking", "map", "choropleth", "tick", "strip", "beeswarm", "distribution")
    ) and chart_type.lower() != "stacked_bar"
    caller_set_year_range = start_year is not None and end_year is not None
    if is_cross_sectional_hint and not caller_set_year_range and not viz_data.empty:
        unique_years = viz_data["year"].nunique() if "year" in viz_data.columns else 0
        unique_countries = viz_data["country"].nunique() if "country" in viz_data.columns else 0
        if unique_years > 1 and unique_countries > 1:
            if "year" in viz_data.columns:
                latest_year = viz_data["year"].max()
                viz_data = viz_data[viz_data["year"] == latest_year].copy()
                _logger.info(
                    f"[get_viz_spec] chart_type={repr(chart_type)}: filtered data to latest year {repr(latest_year)} "
                    f"to prevent cluttered cross-sectional time-series."
                )

    import textwrap

    # Apply text wrapping (Typography T3 constraint) so long single-indicator titles don't overflow
    if chart_title and isinstance(chart_title, str):
        # Allow LLM to override the title completely
        final_title = chart_title
    else:
        # Wrap the auto-generated title
        final_title = (
            textwrap.wrap(chart_title_auto, width=80)
            if isinstance(chart_title_auto, str)
            else chart_title_auto
        )
    # Vega-Lite title + subtitle (geography, year range, unit) after data is cleaned
    chart_title_vl: str | dict = viz_config.build_chart_title_with_context(
        final_title, raw_unit_label or None, viz_data
    )

    # Append missing-country warning and hidden-dimension warning to the subtitle so the user sees it.
    if isinstance(chart_title_vl, dict):
        _sub = chart_title_vl.get("subtitle", [])
        if isinstance(_sub, str):
            _sub = [_sub]
        else:
            _sub = list(_sub)

        present_set = set(viz_data["country"].dropna().unique()) if "country" in viz_data.columns else set()
        missing_names = await _detect_missing_countries(country_code, present_set)
        if missing_names:
            missing_str = ", ".join(missing_names)
            _sub.append(f"Note: Data unavailable for {missing_str}.")

        if _hidden_dim_warning:
            _sub.append(_hidden_dim_warning)

        chart_title_vl["subtitle"] = _sub
    elif isinstance(chart_title_vl, str):
        present_set = set(viz_data["country"].dropna().unique()) if "country" in viz_data.columns else set()
        missing_names = await _detect_missing_countries(country_code, present_set)
        _sub = []
        if missing_names:
            missing_str = ", ".join(missing_names)
            _sub.append(f"Note: Data unavailable for {missing_str}.")
        if _hidden_dim_warning:
            _sub.append(_hidden_dim_warning)
        if _sub:
            chart_title_vl = {"text": chart_title_vl, "subtitle": _sub}

    # 7. Fetch dim labels + build data profile BEFORE routing
    # dim_name_labels must be fetched first so the profile can include codelist-
    # resolved breakdown names.
    _pre_dim_name_labels: dict[str, str] = {}
    try:
        from data360.api import get_comp_breakdown_dim_names
        _pre_dim_name_labels = await get_comp_breakdown_dim_names(database_id, indicator_id)
    except Exception as _exc:
        _logger.debug("Could not fetch comp_breakdown dim names: %s", repr(_exc))

    # Build comprehensive data profile from real cleaned data.
    # This runs BEFORE select_strategy so coverage signals (sparse countries,
    # completeness %) can inform routing decisions and so that the profile is
    # available for spec builders — not just post-hoc narration.
    data_profile = _build_data_profile(
        viz_data,
        scale_type=_infer_scale_type(raw_unit, raw_unit_label) if (raw_unit or raw_unit_label) else None,
        indicator_names=[indicator_display] if indicator_display else None,
        units_raw=[raw_unit] if raw_unit else None,
        units_label=[raw_unit_label] if raw_unit_label else None,
        dim_name_labels=_pre_dim_name_labels or None,
    )

    # Determine strategy (informed by data_profile coverage signals)
    n_indicators = 1
    strategy_result = viz_config.select_strategy(
        viz_data,
        n_indicators=n_indicators,
        chart_type_hint=chart_type,
        raw_unit=raw_unit,
        raw_unit_mult=raw_unit_mult,
        data_profile=data_profile,
        strategy_override=strategy_override,
    )

    # Post-routing sparsity filtering: Only drop countries with sparse data points
    # if the strategy is temporal (which requires drawing line segments).
    # Cross-sectional and distribution charts only require 1 data point per country,
    # so dropping them is incorrect.
    is_temporal = strategy_result.strategy in (
        viz_config.ChartStrategy.TEMPORAL_SINGLE,
        viz_config.ChartStrategy.TEMPORAL_MULTI_IND,
        viz_config.ChartStrategy.SMALL_MULTIPLES,
    )
    if is_temporal:
        _sparse_from_profile = (
            data_profile.get("coverage_quality", {}).get("sparse_countries", [])
        )
        _n_countries_in_data = (
            viz_data["country"].nunique() if "country" in viz_data.columns else 0
        )
        if _sparse_from_profile and _n_countries_in_data >= 2:
            _logger.info(
                f"[get_viz_spec] Dropping sparse countries (< 3 years) for temporal strategy: {repr(_sparse_from_profile)}"
            )
            viz_data = viz_data[~viz_data["country"].isin(_sparse_from_profile)].copy()
            if viz_data.empty:
                return _err("Error: No data available for visualization after filtering sparse countries.")

            # Rebuild profile and re-run strategy selection to reflect pruned data shape
            data_profile = _build_data_profile(
                viz_data,
                scale_type=_infer_scale_type(raw_unit, raw_unit_label) if (raw_unit or raw_unit_label) else None,
                indicator_names=[indicator_display] if indicator_display else None,
                units_raw=[raw_unit] if raw_unit else None,
                units_label=[raw_unit_label] if raw_unit_label else None,
                dim_name_labels=_pre_dim_name_labels or None,
            )
            strategy_result = viz_config.select_strategy(
                viz_data,
                n_indicators=n_indicators,
                chart_type_hint=chart_type,
                raw_unit=raw_unit,
                raw_unit_mult=raw_unit_mult,
                data_profile=data_profile,
                strategy_override=strategy_override,
            )

    # Thread detected temporal frequency through to spec builders.
    strategy_result.temporal_frequency = temporal_frequency
    # Thread dim_name_labels into result for spec builders.
    strategy_result.dim_name_labels = _pre_dim_name_labels

    # If the strategy is CROSS_SECTIONAL, keep only the latest available year per country
    if strategy_result.strategy == viz_config.ChartStrategy.CROSS_SECTIONAL and not viz_data.empty:
        if "country" in viz_data.columns and "year" in viz_data.columns:
            try:
                valid_data = viz_data.dropna(subset=["country", "year"])
                if not valid_data.empty:
                    years_numeric = pd.to_numeric(valid_data["year"], errors="coerce")
                    if years_numeric.notna().any():
                        valid_data = valid_data.assign(_years_num=years_numeric)
                        idx = valid_data.groupby("country")["_years_num"].idxmax()
                    else:
                        idx = valid_data.groupby("country")["year"].idxmax()
                    viz_data = viz_data.loc[idx].copy()
                    _logger.info("[get_viz_spec] CROSS_SECTIONAL strategy: filtered to latest year per country.")
            except Exception as e:
                _logger.warning(f"Failed to filter cross-sectional data to latest year per country: {repr(e)}")

    # 7.05  Data sufficiency guards — return error before chart dispatch rather
    # than produce a misleading visualization with too little data.
    #
    # Minimum country counts per strategy:
    #   - CROSS_SECTIONAL: ≥3 bars needed for a meaningful comparison
    #   - DISTRIBUTION:    ≥10 required so tick/strip chart has visual density
    #   - BREAKDOWN_COMPARISON / SMALL_MULTIPLES: ≥2 countries to show contrast
    _MIN_COUNTRIES_BY_STRATEGY: dict[viz_config.ChartStrategy, int] = {
        viz_config.ChartStrategy.CROSS_SECTIONAL:      1,  # 1+ bars always valid
        viz_config.ChartStrategy.DISTRIBUTION:         10,
        viz_config.ChartStrategy.BREAKDOWN_COMPARISON: 2,
        viz_config.ChartStrategy.SMALL_MULTIPLES:      2,
    }
    if strategy_result.strategy in _MIN_COUNTRIES_BY_STRATEGY:
        _actual_countries = (
            int(viz_data["country"].nunique()) if "country" in viz_data.columns else 0
        )
        _minimum = _MIN_COUNTRIES_BY_STRATEGY[strategy_result.strategy]
        # If it is a population pyramid request, it will be transformed into a single-panel chart, so 1 country is sufficient.
        if strategy_result.strategy == viz_config.ChartStrategy.SMALL_MULTIPLES:
            hint_str = (chart_type or "").lower().strip()
            if "pyramid" in hint_str or "population_pyramid" in hint_str:
                _minimum = 1

        # If we facet or group by breakdown dimensions (not by country), 1 country is sufficient.
        if strategy_result.strategy in (viz_config.ChartStrategy.SMALL_MULTIPLES, viz_config.ChartStrategy.BREAKDOWN_COMPARISON):
            facet_dim = getattr(strategy_result, "facet_dim", None)
            color_dim = getattr(strategy_result, "color_dim", None)
            secondary_color_dim = getattr(strategy_result, "secondary_color_dim", None)
            if facet_dim != "country" and color_dim != "country" and secondary_color_dim != "country":
                _minimum = 1

        if _actual_countries < _minimum:
            return _err(
                f"Insufficient data for a {strategy_result.strategy.value} chart: "
                f"only {_actual_countries} country/countries returned data "
                f"(minimum required: {_minimum}). "
                f"The requested countries/indicator/time period combination had too "
                f"few values in the Data360 API."
            )

    # Sparse time-series guard: if average data points per country < 3,
    # lines will be jagged and disconnected — fall back to cross_sectional bar
    # instead of raising an error.  A single-year request is always valid as a
    # bar chart; only raise if we cannot produce any useful visualization at all.
    if strategy_result.strategy in (
        viz_config.ChartStrategy.SMALL_MULTIPLES,
        viz_config.ChartStrategy.TEMPORAL_SINGLE,
    ) and not viz_data.empty and "country" in viz_data.columns:
        _n_series = max(int(viz_data["country"].nunique()), 1)
        if "year" in viz_data.columns:
            _avg_pts = float(viz_data.groupby("country")["year"].nunique().mean())
        else:
            _avg_pts = len(viz_data) / _n_series
        # Population pyramid requests are designed to show cross-sectional cohorts in a single year, so they should not fall back.
        hint_str = (chart_type or "").lower().strip()
        is_pyramid = "pyramid" in hint_str or "population_pyramid" in hint_str

        # Don't fall back to cross-sectional if we are faceting/grouping by breakdown dimensions,
        # as doing so would collapse the breakdown details.
        is_breakdown = False
        if strategy_result.strategy in (viz_config.ChartStrategy.SMALL_MULTIPLES, viz_config.ChartStrategy.BREAKDOWN_COMPARISON):
            facet_dim = getattr(strategy_result, "facet_dim", None)
            color_dim = getattr(strategy_result, "color_dim", None)
            secondary_color_dim = getattr(strategy_result, "secondary_color_dim", None)
            if facet_dim != "country" and color_dim != "country" and secondary_color_dim != "country":
                is_breakdown = True

        if _avg_pts < 3 and not is_pyramid and not is_breakdown:
            _logger.info(
                f"[get_viz_spec] Sparse data ({_avg_pts:.1f} pts/country) — "
                f"falling back from {strategy_result.strategy.value} to cross_sectional bar."
            )
            strategy_result = viz_config.StrategyResult(
                viz_config.ChartStrategy.CROSS_SECTIONAL,
                f"Sparse data ({_avg_pts:.1f} pts/country) → bar chart",
                mark_hint="bar",
            )

    _logger.info(
        f"Chart strategy: {strategy_result.strategy.value} — {repr(strategy_result.reason)}"
    )

    if strategy_result.strategy == viz_config.ChartStrategy.FALLBACK_LINE:
        return _err(
            "Could not determine a suitable visualization strategy for this data structure. "
            "A fallback visualization was not generated."
        )


    try:
        spec = viz_config.dispatch_spec(
            strategy_result.strategy,
            viz_data,
            chart_title_vl,
            strategy_result,
            indicator_labels=series_labels,
            y_label=raw_unit_label if raw_unit_label else "Value",
            x_label="Value",
            unit_measure=_unit_measure_for_formatting(raw_unit, raw_unit_label),
            indicator_name=indicator_display or None,
        )
        # Apply post-processing rules
        spec = _apply_post_processing_rules(
            spec,
            data_frequency=data_frequency,
            unit_measure=_unit_measure_for_formatting(raw_unit, raw_unit_label),
            strategy_result=strategy_result,
            df=viz_data
        )

        out_reason = strategy_result.reason
        if strategy_result.strategy == viz_config.ChartStrategy.TEMPORAL_SINGLE:
            resolved_mark = viz_config.extract_top_level_mark_type(spec)
            if resolved_mark:
                out_reason = viz_config.patch_strategy_reason_chart_phrase(
                    strategy_result.reason, resolved_mark
                )

        dim_summary = _extract_dimension_summary(viz_data)
        data_summary = _build_data_summary(viz_data)
        # data_profile was built before routing — read from strategy_result
        data_profile = strategy_result.data_profile or {}

        warning_msg = None
        if chart_type:
            requested_lower = chart_type.lower()
            core_intent = (
                requested_lower.replace("chart", "")
                .replace("stacked", "")
                .replace("grouped", "")
                .replace("_", " ")
                .strip()
            )
            if core_intent == "scatter":
                core_intent = "point"
            elif core_intent == "strip":
                core_intent = "beeswarm"

            reason_lower = out_reason.lower()
            reason_suffix = (
                reason_lower.split("→")[-1] if "→" in reason_lower else reason_lower
            )

            is_scatter_match = (core_intent == "point" and ("scatter" in reason_suffix or "scatterplot" in reason_suffix))
            is_point_fallback_to_line = (core_intent == "point" and "line" in reason_suffix)
            if core_intent and core_intent not in reason_suffix and not is_scatter_match and not is_point_fallback_to_line:
                warning_msg = (
                    f"You requested '{chart_type}', but the visualization engine "
                    f"selected a different strategy based on data cardinality: {out_reason}. "
                    "The chart was successfully generated. Please ensure your response and the chart title reflect this actual strategy."
                )

        return _ok(
            await _store_spec(spec, charts_api_url_override=charts_api_url_override),
            warning=warning_msg,
            source_attribution=source_attribution,
            strategy=strategy_result.strategy.value,
            reason=out_reason,
            dimensions=dim_summary or None,
            data_summary=data_summary or None,
            data_profile=data_profile or None,
            spec=spec,
        )
    except Exception as e:
        _logger.exception("Strategy builder failed")
        return _err(f"Chart generation failed: {e}")


# ============================================================================
# TOOL: get_multi_indicator_viz_spec (NEW)
# ============================================================================


async def get_multi_indicator_viz_spec(
    indicator_ids: list[dict[str, str]] | None = None,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
    chart_type: str | None = None,
    chart_title: str | None = None,
    series_labels: dict[str, str] | None = None,
    strategy_override: str | None = None,
) -> VizResult:
    """Generate a Vega-Lite chart comparing multiple Data360 indicators.

    REQUIRED for success: indicator_ids (2–4 entries). Call data360_search_indicators first,
    then pass each series as {"database_id": "...", "indicator_id": "..."}. If you omit
    indicator_ids or pass fewer than 2 entries, this tool returns {"error": "..."} — fix
    the arguments and call again (do not rely on country_code alone).

    Use for: scatterplots (2 indicators vs each other), layered/dual-axis line charts
    (2-3 indicators over time in one country), connected scatter (trajectory charts).

    ## Grammar of Graphics Contract

    The pipeline maps data dimensions to Vega-Lite aesthetic channels automatically.
    DO NOT pre-filter a dimension to simplify the chart — that collapses multi-series
    data into a single undifferentiated line and silently discards information.

    ### disaggregation_filters decision rule

    disaggregation_filters applies to ALL indicators in this call. Only pin a dimension
    when the user has explicitly requested a specific value. Follow these exact rules:

    1. COMP_BREAKDOWN_1/2/3: NEVER PRE-FILTER. These contain structural phases or
       statistical components (e.g. Estimate vs Standard Error). Pass them entirely
       unfiltered and let the visualization engine map them to separate panels/colors.
       (e.g. DO NOT filter {"COMP_BREAKDOWN_1": "WGI_EST"} just to simplify the chart).

    2. Demographic Dimensions (SEX, AGE, URBANISATION, EDUCATION): Pre-filter to the
       aggregate total ("_T") by default unless the user explicitly asks for a breakdown.
       - "What is unemployment?" -> filter {"SEX": "_T"}
       - "Unemployment by sex" -> OMIT the filter so the engine colors them.
       If there is no "_T" available, leave it unfiltered.

    3. UNIT_MEASURE: NEVER PRE-FILTER. The pipeline automatically creates vertically-stacked
       panels for each unit of measure. Omitting this filter is the only way to generate
       multi-unit comparison charts.

    ### Chart strategy → Vega-Lite encoding table

    Strategy              | When selected                                       | Vega-Lite encoding
    ----------------------|-----------------------------------------------------|-------------------------------------------
    CORRELATION           | 2 indicators, >1 country, single year               | x=indicator1(Q), y=indicator2(Q), color=country(N) (scatter)
    CORRELATION_TEMPORAL  | 2 indicators, >1 country, multi-year                | x=indicator1(Q), y=indicator2(Q), color=country(N), order=year (connected scatter)
    TEMPORAL_MULTI_IND    | 2-4 indicators, 1 country or 3+ indicators          | layered lines, independent y-scales per indicator
    STACKED_AREA          | chart_type="area"/"stacked_area", composition Q     | x=year(temporal), y=value(Q) stacked, color=indicator(N)

    ### High-cardinality guidance

    DO NOT reduce indicator_ids or disaggregation_filters to simplify the output.
    Pass the full set of indicators the user asked about. The pipeline will:
    - Merge all indicator series into a single aligned DataFrame
    - Auto-select the correct strategy (scatter, layered lines, stacked area)
    - Cap series count if needed and annotate the subtitle with a trim note

    ### Visual channel hierarchy (perception accuracy)

    Channels are ranked by how accurately viewers read encoded values:

    Channel           | Type        | Used for
    ------------------|-------------|----------------------------------------
    Position (x/y)    | Highest     | Quantitative values, time axis
    Color (hue)       | Medium      | Nominal categories (country, indicator)
    Color (lightness) | Medium      | Not used for multi-indicator charts
    Size / Shape      | Low         | Not used in current strategies

    The pipeline always maps quantitative values → position (y-axis) and nominal
    categories → color (hue). Do not override this with custom disaggregation_filters.

    ### Encoding type rules (Vega-Lite v5)

    - Year fields: type="temporal", format="%Y". Never "ordinal" for years.
    - Country / breakdown fields: type="nominal"
    - Numeric values: type="quantitative", scale.zero=False for non-ratio data
    - Multi-indicator layers: each indicator gets its own y-axis (left/right), its own
      WB categorical color. Resolved with resolve.scale.y="independent".
    - Connected scatter uses order channel (order=year, type=temporal) to draw
      trajectory lines in chronological order.

    Args:
        indicator_ids: List of dicts, each with "database_id" and "indicator_id".
            Example: [
                {"database_id": "WB_WDI", "indicator_id": "WB_WDI_NY_GDP_PCAP_KD"},
                {"database_id": "WB_WDI", "indicator_id": "WB_WDI_SP_DYN_LE00_IN"}
            ]
        country_code: Optional ISO code(s): one code or semicolon-separated (e.g. KEN;MAR)
            or comma-separated; normalized for the Data API like data360_get_data.
        start_year: Optional start year (inclusive).
        end_year: Optional end year (inclusive).
        disaggregation_filters: Optional filters applied to ALL indicators; values are str or null.
            See grammar of graphics contract above — only pin a dimension when the user
            explicitly requested a specific value. Omitting a dimension fetches all values
            and lets the pipeline choose the correct Vega-Lite channel automatically.
            REF_AREA uses comma-separated ISO codes (semicolons normalized to commas).
        chart_type: Optional hint — "scatter", "connected_scatter", "layered_lines",
            "stacked_area", "map", "choropleth", etc. Used to guide the underlying ChartStrategy. If omitted, auto-selected by data shape.

            ## chart_type decision rule

            Only pass "area" or "stacked_area" when the user's question implies
            **composition or share over time** — i.e., the multiple indicators together
            add up to a meaningful whole (e.g., age groups summing to total population,
            electricity sources summing to total generation).
            Typical signal words: "breakdown of", "composition of", "share of",
            "proportion", "structure of".

            CORRECT — user asked about composition across multiple indicators:
              chart_type="stacked_area"   # "Show the age group breakdown in Japan"

            WRONG — user asked about a trend:
              chart_type="area"           # "How has GDP and poverty changed?" ← use default

            For trend or comparison questions, omit chart_type and let the pipeline
            select the correct strategy (layered lines, scatter, etc.).

        chart_title: You MUST provide a custom, human-readable chart title here (e.g., 'Electricity Mix by Source'). Do not include chart types (e.g. 'Heatmap', 'Line Chart') in the title, as the strategy engine may override the requested chart type based on data cardinality. Do not leave this blank.
        series_labels: Optional. A dictionary mapping indicator IDs to short,
            human-readable labels (e.g., {"WB_WDI_EG_ELC_HYRO_ZS": "Hydro"}).
            Provide this to shorten long auto-generated indicator names in legends.

    Returns:
        Dict with "url" (chart URL on success), "error" (on failure),
        "strategy" (which chart type was chosen), "warning" (if applicable).
    """
    if indicator_ids is None or len(indicator_ids) < 2:
        return _err(
            "indicator_ids is required: pass a JSON array of 2–4 objects, each "
            '{"database_id":"<db>","indicator_id":"<id>"} from data360_search_indicators '
            "(use idno + database_id). Example: "
            '[{"database_id":"WB_WDI","indicator_id":"WB_WDI_NY_GDP_PCAP_KD"},'
            '{"database_id":"WB_WDI","indicator_id":"WB_WDI_SP_DYN_LE00_IN"}]. '
            "Then add country_code, start_year, end_year as needed. "
            "Do not call this tool with only country_code or chart_type."
        )
    if len(indicator_ids) > 4:
        return _err("Maximum 4 indicators supported in one chart.")

    # 1. Fetch all indicators concurrently
    tasks = [
        _fetch_single_indicator(
            ind["database_id"],
            ind["indicator_id"],
            country_code,
            start_year,
            end_year,
            disaggregation_filters,
        )
        for ind in indicator_ids
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    dfs: list[pd.DataFrame] = []
    titles: list[str] = []
    units: list[str] = []
    indicator_col_names: list[str] = []
    used_cols: set[str] = set()

    for i, (res, ind) in enumerate(zip(results, indicator_ids)):
        if isinstance(res, Exception):
            return _err(f"Failed to fetch indicator {ind['indicator_id']}: {res}")
        df, title, unit = res
        if df.empty:
            return _err(f"No data returned for indicator {ind['indicator_id']}.")

        df, _, _ = _clean_single_df(
            df, relevant_fields=None, chart_type=None, data_frequency=None
        )

        ind_name = title or ind["indicator_id"]
        # Allow LLM to override the indicator name directly in the dataframe
        if series_labels and ind["indicator_id"] in series_labels:
            ind_name = series_labels[ind["indicator_id"]]

        ind_name = viz_config._clean_label_generic(ind_name)



        col_base = _slugify(ind_name)
        col = _make_unique_col(col_base, used_cols)
        used_cols.add(col)

        dfs.append(df)
        titles.append(ind_name)
        units.append(unit or "")
        indicator_col_names.append(col)

    # 2. Standardize each DataFrame
    std_dfs: list[pd.DataFrame] = []
    for df, col in zip(dfs, indicator_col_names):
        # _clean_single_df already renamed time_period -> year, ref_area -> country, obs_value -> value
        df = df.rename(columns={"value": col})

        # Keep only join keys + value column.
        # CRITICAL: Drop any disaggregation dimension that is CONSTANT (nunique <= 1)
        # for this specific indicator. If Indicator 1 is "Male" and Indicator 2 is "Female",
        # keeping `sex` would cause the outer merge to fail (creating disjoint rows with NaNs).
        keep_dims = [
            c for c in _VIZ_DISAGG_DIMS if c in df.columns and df[c].nunique() > 1
        ]
        keep = ["year", "country"] + keep_dims + [col]

        # Only keep columns that actually exist in the dataframe
        keep = [c for c in keep if c in df.columns]
        std_dfs.append(df[keep])

    # 3. Map country codes and auto-resolve dimension codes (reuse the same helpers
    #    as get_viz_spec for consistency — no duplicate logic).
    for i, df in enumerate(std_dfs):
        std_dfs[i] = await _map_country_codes(df)
        std_dfs[i] = await _map_dimension_codes(std_dfs[i])

    # 4. Merge on common keys
    join_keys = [
        c
        for c in ["year", "country", *_VIZ_DISAGG_DIMS]
        if all(c in df.columns for df in std_dfs)
    ]

    if not join_keys:
        return _err(
            "Indicators could not be merged: no common dimensions (for example, one "
            "series is time-only and another is geography-only). Choose indicators "
            "that share at least one of: year, country, or the same disaggregation "
            "columns."
        )

    merged = std_dfs[0]
    for df in std_dfs[1:]:
        merged = pd.merge(merged, df, on=join_keys, how="outer")

    if merged.empty:
        return _err("No overlapping data found across indicators after merging.")

    # Collapse duplicate rows introduced by the outer-merge fanout.
    # Strategy:
    # 1. Drop rows that are identical across every column — safe, no data loss.
    # 2. If true ambiguous duplicates remain (same join keys, different indicator
    #    values), emit a warning and take the mean as a fallback.  This should
    #    not happen for well-aligned indicators but is guarded explicitly so that
    #    any averaging is visible in the server log rather than silent.
    groupby_keys = [k for k in join_keys if k in merged.columns]
    agg_cols = [col for col in indicator_col_names if col in merged.columns]

    merged = merged.drop_duplicates()

    if groupby_keys and agg_cols:
        dup_mask = merged.duplicated(subset=groupby_keys, keep=False)
        if dup_mask.any():
            _logger.warning(
                "Multi-indicator merge produced %d ambiguous duplicate rows "
                "(same %s keys, differing indicator values). Taking mean as "
                "fallback — verify that the indicators share compatible "
                "disaggregation dimensions.",
                dup_mask.sum(),
                repr(groupby_keys),
            )
            agg_dict = {col: "mean" for col in agg_cols}
            merged = merged.groupby(groupby_keys, as_index=False).agg(agg_dict)


    merged = _sanitize_dataframe_for_json_records(merged)

    import textwrap

    # 5. Build chart title (with subtitle when all indicators share the same unit)
    # GoG / AntVis guideline: Do not arbitrarily truncate strings with ellipses.
    # Instead, preserve the full text but word-wrap it so it fits the chart width.
    if chart_title:
        # If the LLM provided a custom title, use it directly without wrapping.
        # We assume the LLM provides a concise, readable title.
        final_chart_title = chart_title
    else:
        if len(titles) == 2:
            full_title = f"{titles[0]} vs. {titles[1]}"
        else:
            full_title = " | ".join(titles)

        # Wrap at 80 characters to ensure it fits safely within standard chart widths
        final_chart_title = textwrap.wrap(full_title, width=80)

    units_raw_per_indicator = list(units)  # preserve raw codes before label resolution
    unique_raw_units = list(dict.fromkeys(u for u in units if u))
    # Resolve raw unit codes (e.g. "PT") to human-readable labels (e.g. "Percent")
    # using the same extdataportal bundle used by get_viz_spec.
    try:
        from data360.providers import get_codelist_manager as _get_cl_mgr

        _cl = _get_cl_mgr()
        units = [_cl.get_label("UNIT_MEASURE", u) if u else u for u in units]
    except Exception:
        pass
    unique_label_units = list(dict.fromkeys(u for u in units if u))
    shared_unit_raw = unique_raw_units[0] if len(unique_raw_units) == 1 else ""
    shared_unit_label = unique_label_units[0] if len(unique_label_units) == 1 else ""
    chart_title_vl: str | dict = viz_config.build_chart_title_with_context(
        final_chart_title, shared_unit_label or None, merged
    )

    # Append missing-country warning to the subtitle so the user sees it.
    if isinstance(chart_title_vl, dict):
        _sub = chart_title_vl.get("subtitle", [])
        if isinstance(_sub, str):
            _sub = [_sub]
        else:
            _sub = list(_sub)

        present_set = set(merged["country"].dropna().unique()) if "country" in merged.columns else set()
        missing_names = await _detect_missing_countries(country_code, present_set)
        if missing_names:
            missing_str = ", ".join(missing_names)
            _sub.append(f"Note: Data unavailable for {missing_str}.")

        chart_title_vl["subtitle"] = _sub
    elif isinstance(chart_title_vl, str):
        present_set = set(merged["country"].dropna().unique()) if "country" in merged.columns else set()
        missing_names = await _detect_missing_countries(country_code, present_set)
        if missing_names:
            missing_str = ", ".join(missing_names)
            chart_title_vl = {
                "text": chart_title_vl,
                "subtitle": [f"Note: Data unavailable for {missing_str}."]
            }

    # 5b. Shorten titles to their differentiators for mapping, legends, and sub-charts
    diff_map = viz_config._get_label_differentiators(titles)
    shortened_titles = [diff_map.get(t, t) for t in titles]

    # 6. Build indicator_labels for axis/tooltip
    def _format_label(t: str, u: str | None) -> str:
        if not u:
            return t
        # Prevent redundant units like "GDP (annual % growth) (%)"
        if str(u).lower() in str(t).lower():
            return t
        return f"{t} ({u})"

    indicator_labels = {
        col: _format_label(title, unit)
        for col, title, unit in zip(indicator_col_names, shortened_titles, units)
    }

    if series_labels:
        for col in indicator_col_names:
            if col in series_labels:
                indicator_labels[col] = series_labels[col]

    # 7. Build data profile BEFORE routing so coverage signals inform strategy selection
    data_profile = _build_data_profile(
        merged,
        indicator_cols=indicator_col_names,
        indicator_names=titles,
        units_raw=units_raw_per_indicator,
        units_label=list(units),
    )

    # Select strategy (informed by data_profile coverage signals)
    strategy_result = viz_config.select_strategy(
        merged,
        n_indicators=len(indicator_ids),
        chart_type_hint=chart_type,
        indicator_cols=indicator_col_names,
        data_profile=data_profile,
        strategy_override=strategy_override,
    )
    _logger.info(
        f"Multi-indicator strategy: {strategy_result.strategy.value} — {repr(strategy_result.reason)}"
    )

    if strategy_result.strategy == viz_config.ChartStrategy.FALLBACK_LINE:
        return _err(
            "Could not determine a suitable visualization strategy for this data structure. "
            "A fallback visualization was not generated."
        )


    # 7b. Reshape for stacked area / stacked bar: melt wide → long
    # build_stacked_area_spec and build_stacked_bar_spec expect a DataFrame with a single "value" column and a
    # "indicator" color column, not the wide merged layout produced by the join above.
    spec_df = merged
    if strategy_result.strategy in (viz_config.ChartStrategy.STACKED_AREA, viz_config.ChartStrategy.STACKED_BAR):
        id_cols = [c for c in merged.columns if c not in indicator_col_names]
        spec_df = merged.melt(
            id_vars=id_cols,
            value_vars=indicator_col_names,
            var_name="indicator",
            value_name="value",
        )
        # Map internal slugified column names back to human-readable indicator titles
        slug_to_title = dict(zip(indicator_col_names, shortened_titles))
        spec_df["indicator"] = spec_df["indicator"].map(slug_to_title)
        spec_df = spec_df.dropna(subset=["value"])
        # Propagate the color_dim so the builder picks up the indicator column
        strategy_result = viz_config.StrategyResult(
            strategy_result.strategy,
            strategy_result.reason,
            indicator_cols=strategy_result.indicator_cols,
            color_dim="indicator",
        )

    # 7c. Reshape for grouped bar or small multiples (multi-indicator path): melt wide → long.
    # build_breakdown_comparison_spec and build_small_multiples_spec expect df["value"] + df[color_dim].
    # When color_dim="indicator" the wide merged frame must be melted so each
    # (country, indicator) pair becomes a row with a single "value" and an
    # "indicator" label column used as the grouping/coloring key.
    elif (
        strategy_result.strategy
        in (
            viz_config.ChartStrategy.BREAKDOWN_COMPARISON,
            viz_config.ChartStrategy.SMALL_MULTIPLES,
        )
        and (strategy_result.color_dim == "indicator" or strategy_result.facet_dim == "indicator")
    ):
        id_cols = [c for c in merged.columns if c not in indicator_col_names]
        spec_df = merged.melt(
            id_vars=id_cols,
            value_vars=indicator_col_names,
            var_name="indicator",
            value_name="value",
        )
        slug_to_title = dict(zip(indicator_col_names, shortened_titles))
        spec_df["indicator"] = spec_df["indicator"].map(slug_to_title)
        spec_df = spec_df.dropna(subset=["value"])

    # 8. Build spec
    try:
        # If there is a shared unit, it makes sense to use it as the Y-axis label.
        # Otherwise, fall back to the second indicator's name (useful for scatterplots).
        # If all units are percentage-based, label the axis as "Percentage".
        is_all_pct = False
        if unique_label_units:
            is_all_pct = all(
                any(x in str(u).lower() for x in ["percent", "pct", "%"])
                for u in unique_label_units
            )

        if shared_unit_label:
            computed_y_label = shared_unit_label
        elif is_all_pct:
            computed_y_label = "Percentage"
        else:
            # For scatterplots, the second indicator maps to the Y axis.
            # Otherwise, use "Value" to avoid misleadingly labeling the axis with only one series name.
            if strategy_result.strategy == viz_config.ChartStrategy.CORRELATION:
                computed_y_label = indicator_labels.get(indicator_col_names[1], "Value")
            else:
                computed_y_label = "Value"

        spec = viz_config.dispatch_spec(
            strategy_result.strategy,
            spec_df,
            chart_title_vl,
            strategy_result,
            indicator_labels=indicator_labels,
            y_label=computed_y_label,
            x_label=indicator_labels.get(indicator_col_names[0], "Value"),
            unit_measure=_unit_measure_for_formatting(
                shared_unit_raw, shared_unit_label
            ),
        )
    except Exception as e:
        _logger.exception(f"Spec build failed: {repr(e)}")
        return _err(f"Error building chart spec: {e}")

    # 9. Store and return
    try:
        db_map_multi = await get_database_mapping()
    except Exception as e:
        _logger.warning(f"Could not load database mapping for source attribution: {repr(e)}")
        db_map_multi = {}
    db_displays = [
        db_map_multi.get(ind["database_id"], ind["database_id"])
        for ind in indicator_ids
    ]
    unique_db_displays = list(dict.fromkeys(db_displays))
    database_display_multi = (
        unique_db_displays[0]
        if len(unique_db_displays) == 1
        else " · ".join(unique_db_displays)
    )
    indicator_display_multi = " | ".join(titles)
    ids_joined = " · ".join(ind["indicator_id"] for ind in indicator_ids)
    dbs_ids_joined = " · ".join(
        dict.fromkeys(ind["database_id"] for ind in indicator_ids)
    )
    source_attribution_multi: dict[str, str] = {
        "database_id": dbs_ids_joined,
        "database_name": database_display_multi,
        "indicator_id": ids_joined,
        "indicator_name": indicator_display_multi,
    }

    # Apply post-processing rules
    spec = _apply_post_processing_rules(
        spec,
        data_frequency=None,
        unit_measure=_unit_measure_for_formatting(
            shared_unit_raw, shared_unit_label
        ),
        strategy_result=strategy_result,
        df=spec_df
    )

    out_reason = strategy_result.reason
    if strategy_result.strategy == viz_config.ChartStrategy.TEMPORAL_SINGLE:
        resolved_mark = viz_config.extract_top_level_mark_type(spec)
        if resolved_mark:
            out_reason = viz_config.patch_strategy_reason_chart_phrase(
                strategy_result.reason, resolved_mark
            )

    url = await _store_spec(spec)

    # data_profile was built before routing — read from strategy_result
    data_profile = strategy_result.data_profile or {}

    warning_msg = None
    if chart_type:
        requested_lower = chart_type.lower()
        core_intent = (
            requested_lower.replace("chart", "")
            .replace("stacked", "")
            .replace("grouped", "")
            .replace("_", " ")
            .strip()
        )
        if core_intent == "scatter":
            core_intent = "point"
        elif core_intent == "strip":
            core_intent = "beeswarm"

        reason_lower = strategy_result.reason.lower()
        reason_suffix = (
            reason_lower.split("→")[-1] if "→" in reason_lower else reason_lower
        )

        is_scatter_match = (core_intent == "point" and ("scatter" in reason_suffix or "scatterplot" in reason_suffix))
        if core_intent and core_intent not in reason_suffix and not is_scatter_match:
            warning_msg = (
                f"You requested '{chart_type}', but the visualization engine "
                f"selected a different strategy based on data cardinality: {strategy_result.reason}. "
                "The chart was successfully generated. Please ensure your response and the chart title reflect this actual strategy."
            )

    return _ok(
        url,
        warning=warning_msg,
        source_attribution=source_attribution_multi,
        strategy=strategy_result.strategy.value,
        reason=out_reason,
        data_profile=data_profile or None,
        spec=spec,
    )
