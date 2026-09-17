import asyncio
import json
import logging
import math
import threading
import zlib
from typing import Any, Literal
from urllib.parse import urlencode

import cachetools
import dotenv
import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError
from sklearn.linear_model import HuberRegressor

from .config import get_data360_settings
from .errors import (
    Data360MCPError,
    NotFoundError,
    ParseError,
    classify_error,
    is_transient_error,
)
from .errors import ValidationError as Data360ValidationError
from .http_client import RETRY_SAFE_EXTENSION, get_shared_httpx_client
from .models import (
    ComparisonSnapshot,
    ComparisonTimeSeries,
    CountryComparisonResponse,
    DatasetDescription,
    DatasetSearchRequest,
    DatasetSearchResponse,
    DataSummaryResponse,
    DerivedDataResponse,
    DiagnosticSummaryResponse,
    DiscoveryResult,
    EnrichedIndicator,
    EnrichedSearchResponse,
    ExcludedCountry,
    GroupSummary,
    IndicatorDataRequest,
    IndicatorDataResponse,
    MetadataRequest,
    MetadataResponse,
    MultiQuerySearchResponse,
    PivotTableResponse,
    QueryGroup,
    QueryGroupResult,
    RankedCountry,
    RankingResponse,
    SearchRequest,
    SearchResponse,
    SeriesDescription,
)
from .providers import get_database_mapping
from .validation import validated_country_codes

dotenv.load_dotenv()
_logger = logging.getLogger(__name__)

data360_config = get_data360_settings()

# Constants for API logic
COUNTRY_CODE_LENGTH = 3
SCORE_THRESHOLD = 70
MAX_RETURN_STATEMENTS = 6
DEFAULT_SEARCH_LIMIT = 5

# ---------------------------------------------------------------------------
# Metadata response cache with degraded serving (Comment 1+8 — avsolatorio PR #75)
# ---------------------------------------------------------------------------
# Metadata changes rarely; a 1-day TTL prevents stale data on long-running
# servers while eliminating redundant HTTP calls across multi-page aggregation
# fetches (e.g. _fetch_all_pages issues one get_data call per page, each of
# which would otherwise re-fetch the same metadata).
_METADATA_CACHE_TTL = 86_400  # 24 hours in seconds
_METADATA_CACHE_MAXSIZE = 256


class _ResilientTtlCache:
    """TTL cache that can serve a previous good value while the API is failing.

    Three bounded tiers:

    * ``_fresh`` — values younger than ``ttl``; the normal hit path.
    * ``_last_good`` — the most recent successful value per key, kept past TTL
      expiry so an upstream outage degrades to the last snapshot instead of an error.
    * ``_degraded`` — outcomes produced while the upstream was failing, reused for
      ``degraded_ttl`` seconds so a burst of callers does not re-hammer a slow
      dependency once per call.
    """

    def __init__(self, *, ttl: float, degraded_ttl: float, maxsize: int) -> None:
        self._fresh: cachetools.TTLCache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl)
        self._last_good: cachetools.LRUCache = cachetools.LRUCache(maxsize=maxsize)
        self._degraded: cachetools.TTLCache = cachetools.TTLCache(
            maxsize=maxsize, ttl=degraded_ttl
        )
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...]) -> Any:
        with self._lock:
            return self._fresh.get(key)

    def get_degraded(self, key: tuple[Any, ...]) -> Any:
        with self._lock:
            return self._degraded.get(key)

    def get_last_good(self, key: tuple[Any, ...]) -> Any:
        with self._lock:
            return self._last_good.get(key)

    def set(self, key: tuple[Any, ...], value: Any) -> None:
        """Record a successful value and clear any degraded outcome for the key."""
        with self._lock:
            self._fresh[key] = value
            self._last_good[key] = value
            self._degraded.pop(key, None)

    def set_degraded(self, key: tuple[Any, ...], value: Any) -> None:
        with self._lock:
            self._degraded[key] = value

    def clear(self) -> None:
        with self._lock:
            self._fresh.clear()
            self._last_good.clear()
            self._degraded.clear()


_metadata_cache = _ResilientTtlCache(
    ttl=_METADATA_CACHE_TTL,
    degraded_ttl=data360_config.degradation_cooldown_seconds,
    maxsize=_METADATA_CACHE_MAXSIZE,
)
_disaggregation_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=256, ttl=_METADATA_CACHE_TTL
)
_disaggregation_cache_lock = threading.Lock()

_DIMENSIONS_API_CACHE_TTL = 600  # 10 minutes in seconds
_dimensions_api_cache: cachetools.TTLCache = cachetools.TTLCache(
    maxsize=256, ttl=_DIMENSIONS_API_CACHE_TTL
)
_dimensions_api_cache_lock = threading.Lock()
_dimensions_api_inflight: dict[tuple[str, str], asyncio.Task] = {}
_dimensions_api_inflight_lock = threading.Lock()

# Fields fetched by _search_raw for LLM-friendly enrichment.
# Shared between single-query and multi-query paths to ensure consistency.
_ENRICHMENT_SELECT_FIELDS = [
    "idno",
    "name",
    "database_id",
    "definition_long",
    "periodicity",
    "time_periods",
    "ref_country",
    "dimensions",
    "measurement_unit",
]

# Prefiltering constants for get_data output.
# Based on a 16-database survey (see payload_analysis.md for full documentation).
# Always-keep fields: the core data the LLM needs.
_CORE_FIELDS = frozenset(
    {
        "OBS_VALUE",
        "TIME_PERIOD",
        "REF_AREA",
        "REF_AREA_NAME",
        "country_name",
        "UNIT_MEASURE",
        "UNIT_MEASURE_NAME",
        "UNIT_MULT",
        "claim_id",
    }
)
# Conditional fields: kept only when their value is non-trivial (not _T or _Z).
# SEX/AGE/URBANISATION carry real disaggregation in WB_HCP, WB_SSGD, OECD_IDD.
# COMP_BREAKDOWN_1/2 carry semantic data in IPC_IPC, OECD_BROADBAND, WB_SE4ALL, WEF_TTDI.
_CONDITIONAL_FIELDS = frozenset(
    {"SEX", "AGE", "URBANISATION", "COMP_BREAKDOWN_1", "COMP_BREAKDOWN_2"}
)
# SDMX standard codes meaning "total" and "not applicable".
_TRIVIAL_VALUES = frozenset({"_T", "_Z"})


def _short_hash(data: dict[str, Any]) -> str:
    """PCN claim_id 8-character hash for data verification."""
    return f"{zlib.crc32(json.dumps(data, sort_keys=True).encode()) & 0xFFFFFFFF:08x}"


def _qualify_unit_name(unit_name: str | None, unit_mult: Any, unit_code: str | None = None) -> str | None:
    """Qualify a unit name/label using its unit multiplier (e.g. 'million people')."""
    if not unit_mult:
        return unit_name
    try:
        mult = int(unit_mult)
    except (ValueError, TypeError):
        return unit_name

    if mult == 0:
        return unit_name

    mult_map = {
        3: "thousand",
        6: "million",
        9: "billion",
        12: "trillion"
    }
    qualifier = mult_map.get(mult)
    if not qualifier:
        return unit_name

    if not unit_name:
        return qualifier

    unit_norm = unit_name.strip().lower()
    is_people = (unit_code and unit_code.upper() == "PS") or (unit_norm in ("persons", "people"))
    if is_people:
        return f"{qualifier} people"

    return f"{qualifier} {unit_name}"


def _obs_value_to_float(val: Any) -> float | None:
    """Parse OBS_VALUE for aggregation paths; None if missing or non-numeric.

    The Data API sometimes returns the literal string ``\"null\"`` for missing
    values; ``val is not None`` is therefore not sufficient before ``float()``."""
    if val is None:
        return None
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped or stripped.lower() in ("null", "nan", "none"):
            return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _get_valid_disaggregations(
    disagg_res: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Get valid disaggregation options from the raw response."""
    null_values = ["_Z"]
    valid = []
    for field in disagg_res:
        if field.get("field_value", ["_T"])[0] in null_values:
            continue
        else:
            valid.append(field)
    return valid


def _parse_dimensions_response(dimensions_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate the new dimensions API response back to the old disaggregation structure."""
    raw_disaggregations = []
    null_values = {"_Z"}
    for dim in dimensions_data.get("dimensions", []):
        field_name = dim.get("field_name")
        label_name = dim.get("label_name")
        codes = []
        for val in dim.get("field_value", []):
            code = None
            if isinstance(val, dict) and "code" in val:
                code = val["code"]
            elif isinstance(val, str):
                code = val

            if code is not None and code not in null_values:
                codes.append(code)

        if field_name and codes:
            item = {
                "field_name": field_name,
                "field_value": codes,
            }
            if label_name is not None:
                item["label_name"] = label_name
            raw_disaggregations.append(item)
    return raw_disaggregations


async def _fetch_dimensions_raw_uncached(
    database_id: str,
    indicator_id: str,
) -> dict[str, Any]:
    """Fetch dimensions from the API without caching, raising exceptions on failure.

    Only updates the cache on success. Pops itself from inflight tasks on completion.
    """
    _cache_key = (database_id, indicator_id)
    try:
        dimensions_url = (
            data360_config.dimensions_url
            or f"{data360_config.api_url}/portal/v1/dimensions"
        )
        headers = {"accept": "*/*", "Content-Type": "application/json"}
        payload = {"database_id": database_id, "indicator_id": indicator_id}

        client = get_shared_httpx_client()
        response = await client.post(
            dimensions_url,
            json=payload,
            headers=headers,
            extensions={RETRY_SAFE_EXTENSION: True},
        )

        if response.status_code in (400, 404, 417):
            result = {"dimensions": []}
            with _dimensions_api_cache_lock:
                _dimensions_api_cache[_cache_key] = result
            return result

        response.raise_for_status()

        if not response.content or not response.text.strip():
            result = {"dimensions": []}
        else:
            result = response.json()

        with _dimensions_api_cache_lock:
            _dimensions_api_cache[_cache_key] = result

        return result
    finally:
        with _dimensions_api_inflight_lock:
            _dimensions_api_inflight.pop(_cache_key, None)


async def _fetch_dimensions_with_cache(
    database_id: str,
    indicator_id: str,
) -> dict[str, Any]:
    """Fetch dimensions from the API with a 10-minute in-memory cache and concurrent request deduplication.

    Only caches successful responses. Returns parsed JSON dict.
    Raises httpx.HTTPStatusError or other request exceptions on non-400/404 failures.
    """
    _cache_key = (database_id, indicator_id)
    with _dimensions_api_cache_lock:
        _cached = _dimensions_api_cache.get(_cache_key)
    if _cached is not None:
        return _cached

    with _dimensions_api_inflight_lock:
        task = _dimensions_api_inflight.get(_cache_key)
        if task is None:
            coro = _fetch_dimensions_raw_uncached(database_id, indicator_id)
            task = asyncio.create_task(coro)
            _dimensions_api_inflight[_cache_key] = task

    return await task


def _strip_data_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip boilerplate fields from a data row for LLM token savings.

    Keeps core fields (OBS_VALUE, TIME_PERIOD, REF_AREA, UNIT_MEASURE, claim_id)
    and conditionally includes disaggregation fields (SEX, AGE, URBANISATION,
    COMP_BREAKDOWN_1, COMP_BREAKDOWN_2) only when their value is non-trivial
    (i.e. not _T or _Z).

    Note: COMP_BREAKDOWN_3 is intentionally excluded -- it was observed as ``_Z``
    across all 16 surveyed databases and never carries data.

    Based on a 16-database survey documented in docs/payload_analysis.md.
    """
    filtered = {k: v for k, v in row.items() if k in _CORE_FIELDS}
    for field in _CONDITIONAL_FIELDS:
        val = row.get(field)
        if val and val not in _TRIVIAL_VALUES:
            filtered[field] = val
    return filtered


# Dimensions to always strip from disaggregation output.
_STRIP_DIMENSIONS = frozenset({"INDICATOR", "FREQ"})
# Dimensions where a single _T value means "no disaggregation available".
_TRIVIAL_SINGLE_DIMENSIONS = frozenset({"SEX", "AGE", "URBANISATION"})
# Maximum number of REF_AREA codes to include in the sample.
_REF_AREA_SAMPLE_SIZE = 5


def _strip_disaggregation(
    dimensions: list[dict[str, Any]],
    queried_countries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Strip bloat from disaggregation dimensions for LLM token savings.

    Rules:
    1. Remove INDICATOR and FREQ (always single-value, already known).
    2. Remove SEX/AGE/URBANISATION if only value is _T (no disaggregation).
    3. Sort TIME_PERIOD chronologically.
    4. Summarize REF_AREA: total count + which queried countries have data.
    """
    result = []
    for dim in dimensions:
        name = dim.get("field_name", "")
        values = dim.get("field_value", [])

        # Rule 1: strip trivial dimensions
        if name in _STRIP_DIMENSIONS:
            continue

        # Rule 2: strip single-value _T dimensions
        if name in _TRIVIAL_SINGLE_DIMENSIONS and values == ["_T"]:
            continue

        entry: dict[str, Any] = {"field_name": name}

        # Preserve label_name if present
        if "label_name" in dim:
            entry["label_name"] = dim["label_name"]

        # Rule 3: sort TIME_PERIOD
        if name == "TIME_PERIOD":
            entry["field_value"] = sorted(values)
        # Rule 4: summarize REF_AREA with country lookup
        elif name == "REF_AREA":
            entry["count"] = len(values)
            if queried_countries:
                ref_set = set(values)
                entry["queried"] = {code: code in ref_set for code in queried_countries}
            else:
                entry["sample"] = sorted(values)[:_REF_AREA_SAMPLE_SIZE]
        else:
            entry["field_value"] = values

        result.append(entry)
    return result


async def _resolve_queried_countries(
    required_country: str | None,
) -> list[str] | None:
    """Resolve a required_country string into a list of 3-letter codes.

    Returns None if required_country is falsy or resolution fails.
    """
    if not required_country:
        return None
    resolved = await _resolve_country_code(required_country)
    if not resolved:
        return None
    return [c.strip() for c in resolved.split(";") if c.strip()]


def _validate_user_filters(
    user_filters: dict[str, str | None] | None,
    available_disaggregations: dict[str, list[str]],
) -> tuple[dict[str, str | None], list[str]]:
    """Validate user filters against available options.

    Returns:
        Tuple of (valid_filters_dict, error_messages_list).
    """
    if not user_filters:
        return {}, []

    valid_filters = {}
    errors = []
    for dim, val in user_filters.items():
        # Skip special 'None' filters (meaning "all") or known non-dims like REF_AREA if we don't have metadata for them
        # Note: API metadata usually returns REF_AREA as a dimension too if valid.
        if val is None:
            valid_filters[dim] = None
            continue

        # Treat empty or whitespace-only strings as unspecified (skip to let smart defaults apply)
        if isinstance(val, str) and not val.strip():
            continue

        # REF_AREA arrives as a delimited code list, so it gets the same positive
        # validation as the dedicated country_code argument: a malformed value must
        # not reach the request (semicolons are the tool-side separator, commas the
        # API's). Anything unusable is reported instead of silently dropped.
        if dim == "REF_AREA" and isinstance(val, str):
            codes = validated_country_codes(val)
            if not codes:
                errors.append(
                    "REF_AREA filter must contain ISO-style country codes "
                    "(letters or digits), separated by ',' or ';'."
                )
                continue
            val = ",".join(codes)

        # If dimension exists in metadata, check value
        if dim in available_disaggregations:
            valid_values = available_disaggregations[dim]
            is_valid = True

            # Support comma-separated lists for any dimension
            if "," in val:
                parts = [p.strip() for p in val.split(",") if p.strip()]
                for part in parts:
                    if part not in valid_values:
                        errors.append(
                            f"Invalid value '{part}' in '{val}' for dimension '{dim}'. Available options: {valid_values}"
                        )
                        is_valid = False
            # Standard single value check
            elif val not in valid_values:
                errors.append(
                    f"Invalid value '{val}' for dimension '{dim}'. Available options: {valid_values}"
                )
                is_valid = False

            if is_valid:
                valid_filters[dim] = val
        else:
            # If dimension is unknown, we treat it as valid/passthrough for now
            # but we could also flag it. For consistency with previous behavior,
            # we'll keep it as valid.
            valid_filters[dim] = val

    return valid_filters, errors


def _build_disaggregation_params(
    disaggregation_filters: dict[str, str | None] | None,
    available_disaggregations: dict[str, list[str]] | None = None,
) -> dict[str, str]:
    """Build effective disaggregation params with smart defaults.

    This is the single source of truth for disaggregation filtering logic.
    Used by both get_data() and get_data_api_url().

    Args:
        disaggregation_filters: User-provided filters.
            - None or {}: Use defaults (SEX=_T, AGE=_T, URBANISATION=_T)
            - {"SEX": "F"}: Use F for SEX, defaults for others
            - {"SEX": None}: Omit SEX filter (get all values), defaults for others
        available_disaggregations: Optional dict of {dimension: [values]}
            derived from indicator metadata. If provided, defaults (like _T)
            are only applied if they exist in the available values.

    Returns:
        Dict of dimension -> value to add to API params.
        Dimensions with None values are omitted (API returns all).
    """
    effective = {}

    # 1. Start with user-provided filters
    if disaggregation_filters:
        for dim, raw_val in disaggregation_filters.items():
            # Skip FREQ as it's not used for filtering in this API
            if dim == "FREQ":
                continue
            if raw_val is not None:
                # Clean value: remove spaces around commas for multi-value support
                val = raw_val
                if isinstance(val, str) and "," in val:
                    val = ",".join([p.strip() for p in val.split(",") if p.strip()])
                effective[dim] = val

    # 2. Apply smart defaults for unspecified dimensions
    if available_disaggregations:
        for dim, values in available_disaggregations.items():
            # Skip if user already specified/excluded this dimension
            if disaggregation_filters and dim in disaggregation_filters:
                continue

            # Check if this dimension has a "Total" option (_T)
            if "_T" in values:
                # Redundancy check: if _T is the ONLY option, don't force it
                if len(values) == 1:
                    continue

                # Otherwise, apply default
                effective[dim] = "_T"

    else:
        # Fallback for when metadata wasn't fetched or failed.
        # We CANNOT safely apply defaults like AGE=_T because we don't know if they exist.
        # It is better to return ALL data (no filter) than NONE (invalid filter).
        # So we leave 'effective' as-is (containing only user provided filters).
        pass

    return effective


def _get_items_from_response(response_data: dict[str, Any]) -> list[SeriesDescription]:
    """Extract and validate series descriptions from API response.

    Supports both legacy nested SearchV2 format and flat SearchV3 format.
    """
    values = response_data.get("value")
    if values is None:
        values = response_data.get("results")
    if values is None:
        values = response_data.get("items", [])

    items = []
    for value in values:
        if "series_description" in value:
            # V2 format
            series_description = value.get("series_description", {})
            additional = value.get("additional")
            if additional and isinstance(additional, dict):
                ml = additional.get("metadata_link")
                if ml and isinstance(ml, list):
                    series_description["metadata_link"] = ml
        else:
            # V3 flat format
            databases = value.get("databases", [])
            db_id = None
            if databases and isinstance(databases, list) and len(databases) > 0:
                db_id = (
                    databases[0].get("idno") if isinstance(databases[0], dict) else None
                )

            tp = value.get("time_period")
            time_periods = None
            if tp:
                time_periods = tp if isinstance(tp, list) else [tp]

            dims = value.get("dimensions", [])
            dimensions = []
            if isinstance(dims, list):
                for d in dims:
                    if isinstance(d, dict):
                        dimensions.append(d)
                    elif isinstance(d, str):
                        dimensions.append({"label": d})

            ml = value.get("metadata_link")
            if ml is None:
                additional = value.get("additional")
                if isinstance(additional, dict):
                    ml = additional.get("metadata_link")

            series_description = {
                "idno": value.get("idno"),
                "name": value.get("name"),
                "database_id": db_id or value.get("database_id"),
                "definition_long": value.get("description")
                or value.get("definition_long"),
                "periodicity": value.get("frequency") or value.get("periodicity"),
                "time_periods": time_periods,
                "ref_country": value.get("ref_country"),
                "dimensions": dimensions,
                "metadata_link": ml or [],
                "connected_entities": value.get("connected_entities")
                if isinstance(value.get("connected_entities"), list)
                else None,
            }

        if (
            series_description
            and series_description.get("idno")
            and series_description.get("name")
            and series_description.get("database_id")
        ):
            try:
                items.append(SeriesDescription.model_validate(series_description))
            except Exception as e:
                _logger.warning(
                    f"Failed to validate series_description: [{repr(series_description)}], error: {repr(e)}, skipping item"
                )
    return items


def _process_search_response(
    response_data: dict[str, Any], request: SearchRequest
) -> SearchResponse:
    """Process API response and build SearchResponse."""
    search_response_data = {
        "items": _get_items_from_response(response_data),
        "total_count": response_data.get("count")
        if "count" in response_data
        else response_data.get("@odata.count"),
        "offset": request.offset,
    }
    search_response_data["count"] = len(search_response_data["items"])

    # Calculate has_more and next_offset
    if (
        search_response_data["total_count"] is not None
        and search_response_data["total_count"] > request.offset + request.limit
    ):
        search_response_data["has_more"] = True
        search_response_data["next_offset"] = request.offset + request.limit
    else:
        search_response_data["has_more"] = False
        search_response_data["next_offset"] = None

    return SearchResponse.model_validate(search_response_data)


async def _search_raw(
    query: str,
    limit: int = 5,
    offset: int = 0,
    count: bool = True,
    economy_codes: list[str] | None = None,
    database: str | None = None,
) -> SearchResponse:
    """Internal: Raw search for data360 indicators using the World Bank Data360 API.

    This is the low-level API. Use `search()` for the enriched LLM-friendly version.

    Args:
        query: Search query string to find relevant data series
        limit: Number of results to return (default is 5)
        offset: Offset of the current page
        count: Whether to include total count in response
        economy_codes: Optional list of economy codes to filter the search results
        database: Optional database filter name or ID

    Returns:
        SearchResponse with raw API results.
    """
    database_names = []
    if database:
        from .providers import get_database_manager
        db_mgr = get_database_manager()
        try:
            db_ids = db_mgr.resolve_database_ids(database)
            mapping = await db_mgr.get_mapping()
            for db_id in db_ids:
                db_name = mapping.get(db_id)
                if db_name:
                    database_names.append(db_name)
        except ValueError as e:
            return SearchResponse(
                error=str(e),
                items=[],
                total_count=0,
                count=0,
            )

    request = SearchRequest(
        query=query,
        limit=limit,
        offset=offset,
        count=count,
    )

    url = (
        data360_config.search_url
        or f"{data360_config.api_url}/portal/v1/public_data360_search"
    )
    payload = {
        "site": "data360",
        "query_string": request.query,
        "types": ["indicator"],
        "data_classification": ["public"],
        "skip": request.offset,
        "items_per_page": request.limit,
    }
    if economy_codes:
        payload["economy_codes"] = economy_codes
    if database_names:
        payload["database_names"] = database_names

    mcp_error: Data360MCPError | None = None
    try:
        client = get_shared_httpx_client()
        response = await client.post(
            url, json=payload, extensions={RETRY_SAFE_EXTENSION: True}
        )
        response.raise_for_status()

        try:
            response_data = response.json()
        except ValueError as e:
            raise ParseError(context="search", original_error=e)

        return _process_search_response(response_data, request)

    except Data360MCPError:
        # Re-raise our own errors to propagate isError: true to MCP client
        raise
    except Exception as e:
        # Convert unknown exceptions to Data360MCPError and raise
        raise classify_error(e, context="search")


async def _resolve_country_code(country_query: str) -> str | None:
    """Resolve country name to code using cached REF_AREA codelist."""
    from . import providers as data360_providers  # noqa: PLC0415

    if not country_query:
        return None

    # Handle multi-country: semicolons are the user-facing delimiter because
    # some country names contain commas (e.g. "Korea, Republic of").
    if ";" in country_query:
        parts = [p.strip() for p in country_query.split(";") if p.strip()]
        resolved_codes = []
        for part in parts:
            code = await _resolve_country_code(part)
            if code:
                resolved_codes.append(code)

        return ";".join(resolved_codes) if resolved_codes else None

    # Already a 3-letter code
    if len(country_query) == COUNTRY_CODE_LENGTH and country_query.isupper():
        return country_query
    # Look up in codelist
    matches = await data360_providers.find_codelist_value(
        "REF_AREA", country_query, limit=1
    )
    if matches and matches[0].get("score", 0) >= SCORE_THRESHOLD:
        return matches[0].get("id")
    return None


def _enrich_search_results(
    search_result: "SearchResponse",
    country_code: str | None,
    db_mapping: dict[str, str] | None = None,
    query: str | None = None,
) -> tuple[list["EnrichedIndicator"], list["EnrichedIndicator"]]:
    """Convert raw SearchResponse items into a list of EnrichedIndicator objects.

    Extracted to avoid duplicating enrichment logic between the single-query
    and multi-query paths in search().

    Args:
        search_result: Raw response from _search_raw().
        country_code: Resolved country code (or None). Used to compute covers_country.
        db_mapping: Optional dict mapping database_id -> human-readable name.

    Returns:
        Tuple of (indicators, indicators_to_verify).
        indicators: Full list of EnrichedIndicator objects.
        indicators_to_verify: Subset where covers_country has False entries that
            may be wrong because the search API omits group codes from ref_country.
    """
    if not search_result.items:
        return [], []

    _db = db_mapping or {}

    _label_to_code = {
        "sex": "SEX",
        "age": "AGE",
        "residential area": "URBANISATION",
        "urbanisation": "URBANISATION",
        "education": "EDUCATION",
    }

    # Determine upfront which requested codes are known groups/WLD — the search
    # API's ref_country array omits these even when the data endpoint supports them.
    from .providers import get_group_hierarchy_manager

    _group_manager = get_group_hierarchy_manager()
    _regional_codes: set[str] = set()
    if country_code:
        for c in country_code.split(";"):
            c = c.strip()
            if c and (c == "WLD" or _group_manager.is_group(c)):
                _regional_codes.add(c)

    indicators: list[EnrichedIndicator] = []
    indicators_to_verify: list[EnrichedIndicator] = []
    for item in search_result.items:
        raw = item.model_dump()

        # Extract latest_data and time_period_range
        time_periods = raw.get("time_periods", [])
        latest_data = None
        time_period_range = None
        if time_periods and isinstance(time_periods, list):
            tp = time_periods[0] if isinstance(time_periods[0], dict) else {}
            latest_data = tp.get("LATEST_DATA_POINT") or tp.get("end")
            start = tp.get("start")
            end = tp.get("end")
            if start and end:
                time_period_range = f"{start}-{end}"

        covers_country: dict[str, bool] | None = None
        if country_code:
            requested_codes = [c.strip() for c in country_code.split(";") if c.strip()]
            ref_list = raw.get("ref_country")
            # Under SearchV3, ref_country is absent or None.
            is_search_v3 = "ref_country" not in raw or raw["ref_country"] is None

            if len(requested_codes) == 1 and is_search_v3:
                # O(1) Optimization: Single-country queries filtered by economy_codes
                # are guaranteed to cover that country under SearchV3.
                covers_country = {requested_codes[0]: True}
            else:
                ref_countries = set()
                if ref_list and isinstance(ref_list, list):
                    for rc in ref_list:
                        if isinstance(rc, dict) and rc.get("code"):
                            ref_countries.add(rc["code"])
                        elif isinstance(rc, str):
                            ref_countries.add(rc)
                covers_country = {
                    code: (code in ref_countries) for code in requested_codes
                }

        # Extract dimension names
        dimensions = raw.get("dimensions", [])
        useful_dims: list[str] = []
        if dimensions and isinstance(dimensions, list):
            for dim in dimensions:
                if isinstance(dim, dict):
                    label = (dim.get("label") or "").lower()
                    if label in _label_to_code:
                        useful_dims.append(_label_to_code[label])

        # Apply primary indicator redirect if metadata_link has type='primary'
        # and a valid database_id (some entries have database_id=None).
        primary = item.primary_source
        original_idno: str | None = None
        if primary and primary.metadata_id and primary.database_id:
            original_idno = raw.get("idno")
            # Overwrite with primary source coordinates; raw["database_id"] is
            # read back into db_id on the next unconditional line below.
            raw["idno"] = primary.indicator_id
            raw["database_id"] = primary.database_id
            _logger.debug(
                "Redirected %s -> %s/%s (primary source)",
                repr(original_idno),
                repr(primary.database_id),
                repr(primary.indicator_id),
            )
        elif primary and primary.metadata_id and not primary.database_id:
            # database_id is None (e.g. META_SI.POV.MPWB) — cannot redirect
            # without a target database. Log for observability.
            _logger.warning(
                "Skipping primary redirect for %s: metadata_link has type='primary' "
                "but database_id is None (metadata_id=%s)",
                repr(raw.get("idno")),
                repr(primary.metadata_id),
            )
        elif not primary and query and isinstance(raw.get("connected_entities"), list):
            # Check if query matches a connected entity (SearchV3 redirect direction is reversed)
            clean_query = query.strip().upper()
            for entity in raw["connected_entities"]:
                if (
                    isinstance(entity, dict)
                    and entity.get("idno", "").upper() == clean_query
                ):
                    original_idno = entity.get("idno")
                    _logger.debug(
                        "Mapped primary source %s -> %s/%s via connected_entities for query %s",
                        repr(original_idno),
                        repr(raw.get("database_id")),
                        repr(raw.get("idno")),
                        repr(query),
                    )
                    break

        db_id = raw.get("database_id", "")
        ind = EnrichedIndicator(
            idno=raw.get("idno", ""),
            database_id=db_id,
            database_name=_db.get(db_id),
            name=raw.get("name", ""),
            truncated_definition=(raw.get("definition_long") or "")[:100],
            unit=raw.get("measurement_unit"),
            periodicity=raw.get("periodicity"),
            latest_data=latest_data,
            time_period_range=time_period_range,
            covers_country=covers_country,
            dimensions=useful_dims if useful_dims else None,
            primary_source_of=original_idno,
        )
        indicators.append(ind)

        # Flag for verification if country_code is provided.
        if country_code and covers_country is not None:
            requested_codes = [c.strip() for c in country_code.split(";") if c.strip()]
            if len(requested_codes) > 1:
                indicators_to_verify.append(ind)

    # Set requested_country on all indicators
    for ind in indicators:
        ind.requested_country = country_code

    # Do not deduplicate here. `_enrich_search_results()` is used by both
    # single-query and multi-query flows, and multi-query needs access to the
    # full post-redirect candidate list so its `dedupe` flag and candidate
    # counts remain accurate. Any deduplication should happen in the caller
    # that owns the response semantics.
    return indicators, indicators_to_verify


async def _backfill_primary_metadata(
    redirected: "list[EnrichedIndicator]",
) -> None:
    """Fetch fresh metadata for redirected primary indicators and update
    latest_data / time_period_range in place.

    When a secondary indicator is redirected to its primary source, the
    time-period data carried by the search result still belongs to the
    secondary (which may be a frozen snapshot while the primary is
    actively updated). This helper fetches the real time_periods from
    the primary indicator's metadata and patches both fields.

    Duplicate primary targets (multiple secondaries pointing to the same
    primary) are collapsed to a single fetch via the metadata cache.

    Args:
        redirected: EnrichedIndicator objects whose idno/database_id were
            rewritten to point at a primary source (identified by
            primary_source_of is not None).
    """
    if not redirected:
        return

    # Deduplicate by (database_id, idno) so we issue at most one metadata
    # request per unique primary target. The metadata cache also handles
    # this, but deduplicating here avoids concurrent redundant fetches.
    seen: set[tuple[str, str]] = set()
    unique_redirected: list[EnrichedIndicator] = []
    for ind in redirected:
        key = (ind.database_id, ind.idno)
        if key not in seen:
            seen.add(key)
            unique_redirected.append(ind)

    async def _fetch_and_patch(ind: "EnrichedIndicator") -> None:
        try:
            meta = await get_metadata(
                database_id=ind.database_id,
                indicator_id=ind.idno,
                select_fields=["time_periods"],
                fetch_disaggregation=False,
            )
            if meta.indicator_metadata:
                time_periods = meta.indicator_metadata.get("time_periods", [])
                if time_periods and isinstance(time_periods, list):
                    tp = time_periods[0] if isinstance(time_periods[0], dict) else {}
                    ind.latest_data = tp.get("LATEST_DATA_POINT") or tp.get("end")
                    start = tp.get("start")
                    end = tp.get("end")
                    if start and end:
                        ind.time_period_range = f"{start}-{end}"
                    _logger.debug(
                        "Backfilled primary metadata for %s/%s: latest=%s range=%s",
                        repr(ind.database_id),
                        repr(ind.idno),
                        repr(ind.latest_data),
                        repr(ind.time_period_range),
                    )
        except Exception as e:
            _logger.warning(
                "Failed to backfill primary metadata for %s/%s: %s",
                repr(ind.database_id),
                repr(ind.idno),
                repr(e),
            )

    await asyncio.gather(*(_fetch_and_patch(ind) for ind in unique_redirected))


async def search(  # noqa: PLR0911
    query: str | None = None,
    required_country: str | None = None,
    limit: int = DEFAULT_SEARCH_LIMIT,
    offset: int = 0,
    # Multi-query parameters
    queries: list[str] | None = None,
    query_groups: list[QueryGroup] | None = None,
    result_layout: str = "merged",
    dedupe: bool = True,
    database: str | None = None,
    # The following parameters are accepted for robustness; LLM clients sometimes
    # hallucinate them from the internal SearchRequest model.
    # n_results/skip are treated as aliases for limit/offset when the primary
    # parameter is still at its default; the rest are silently ignored.
    count: bool = True,
    n_results: int | None = None,
    filter: str | None = None,  # noqa: A002 - name must match LLM-hallucinated param
    orderby: str | None = None,
    select: str | None = None,
    skip: int | None = None,
    odata_options: dict[str, str] | None = None,
) -> "EnrichedSearchResponse | MultiQuerySearchResponse":
    """Search for Data360 indicators with enriched metadata for selection.

    Use this first when the user asks for data on a topic (e.g. unemployment, poverty, GDP).
    No other tools are required before this one.

    ENRICHED DATA VS. FETCHING DATA:
    - For METADATA questions (e.g. "What is the definition of the unemployment rate indicator?", "How frequently is it updated?"): The enriched data returned by this search tool is often sufficient! You can directly use the `truncated_definition`, `name`, `periodicity`, `database_id`, and `latest_data` fields from the search results to answer the user WITHOUT needing to call `data360_get_metadata` or `data360_get_data`.
    - For DATA questions (e.g. "What was Kenya's GDP in 2020?", "Show me the trend of poverty"): The enriched data does NOT contain actual data values (OBS_VALUE). You MUST proceed to call `data360_get_disaggregation` and then `data360_get_data` (or `data360_get_viz_spec` for charts) to retrieve real numbers.

    Use when the user already names a specific indicator or metric — for example:
    "GDP per capita for Kenya", "unemployment rate in Morocco", "life expectancy in Sub-Saharan Africa".

    For multiple topics in one call (e.g. "GDP, inflation, employment for Kenya"), pass them as
    queries=["GDP growth", "inflation rate", "unemployment"] instead of making separate calls.

    Do NOT use this tool when the user asks a broad or vague question that does not name a
    specific indicator — for example: "What makes a country great?",
    "What are Ghana's economic challenges?", "How is education performing in Africa?"
    In those cases, use data360_analyze_development_topic instead, which decomposes
    the question into specific sub-queries and searches for each one.

    PARAMETER SELECTION — follow this decision tree strictly:
    1. ONE topic, any number of countries → use `query` + `required_country`.
    2. MULTIPLE topics, ALL in the SAME country → use `queries` + `required_country`.
    3. Topics targeting DIFFERENT countries → MUST use `query_groups`. Do NOT use `queries`.
       Example — "GDP for Japan and population for Philippines":
         query_groups=[
           {"queries": ["GDP per capita"], "country": "Japan"},
           {"queries": ["population"], "country": "Philippines"}
         ]
       Using `queries` for cross-country requests will lose per-country coverage data.

    Pass exactly ONE of query, queries, or query_groups. Omit the other two entirely.

    Args:
        query: Single search query (e.g., "unemployment rate", "poverty", "GDP per capita").
            Use this for ONE topic. If using this, do not pass queries or query_groups.
        queries: List of search terms for multi-topic search in one call (e.g.
            ["GDP growth", "inflation rate", "unemployment"]).
            Use ONLY when ALL topics target the SAME country (set via required_country).
            If topics span different countries, use query_groups instead.
            Requires at least 2 non-empty strings.
            If using this, do not pass query or query_groups.
        query_groups: List of QueryGroup objects, each binding one or more search terms to
            an optional country scope. Use when different queries target different countries.
            Use this instead of queries when each query targets a different country.
            JSON schema for each group: {"queries": ["<term1>", "<term2>"], "country": "<name or 3-letter code>"}
            Example: [
                {"queries": ["GDP per capita", "inflation"], "country": "Kenya"},
                {"queries": ["Gini coefficient"], "country": "Morocco"}
            ]
            Requires at least 2 non-empty queries total across all groups.
            If using this, do not pass query or queries. required_country is ignored.
        required_country: Optional country name or 3-letter code (e.g. "Kenya", "KEN").
            Use semicolon-separated names or codes to check multiple countries in one call
            (e.g. "China; USA"). Semicolons are used because some country names contain
            commas (e.g. "Korea, Republic of").
            Shared across all queries — only when all topics share the same geographic scope.
            Ignored when query_groups is used (each group has its own country).
        limit: Maximum number of indicators per query (default 5).
        offset: Number of results to skip per query for pagination (default 0).
        result_layout: Only used with queries/query_groups.
            Use "merged" (default) when you want a single flat list to pick from.
            Use "by_query" when you need to know which indicators came from which query —
            e.g. to attribute country coverage per query or display grouped results.
        dedupe: Only used with queries/query_groups. When True (default), deduplicates by
            (database_id, idno) across all query groups. First-seen order is preserved.

    Returns:
        With query: EnrichedSearchResponse with indicators, required_country, pagination fields.
            Each indicator has covers_country (dict[str, bool], e.g. {\"KEN\": True}) and
            requested_country (resolved semicolon-separated code string).
        With queries/query_groups: MultiQuerySearchResponse with indicators (merged) or
            results (by_query), total_candidates, deduplicated_count, and per-group errors.
            Each indicator has requested_country showing which group's country it was evaluated against.
        error: Error message string if the request failed; otherwise None.
    """

    # --- Validation ---
    # Normalise LLM-hallucinated empty defaults before mode detection.
    # When a client sends query="" or queries=[] alongside the real parameter
    # (e.g. query_groups), treat these as "not provided" — identical to None.
    if query is not None and not query.strip():
        _logger.debug("query=%s normalised to None (empty/whitespace-only)", repr(query))
        query = None
    if queries is not None and not any(q and q.strip() for q in queries):
        _logger.debug("queries=%s normalised to None (all entries empty)", repr(queries))
        queries = None
    if query_groups is not None and not query_groups:
        _logger.debug("query_groups=%s normalised to None (empty list)", repr(query_groups))
        query_groups = None

    if query_groups is not None:
        parsed_groups = []
        for g in query_groups:
            if isinstance(g, dict):
                try:
                    parsed_groups.append(QueryGroup(**g))
                except Exception as e:
                    return MultiQuerySearchResponse(
                        error=f"Invalid QueryGroup structure: {e}",
                        queries=[],
                    )
            elif isinstance(g, QueryGroup):
                parsed_groups.append(g)
            else:
                return MultiQuerySearchResponse(
                    error="query_groups must be a list of QueryGroup objects or dictionaries.",
                    queries=[],
                )
        query_groups = parsed_groups

    active_modes = sum(
        (
            query is not None,
            queries is not None,
            query_groups is not None,
        )
    )
    if active_modes > 1:
        return EnrichedSearchResponse(
            error="Provide exactly one of 'query', 'queries', or 'query_groups', not multiple."
        )
    if active_modes == 0:
        return EnrichedSearchResponse(
            error="Missing search term. You must provide exactly one of 'query', 'queries', or 'query_groups' to search for indicators, even when filtering by database."
        )

    # --- Multi-query path (queries= flat list) ---
    if queries is not None:
        clean_queries = [q.strip() for q in queries if q and q.strip()]
        if len(clean_queries) < len(queries):
            _logger.warning(
                "Stripped %d empty/whitespace-only entries from queries "
                "(original: %d, kept: %d)",
                len(queries) - len(clean_queries),
                len(queries),
                len(clean_queries),
            )
        if len(clean_queries) < 2:  # noqa: PLR2004
            return MultiQuerySearchResponse(
                error="'queries' must contain at least 2 non-empty search strings.",
                queries=queries or [],
            )
        if result_layout not in ("merged", "by_query"):
            return MultiQuerySearchResponse(
                error="result_layout must be 'merged' or 'by_query'.",
                queries=clean_queries,
            )

        # Handle limit/offset aliases (mirror single-query path warnings)
        if n_results is not None:
            if limit == DEFAULT_SEARCH_LIMIT:
                limit = n_results
            elif limit != n_results:
                _logger.warning(
                    "Both limit=%d and n_results=%d provided; using limit",
                    limit,
                    n_results,
                )
        if skip is not None:
            if offset == 0:
                offset = skip
            elif offset != skip:
                _logger.warning(
                    "Both offset=%d and skip=%d provided; using offset",
                    offset,
                    skip,
                )

        # Resolve shared country once for all sub-queries
        country_code: str | None = None
        if required_country:
            country_code = await _resolve_country_code(required_country)

        # Each query gets the same country code
        per_query_codes: list[str | None] = [country_code] * len(clean_queries)

        # Fan out concurrent _search_raw calls, one per query
        raw_tasks = [
            _search_raw(
                query=q,
                limit=limit,
                offset=offset,
                economy_codes=[c.strip() for c in country_code.split(";")]
                if country_code
                else None,
                database=database,
            )
            for q in clean_queries
        ]
        raw_results = await asyncio.gather(*raw_tasks, return_exceptions=True)

        db_mapping = await get_database_mapping()
        return await _build_multi_query_response(
            clean_queries=clean_queries,
            raw_results=raw_results,
            per_query_codes=per_query_codes,
            result_layout=result_layout,
            dedupe=dedupe,
            db_mapping=db_mapping,
        )

    # --- query_groups path ---
    if query_groups is not None:
        if required_country:
            _logger.warning(
                "required_country is ignored when query_groups is used; "
                "set country per QueryGroup instead."
            )

        # Flatten groups into (query, raw_country) pairs, stripping empty entries
        flat_pairs: list[tuple[str, str | None]] = []
        for group in query_groups:
            for q in group.queries:
                stripped = q.strip() if q else ""
                if stripped:
                    flat_pairs.append((stripped, group.country))
                else:
                    _logger.warning(
                        "Empty/whitespace-only query stripped from query_groups."
                    )

        if len(flat_pairs) < 2:  # noqa: PLR2004
            return MultiQuerySearchResponse(
                error="query_groups must produce at least 2 non-empty queries total.",
                queries=[fp[0] for fp in flat_pairs],
            )
        if result_layout not in ("merged", "by_query"):
            return MultiQuerySearchResponse(
                error="result_layout must be 'merged' or 'by_query'.",
                queries=[fp[0] for fp in flat_pairs],
            )

        # Handle limit/offset aliases
        if n_results is not None:
            if limit == DEFAULT_SEARCH_LIMIT:
                limit = n_results
            elif limit != n_results:
                _logger.warning(
                    "Both limit=%d and n_results=%d provided; using limit",
                    limit,
                    n_results,
                )
        if skip is not None:
            if offset == 0:
                offset = skip
            elif offset != skip:
                _logger.warning(
                    "Both offset=%d and skip=%d provided; using offset",
                    offset,
                    skip,
                )

        # Resolve unique country values concurrently to avoid redundant codelist lookups
        unique_countries = list({c for _, c in flat_pairs if c})
        resolved_map: dict[str, str | None] = {}
        if unique_countries:
            codes = await asyncio.gather(
                *[_resolve_country_code(c) for c in unique_countries]
            )
            resolved_map = dict(zip(unique_countries, codes))

        clean_queries = [q for q, _ in flat_pairs]
        per_query_codes = [resolved_map.get(c) if c else None for _, c in flat_pairs]

        # Fan out concurrent _search_raw calls, one per flattened query
        raw_tasks = [
            _search_raw(
                query=q,
                limit=limit,
                offset=offset,
                economy_codes=[c.strip() for c in code.split(";")] if code else None,
                database=database,
            )
            for q, code in zip(clean_queries, per_query_codes)
        ]
        raw_results = await asyncio.gather(*raw_tasks, return_exceptions=True)

        db_mapping = await get_database_mapping()
        return await _build_multi_query_response(
            clean_queries=clean_queries,
            raw_results=raw_results,
            per_query_codes=per_query_codes,
            result_layout=result_layout,
            dedupe=dedupe,
            db_mapping=db_mapping,
        )

    # --- Single-query path (original behavior, fully preserved) ---
    # Handle common parameter aliases sent by LLM clients.
    # Aliases only apply when the primary parameter is at its default value;
    # an explicit limit/offset always takes precedence over n_results/skip.
    if n_results is not None:
        if limit == DEFAULT_SEARCH_LIMIT:
            limit = n_results
        elif limit != n_results:
            _logger.warning(
                "Both limit=%d and n_results=%d provided; using limit",
                limit,
                n_results,
            )
    if skip is not None:
        if offset == 0:
            offset = skip
        elif offset != skip:
            _logger.warning(
                "Both offset=%d and skip=%d provided; using offset",
                offset,
                skip,
            )

    # NOTE: This function is for MVP only, we should be testing the relevance and performance
    # of the retrieval process in the future.
    # Resolve country code upfront using cached codelist
    country_code = None
    if required_country:
        country_code = await _resolve_country_code(required_country)

    # Fetch all needed metadata in ONE search call
    try:
        search_result = await _search_raw(
            query=query,  # type: ignore[arg-type]  # validated non-None above
            limit=limit,
            offset=offset,
            economy_codes=[c.strip() for c in country_code.split(";")]
            if country_code
            else None,
            database=database,
        )
    except PydanticValidationError as e:
        return EnrichedSearchResponse(error=str(e))
    except Data360MCPError as e:
        return EnrichedSearchResponse(error=e.detail)

    if search_result.error:
        return EnrichedSearchResponse(error=search_result.error)

    if not search_result.items:
        return EnrichedSearchResponse(
            error=f"No indicators found for: '{query}'",
            total_count=search_result.total_count,
            count=0,
            offset=search_result.offset,
            has_more=search_result.has_more,
            next_offset=search_result.next_offset,
        )

    db_mapping = await get_database_mapping()
    indicators, indicators_to_verify = _enrich_search_results(
        search_result, country_code, db_mapping, query=query
    )

    # Backfill latest_data / time_period_range for redirected indicators so
    # the LLM sees the primary source's actual data range, not the secondary's.
    redirected = [ind for ind in indicators if ind.primary_source_of is not None]
    await _backfill_primary_metadata(redirected)

    # Secondary verification pass: the search API's ref_country array omits group
    # aggregate codes (SAS, WLD, etc.) even when the data endpoint supports them.
    # For flagged indicators, concurrently query the disaggregation endpoint to
    # determine the definitive True/False for each regional code.
    if country_code and indicators_to_verify:
        regional_codes = [c.strip() for c in country_code.split(";") if c.strip()]

        async def _verify_regional_coverage(ind: EnrichedIndicator) -> None:
            try:
                res = await get_disaggregation(
                    ind.database_id, ind.idno, required_country=country_code
                )
                for dim in res.get("dimensions", []):
                    if dim.get("field_name") == "REF_AREA":
                        queried = dim.get("queried", {})
                        if ind.covers_country is not None:
                            for c in regional_codes:
                                if c in ind.covers_country:
                                    ind.covers_country[c] = queried.get(c, False)
                        break
            except Exception as e:
                _logger.warning(
                    "Failed to verify regional coverage for %s: %s", repr(ind.idno), repr(e)
                )

        await asyncio.gather(
            *(_verify_regional_coverage(ind) for ind in indicators_to_verify)
        )

        indicators.sort(
            key=lambda x: (
                not any((x.covers_country or {}).values()),
                -(int(x.latest_data or 0) if str(x.latest_data or "").isdigit() else 0),
            )
        )

    name_map = None
    if country_code:
        name_map = await _resolve_country_names([c.strip() for c in country_code.split(";") if c.strip()])

    return EnrichedSearchResponse(
        indicators=indicators,
        required_country=country_code,
        country_names=name_map,
        # Map pagination fields from underlying search result
        count=search_result.count,
        total_count=search_result.total_count,
        offset=search_result.offset,
        has_more=search_result.has_more,
        next_offset=search_result.next_offset,
    )


async def _build_multi_query_response(
    clean_queries: list[str],
    raw_results: list[Any],
    per_query_codes: list[str | None],
    result_layout: str,
    dedupe: bool,
    db_mapping: dict[str, str] | None = None,
) -> MultiQuerySearchResponse:
    """Shared response builder for queries= and query_groups= paths.

    Encapsulates enrichment, deduplication, and layout selection so both
    paths stay in sync without code duplication.
    """
    seen_dict: dict[tuple[str, str], EnrichedIndicator] = {}
    groups: list[QueryGroupResult] = []
    total_candidates = 0
    deduplicated_count = 0

    # Collect indicators to verify concurrently
    verify_tasks = []

    # Helper to verify indicators for a specific country code
    async def _verify_coverage(ind: EnrichedIndicator, code: str) -> None:
        try:
            codes = [c.strip() for c in code.split(";") if c.strip()]
            res = await get_disaggregation(
                ind.database_id, ind.idno, required_country=code
            )
            for dim in res.get("dimensions", []):
                if dim.get("field_name") == "REF_AREA":
                    queried = dim.get("queried", {})
                    if ind.covers_country is not None:
                        for c in codes:
                            if c in ind.covers_country:
                                ind.covers_country[c] = queried.get(c, False)
                    break
        except Exception as e:
            _logger.warning("Failed to verify coverage for %s: %s", repr(ind.idno), repr(e))

    # First pass: enrich indicators and collect verification tasks
    enriched_lists = []
    for i, (q, raw_result) in enumerate(zip(clean_queries, raw_results)):
        code_for_query = per_query_codes[i]
        if (
            isinstance(raw_result, Exception)
            or raw_result.error
            or not raw_result.items
        ):
            enriched_lists.append((None, None))
            continue

        enriched, to_verify = _enrich_search_results(
            raw_result, code_for_query, db_mapping, query=q
        )
        # Backfill latest_data / time_period_range for any redirected indicators.
        redirected = [ind for ind in enriched if ind.primary_source_of is not None]
        await _backfill_primary_metadata(redirected)

        if code_for_query and to_verify:
            for ind in to_verify:
                verify_tasks.append(_verify_coverage(ind, code_for_query))

        enriched_lists.append((enriched, code_for_query))

    # Run verification tasks concurrently
    if verify_tasks:
        await asyncio.gather(*verify_tasks)

    # Second pass: build response groups and deduplicate
    for i, (q, raw_result) in enumerate(zip(clean_queries, raw_results)):
        if isinstance(raw_result, Exception):
            groups.append(
                QueryGroupResult(
                    query=q,
                    country_code=per_query_codes[i],
                    error=str(raw_result),
                )
            )
            continue
        if raw_result.error or not raw_result.items:
            groups.append(
                QueryGroupResult(
                    query=q,
                    country_code=per_query_codes[i],
                    error=raw_result.error or f"No indicators found for: '{q}'",
                )
            )
            continue

        enriched, code_for_query = enriched_lists[i]
        total_candidates += len(enriched)

        group_indicators: list[EnrichedIndicator] = []
        group_seen: set[tuple[str, str]] = set()

        for ind in enriched:
            key = (ind.database_id, ind.idno)

            if key in seen_dict:
                existing_ind = seen_dict[key]

                # Merge covers_country data across queries/groups
                if (
                    existing_ind.covers_country is not None
                    and ind.covers_country is not None
                ):
                    existing_ind.covers_country.update(ind.covers_country)

                if dedupe and result_layout == "merged":
                    deduplicated_count += 1
                elif dedupe and key in group_seen:
                    deduplicated_count += 1
                else:
                    group_seen.add(key)
                    group_indicators.append(existing_ind)
            else:
                seen_dict[key] = ind
                group_seen.add(key)
                group_indicators.append(ind)

        groups.append(
            QueryGroupResult(
                query=q,
                country_code=code_for_query,
                indicators=group_indicators,
                count=len(group_indicators),
            )
        )

    # Compute response-level required_country: join all unique resolved codes with ";"
    all_codes = sorted({c for c in per_query_codes if c})
    response_country = ";".join(all_codes) if all_codes else None

    # Resolve all individual codes from all queries
    individual_codes = set()
    for code in all_codes:
        individual_codes.update(c.strip() for c in code.split(";") if c.strip())
    name_map = await _resolve_country_names(list(individual_codes)) if individual_codes else None

    if result_layout == "merged":
        merged_indicators: list[EnrichedIndicator] = [
            ind for g in groups for ind in g.indicators
        ]
        # Sort: covers_country=True first (per-indicator, already correct), then recency
        if response_country:
            merged_indicators.sort(
                key=lambda x: (
                    not any((x.covers_country or {}).values()),
                    -(
                        int(x.latest_data or 0)
                        if str(x.latest_data or "").isdigit()
                        else 0
                    ),
                )
            )
        return MultiQuerySearchResponse(
            indicators=merged_indicators,
            result_layout="merged",
            queries=clean_queries,
            required_country=response_country,
            country_names=name_map,
            total_candidates=total_candidates,
            deduplicated_count=deduplicated_count if dedupe else None,
        )
    else:  # by_query
        return MultiQuerySearchResponse(
            results=groups,
            result_layout="by_query",
            queries=clean_queries,
            required_country=response_country,
            country_names=name_map,
            total_candidates=total_candidates,
            deduplicated_count=deduplicated_count if dedupe else None,
        )


async def search_datasets(
    query: str,
    limit: int = 10,
    offset: int = 0,
) -> DatasetSearchResponse:
    """Search for Data360 datasets matching the query.

    Use this first when the user asks for dataset details (e.g. Findex database, WDI database).
    Sanitizes special characters from the query string to prevent Search V3 API failures.
    """
    try:
        request = DatasetSearchRequest(query=query, limit=limit, offset=offset)
    except PydanticValidationError as e:
        return DatasetSearchResponse(error=str(e))

    url = (
        data360_config.search_url
        or f"{data360_config.api_url}/portal/v1/public_data360_search"
    )
    payload = {
        "site": "data360",
        "query_string": request.query,
        "types": ["dataset"],
        "data_classification": ["public"],
        "skip": request.offset,
        "items_per_page": request.limit,
    }

    try:
        client = get_shared_httpx_client()
        response = await client.post(
            url, json=payload, extensions={RETRY_SAFE_EXTENSION: True}
        )
        response.raise_for_status()

        response_data = response.json()
        results = response_data.get("results", [])

        items = []
        for value in results:
            items.append(
                DatasetDescription(
                    idno=value.get("idno", ""),
                    name=value.get("name", ""),
                    description=value.get("description"),
                    data_classification=value.get("data_classification"),
                    data_last_updated=value.get("data_last_updated"),
                    economies_count=value.get("economies_count"),
                    time_period=value.get("time_period"),
                )
            )

        total_count = (
            response_data.get("count")
            or response_data.get("@odata.count")
            or len(items)
        )

        has_more = False
        next_offset = None
        if total_count is not None and total_count > request.offset + request.limit:
            has_more = True
            next_offset = request.offset + request.limit

        return DatasetSearchResponse(
            items=items,
            count=len(items),
            total_count=total_count,
            offset=request.offset,
            has_more=has_more,
            next_offset=next_offset,
        )

    except Exception as e:
        _logger.exception("Error searching datasets")
        mcp_err = classify_error(e, context="dataset")
        return DatasetSearchResponse(error=mcp_err.detail)


# ruff: noqa: PLR0913, PLR0912, PLR0915
async def get_metadata(
    database_id: str,
    indicator_id: str,
    select_fields: list[str] | None = None,
    fetch_disaggregation: bool = True,
    required_country: str | None = None,
) -> MetadataResponse:  # noqa: PLR0912
    """Get metadata and disaggregation options for a Data360 indicator.

    Call after data360_search_indicators only when you need deep metadata NOT included in the enriched search results (e.g. methodology, source notes). If the user asks a basic metadata question (like definition or periodicity), simply answer using the fields provided by data360_search_indicators.

    For valid filter values (years, country codes, SEX/AGE/URBANISATION), prefer data360_get_disaggregation. data360_get_data and data360_get_viz_spec call get_metadata internally.

    Args:
        database_id: Database identifier (e.g., IPC_IPC, WB_GS).
        indicator_id: Indicator ID (e.g., IPC_IPC_PHASE, WB_GS_NY_GDP_PCAP_KD).
        select_fields: Optional list of metadata fields to return. If None, returns all fields.
            Available fields: methodology, statistical_concept, definition_long, limitation,
            relevance, aggregation_method, periodicity, time_periods, ref_country, sources_note.
        fetch_disaggregation: If True (default), also fetch disaggregation dimensions (field_name, field_value).
        required_country: Optional country name or 3-letter code (e.g. "Kenya", "KEN").
            Use semicolon-separated for multiple (e.g. "China; USA"). When provided,
            REF_AREA in disaggregation shows which queried countries have data.

    Returns:
        MetadataResponse:
            indicator_metadata: Dict of requested metadata fields for the indicator, or None if not found.
            disaggregation_options: List of dicts with field_name and field_value (list of valid codes).
                Dimensions with no disaggregation (INDICATOR, FREQ, single-_T SEX/AGE/URBANISATION)
                are omitted. REF_AREA is summarized as {count, sample} or {count, queried}
                instead of the full field_value list.
            error: Error message string if any request failed; otherwise None.
    """

    # Cache lookup — key includes select_fields (as frozenset) and required_country
    # because those affect what is returned.
    _cache_key = (
        database_id,
        indicator_id,
        frozenset(select_fields) if select_fields else None,
        fetch_disaggregation,
        required_country,
    )
    _cached = _metadata_cache.get(_cache_key)
    if _cached is not None:
        return _cached

    # A recent failed attempt is reused verbatim until the cooldown expires, so a
    # slow downstream dependency is contacted once per window instead of per call.
    _degraded = _metadata_cache.get_degraded(_cache_key)
    if _degraded is not None:
        _logger.info(
            "Reusing degraded metadata for %s/%s (cooldown active)",
            repr(database_id),
            repr(indicator_id),
        )
        return _degraded

    # Resolve country codes if provided
    queried_countries = await _resolve_queried_countries(required_country)

    # Validate inputs
    try:
        MetadataRequest(database_id=database_id, indicator_id=indicator_id)
    except PydanticValidationError as e:
        mcp_err = Data360ValidationError(
            context="metadata",
            detail=f"Invalid arguments: {e}",
            original_error=e,
        )
        return MetadataResponse(error=mcp_err.detail)

    # Determine URLs
    metadata_url = data360_config.metadata_url or f"{data360_config.api_url}/metadata"
    disaggregation_url = (
        data360_config.disaggregation_url or f"{data360_config.api_url}/disaggregation"
    )

    indicator_metadata: dict[str, Any] | None = None
    disaggregations: list[dict[str, Any]] = []
    errors: list[str] = []
    metadata_transient = False
    headers = {"accept": "*/*", "Content-Type": "application/json"}

    # Build query with optional select clause
    query = f"series_description/idno eq '{indicator_id}'"
    if select_fields:
        select_clause = ", ".join(f"series_description/{f}" for f in select_fields)
        metadata_payload = {"query": query, "select": select_clause}
    else:
        metadata_payload = {"query": query}

    # 1. Fetch Metadata
    try:
        client = get_shared_httpx_client()
        metadata_res = await client.post(
            metadata_url,
            json=metadata_payload,
            headers=headers,
            extensions={RETRY_SAFE_EXTENSION: True},
        )
        metadata_res.raise_for_status()

        try:
            metadata_json = metadata_res.json()
            if metadata_json and metadata_json.get("value"):
                indicator_metadata = metadata_json["value"][0].get(
                    "series_description", {}
                )
                # Inject database_name so clients always have the correct, grounded
                # label for the database_id — prevents LLMs from guessing
                # (e.g. WB_GS is "Gender Statistics", not "Global Statistics").
                if indicator_metadata:
                    db_id = indicator_metadata.get("database_id", database_id)
                    db_mapping = await get_database_mapping()
                    indicator_metadata["database_name"] = db_mapping.get(db_id)
                # Force filtering if select_fields provided (API might return more).
                # Always retain database_name regardless of select_fields.
                if select_fields and indicator_metadata:
                    indicator_metadata = {
                        k: v
                        for k, v in indicator_metadata.items()
                        if k in select_fields or k == "database_name"
                    }
            else:
                mcp_err = NotFoundError(
                    context="metadata",
                    detail=f"No metadata found for indicator ID '{indicator_id}'",
                )
                errors.append(mcp_err.detail)
        except ValueError as e:
            mcp_err = ParseError(context="metadata", original_error=e)
            errors.append(mcp_err.detail)

    except Exception as e:
        mcp_err = classify_error(e, context="metadata")
        errors.append(mcp_err.detail)
        metadata_transient = is_transient_error(mcp_err)

    # 2. Fetch Dimensions
    if fetch_disaggregation:
        try:
            dimensions_json = await _fetch_dimensions_with_cache(
                database_id, indicator_id
            )
            raw_disaggregations = _parse_dimensions_response(dimensions_json)
            disaggregations = _strip_disaggregation(
                _get_valid_disaggregations(raw_disaggregations),
                queried_countries,
            )
        except Exception as e:
            mcp_err = classify_error(e, context="disaggregation")
            errors.append(mcp_err.detail)

    # 3. Combine and Return
    error_message = "; ".join(errors) if errors else None

    result = MetadataResponse(
        indicator_metadata=indicator_metadata,
        disaggregation_options=disaggregations,
        error=error_message,
    )
    if error_message is None:
        _metadata_cache.set(_cache_key, result)
        return result

    # The live metadata fetch failed. When the failure is transient, degrade to the
    # last successful snapshot rather than an empty result: `error` keeps the reason,
    # `stale` marks the values as possibly out of date, and fresher disaggregation
    # options from this attempt win over the snapshot's.
    if metadata_transient and indicator_metadata is None:
        last_good = _metadata_cache.get_last_good(_cache_key)
        if last_good is not None:
            result = last_good.model_copy(
                update={
                    "stale": True,
                    "error": error_message,
                    "disaggregation_options": disaggregations
                    or last_good.disaggregation_options,
                }
            )

    # Hold the degraded outcome for the cooldown window so a burst of callers does
    # not repeat the slow upstream calls; a later success replaces it via set().
    _metadata_cache.set_degraded(_cache_key, result)
    _logger.warning(
        "Metadata degraded for %s/%s (cooldown %.0fs): %s",
        repr(database_id),
        repr(indicator_id),
        data360_config.degradation_cooldown_seconds,
        repr(error_message),
    )
    return result


async def get_disaggregation(
    database_id: str,
    indicator_id: str,
    required_country: str | None = None,
) -> dict[str, Any]:
    """Get disaggregation options for a Data360 indicator (valid filter values).

    Call before data360_get_data or data360_get_viz_spec to see which filter values are available.
    Typically call after data360_search_indicators when you need available years or breakdowns.
    Use the returned values in disaggregation_filters; do not use FREQ for filtering (it breaks queries).

    After calling this tool, use the returned field_value codes directly as disaggregation_filters
    in data360_get_data or data360_get_viz_spec. For example, if SEX returns ["M", "F", "_T"],
    pass {"SEX": "F"} to filter to female-only data.

    Args:
        database_id: Database identifier (e.g., WB_GS, WB_SSGD).
        indicator_id: Indicator ID (e.g., WB_GS_NY_GDP_PCAP_KD).
        required_country: Optional country name or 3-letter code (e.g. "Kenya", "KEN").
            Use semicolon-separated names or codes to check multiple countries in one call (e.g. "China; USA").
            When provided, REF_AREA shows which queried countries have data for this indicator.

    Returns:
        On success: dict with key "dimensions", a list of dicts. Trivial dimensions
            (INDICATOR, FREQ, single-_T SEX/AGE/URBANISATION) are omitted. Each dict has:
            field_name: Dimension name (e.g. TIME_PERIOD, REF_AREA, SEX).
            field_value: List of valid codes (for TIME_PERIOD sorted chronologically; for SEX/AGE etc.).
            REF_AREA is special: returns {count, sample} (5 sorted codes) or, when
            required_country is given, {count, queried: {code: bool}}.
        On failure: dict with key "error" and an error message string.
        TIME_PERIOD gives actual available years (may have gaps).
    """
    # Cache lookup — required_country affects how REF_AREA is reported.
    _disagg_cache_key = (database_id, indicator_id, required_country)
    with _disaggregation_cache_lock:
        _cached_disagg = _disaggregation_cache.get(_disagg_cache_key)
    if _cached_disagg is not None:
        return _cached_disagg

    # Resolve country codes if provided
    queried_countries = await _resolve_queried_countries(required_country)

    try:
        dimensions_json = await _fetch_dimensions_with_cache(database_id, indicator_id)
        raw_data = _parse_dimensions_response(dimensions_json)
        # Filter out _Z values and format response
        valid_dimensions = _get_valid_disaggregations(raw_data)
        result_disagg = {
            "dimensions": _strip_disaggregation(valid_dimensions, queried_countries)
        }
        with _disaggregation_cache_lock:
            _disaggregation_cache[_disagg_cache_key] = result_disagg
        return result_disagg

    except Exception as e:
        mcp_err = classify_error(e, context="disaggregation")
        return {"error": mcp_err.detail}


async def get_comp_breakdown_dim_names(
    database_id: str,
    indicator_id: str,
) -> dict[str, str]:
    """Return human-readable dimension names for comp_breakdown_1/2/3.

    Re-uses the cached disaggregation response so no extra HTTP call is made.
    Maps the column names used in the viz DataFrame (lowercase snake_case) to
    the best available human label for use as legend/tooltip titles.

    Returns a dict like::

        {"comp_breakdown_1": "Analysis Period", "comp_breakdown_2": "Severity Phase"}

    Only dimensions that are actually present in the disaggregation response
    (i.e., have non-trivial values) are included.  Falls back to "Dimension N"
    when the API echoes the field name back (e.g. WGI returns "COMP_BREAKDOWN_1"
    as its own label_name).
    """
    import re as _re

    # Matches generic placeholder labels that APIs sometimes echo back instead of a
    # real human-readable name: "Custom Dimension 2", "Dimension 3", "Breakdown 1", etc.
    # These are no more informative than the fallback, so they are rejected.
    _GENERIC_LABEL_RE = _re.compile(
        r"^(custom\s+)?(dimension|dim|breakdown|comp_breakdown)\s*\d*$",
        _re.IGNORECASE,
    )

    _FIELD_TO_COL = {
        "COMP_BREAKDOWN_1": "comp_breakdown_1",
        "COMP_BREAKDOWN_2": "comp_breakdown_2",
        "COMP_BREAKDOWN_3": "comp_breakdown_3",
    }
    _FALLBACK = {
        "comp_breakdown_1": "Dimension 1",
        "comp_breakdown_2": "Dimension 2",
        "comp_breakdown_3": "Dimension 3",
    }

    result = await get_disaggregation(database_id, indicator_id)
    dimensions = result.get("dimensions", [])

    dim_names: dict[str, str] = {}
    for dim in dimensions:
        field_name = dim.get("field_name", "")
        if field_name not in _FIELD_TO_COL:
            continue
        col = _FIELD_TO_COL[field_name]
        label_name = (dim.get("label_name") or "").strip()
        # Reject: (a) echoed field names e.g. "COMP_BREAKDOWN_1",
        #         (b) generic placeholders e.g. "Custom Dimension 2", "Dimension 3".
        # Both are no more informative than the _FALLBACK label.
        is_generic = (
            not label_name
            or label_name.upper() == field_name.upper()
            or bool(_GENERIC_LABEL_RE.match(label_name))
        )
        dim_names[col] = _FALLBACK[col] if is_generic else label_name

    return dim_names


_DEFAULT_TIME_WINDOW_YEARS = 5


def _resolve_time_range(
    start_year: int | None,
    end_year: int | None,
) -> tuple[int, int]:
    """Resolve inclusive [start_year, end_year] for Data API timePeriodFrom/To.

    - Neither bound: last ``_DEFAULT_TIME_WINDOW_YEARS`` calendar years through today.
    - Only ``end_year``: same-width window ending at ``end_year``.
    - Only ``start_year``: from ``start_year`` through the current calendar year.
    - Both: use as given (swapped if reversed).
    """
    from datetime import datetime  # noqa: PLC0415

    current_year = datetime.now().year
    span = _DEFAULT_TIME_WINDOW_YEARS - 1

    if start_year is not None:
        # TODO: Handle invalid start year, e.g., "2020-01-01" or float
        start_year = int(start_year)
    if end_year is not None:
        # TODO: Handle invalid end year, e.g., "2020-01-01"
        end_year = int(end_year)

    if start_year is None and end_year is None:
        return current_year - span, current_year

    if start_year is None:
        return end_year - span, end_year

    if end_year is None:
        return start_year, max(start_year, current_year)

    if start_year > end_year:
        return end_year, start_year

    return start_year, end_year


async def get_data(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    limit: int = 50,
    offset: int = 0,
    ref_area_filter: Literal["none", "member_economies_only"] = "none",
    auto_resolve_time_range: bool = True,
) -> IndicatorDataResponse:
    """Fetch indicator data from the Data360 API with pagination.

    Call after you have database_id and indicator_id (from data360_search_indicators). Use
    data360_get_disaggregation to get valid filter values; passing invalid values can yield empty results.
    For charts, prefer data360_get_viz_spec, which fetches data internally.

    Args:
        database_id: Database identifier (e.g., "IPC_IPC", "WB_GS").
        indicator_id: Indicator ID (e.g., "IPC_IPC_PHASE", "WB_GS_NY_GDP_PCAP_KD").
        country_code: Optional 3-letter code or semicolon-separated list (e.g. "KEN" or "KEN;MAR").
            Applied as REF_AREA filter. Takes precedence over REF_AREA in disaggregation_filters.
        disaggregation_filters: Optional dict of dimension filters. Keys: REF_AREA, SEX, AGE,
            URBANISATION, UNIT_MEASURE, etc. Values are str or None (not lists). REF_AREA supports
            comma-separated ISO codes (e.g. "KEN,TZA"); semicolons are normalized to commas.
            Use value None to request all values for a dimension (e.g. {"SEX": None}). When REF_AREA
            is omitted or None, the Data API returns all geographic series—including regional aggregates
            (e.g. EAS, EMU)—mixed with member economies.
        start_year: Optional start year (inclusive). With neither bound, defaults to last 5 years.
            With only ``end_year``, defaults to a 5-year window ending at ``end_year``.
            With only ``start_year``, defaults through the current calendar year.
        end_year: Optional end year (inclusive). See ``start_year`` for partial-bound behavior.
        limit: Maximum records per page (default 50, max 100).
        offset: Number of records to skip for pagination (default 0).
        ref_area_filter: When ``member_economies_only``, drop rows whose ``REF_AREA`` is not an FMR
            leaf member economy **only if** REF_AREA is not pinned (no ``country_code`` and no explicit
            ``REF_AREA`` string in filters). Otherwise a note is added to ``failed_validation`` and
            no filtering is applied.

    Returns:
        IndicatorDataResponse:
            data: List of data point dicts (e.g. TIME_PERIOD, REF_AREA, OBS_VALUE, claim_id).
            metadata: Indicator metadata dict if available.
            count: Number of records in this response.
            total_count: Total records available, or None.
            offset, has_more, next_offset: Use next_offset for the next page when has_more is True.
                PAGINATION NOTE: This tool returns ONE page. When has_more=True, call again with
                next_offset to retrieve more rows. For queries involving large country groups
                (e.g. from data360_expand_country_group with 20+ countries), prefer calling
                data360_rank_countries or data360_summarize_data instead — those tools paginate
                internally and return complete aggregated results without requiring manual looping.
            error: Error message if the request failed; otherwise None.
                If error contains "No metadata found", the indicator_id is invalid or stale.
                Do NOT retry with the same ID and do NOT call data360_search_indicators again.
                Instead, look back at the other indicators already returned by the previous
                data360_search_indicators call in this conversation and try the next best match.
                Only call data360_search_indicators again if no prior search results exist in context.
                If error is about a disaggregation or HTTP failure but data is still None,
                the upstream API may be temporarily unavailable — retry once or report the error.
            failed_validation: Optional list of filter validation messages. Non-empty means
                some filters were invalid; data may still be returned with valid filters applied.
    """
    data_url = data360_config.data_url or f"{data360_config.api_url}/data"

    # Cap limit to prevent token overflow
    limit = min(limit, 100)

    if auto_resolve_time_range:
        resolved_start, resolved_end = _resolve_time_range(start_year, end_year)
        if (start_year, end_year) != (resolved_start, resolved_end):
            _logger.info(
                "Resolved time range %s-%s from start_year=%s end_year=%s",
                repr(resolved_start),
                repr(resolved_end),
                repr(start_year),
                repr(end_year),
            )
        start_year, end_year = resolved_start, resolved_end

    # Validate arguments using Pydantic model
    try:
        IndicatorDataRequest(
            database_id=database_id,
            indicator_id=indicator_id,
            disaggregation_filters=disaggregation_filters,
        )
    except PydanticValidationError as e:
        mcp_err = Data360ValidationError(
            context="data",
            detail=f"Invalid arguments: {e}",
            original_error=e,
        )
        return IndicatorDataResponse(error=mcp_err.detail)

    # Prepare API parameters
    params: dict[str, Any] = {
        "DATABASE_ID": database_id,
        "INDICATOR": indicator_id,
        "timePeriodFrom": start_year,
        "timePeriodTo": end_year,
        "skip": offset,
        # Request one extra to detect if there are more results
        # Note: API uses "top" not "$top" (OData style would be URL-encoded to %24top which API ignores)
        "top": limit + 1,
    }

    # Convert semicolon-separated list into comma-separated list for Data API
    if country_code:
        codes = validated_country_codes(country_code)
        if not codes:
            return IndicatorDataResponse(
                error="country_code must contain ISO-style country codes (letters or digits)."
            )
        # Takes precedence over REF_AREA in disaggregation_filters
        params["REF_AREA"] = ",".join(codes)

    # Fetch metadata and disaggregations FIRST to inform parameter building
    # This ensures we don't apply invalid defaults (like AGE=_T) which cause empty results
    metadata_res = await get_metadata(
        database_id,
        indicator_id,
        select_fields=[
            "idno",
            "name",
            "database_id",
            "periodicity",
            "measurement_unit",
            "definition_short",
        ],
        fetch_disaggregation=True,  # Crucial: fetch valid options
        required_country=country_code,  # Pass for REF_AREA validation
    )

    if metadata_res.error:
        if metadata_res.indicator_metadata is None:
            # Fatal: indicator not found or completely unavailable.
            _logger.warning(
                "Aborting get_data for %s: indicator metadata missing. Error: %s",
                repr(indicator_id),
                repr(metadata_res.error),
            )
            return IndicatorDataResponse(error=metadata_res.error)
        # Non-fatal: disaggregation lookup failed but indicator metadata is valid.
        # Proceed without validated disaggregation defaults.
        _logger.warning(
            "Non-fatal metadata error for %s (proceeding without disaggregation defaults): %s",
            repr(indicator_id),
            repr(metadata_res.error),
        )

    api_metadata = metadata_res.indicator_metadata or {}

    # Process valid disaggregations into {dim: [values]} format
    available_disaggregations = {}
    for d in metadata_res.disaggregation_options or []:
        if d.get("field_name") and d.get("field_value"):
            available_disaggregations[d["field_name"]] = d["field_value"]

    # Validate user filters BEFORE applying defaults
    # This gives hints for hallucinations like SEX=ALIEN
    valid_filters, validation_errors = _validate_user_filters(
        disaggregation_filters, available_disaggregations
    )
    if validation_errors:
        _logger.warning(f"Validation errors for {repr(indicator_id)}: {repr(validation_errors)}")

    # Use only valid filters for building params
    effective_disagg = _build_disaggregation_params(
        valid_filters, available_disaggregations=available_disaggregations
    )
    params.update(effective_disagg)
    ref_area_unpinned = not country_code and "REF_AREA" not in params
    try:
        client = get_shared_httpx_client()
        _logger.debug("Fetching data from %s with params: %s", repr(data_url), repr(params))
        data_res = await client.get(data_url, params=params)
        data_res.raise_for_status()

        try:
            data_json = data_res.json()
        except ValueError as e:
            # mcp_err = ParseError(context="data", original_error=e)
            # return IndicatorDataResponse(data=None, error=mcp_err.detail)
            raise ParseError(context="data", original_error=e)

        raw_data = data_json.get("value", [])
        total_count = data_json.get("@odata.count")  # May be None

        # Compute API-level pagination BEFORE any filtering
        # This ensures next_offset correctly tracks position in the API result set
        api_returned_count = len(raw_data)
        has_more = api_returned_count > limit

        # Trim the extra detection row (we requested limit+1 to detect has_more)
        if api_returned_count > limit:
            raw_data = raw_data[:limit]

        # Compute API-based next_offset - the true cursor position for next page
        api_next_offset = offset + len(raw_data) if has_more else None

        # Sort by TIME_PERIOD descending (most recent first)
        raw_data.sort(key=lambda x: str(x.get("TIME_PERIOD", "")), reverse=True)

        # Final limit enforcement (safety check)
        if len(raw_data) > limit:
            raw_data = raw_data[:limit]

        from .providers import get_codelist_manager  # noqa: PLC0415

        _cm = get_codelist_manager()
        await _cm._ensure_extdataportal_loaded()
        for row in raw_data:
            row["claim_id"] = _short_hash(row)
            ref_area = row.get("REF_AREA")
            if ref_area:
                label = _cm.get_label("REF_AREA", str(ref_area))
                if label and label != str(ref_area):
                    row["REF_AREA_NAME"] = label
            unit_measure = row.get("UNIT_MEASURE")
            if unit_measure:
                unit_label = _cm.get_label("UNIT_MEASURE", str(unit_measure))
                has_mapping = bool(unit_label and unit_label != str(unit_measure))

                # Check if we have a valid multiplier that warrants qualification
                unit_mult = row.get("UNIT_MULT")
                try:
                    mult = int(unit_mult) if unit_mult is not None else 0
                except (ValueError, TypeError):
                    mult = 0
                has_valid_mult = mult in (3, 6, 9, 12)

                if has_mapping or has_valid_mult:
                    base_unit = unit_label if has_mapping else row.get("UNIT_MEASURE_NAME") or str(unit_measure)
                    qualified_unit = _qualify_unit_name(base_unit, unit_mult, str(unit_measure))
                    if qualified_unit:
                        row["UNIT_MEASURE_NAME"] = qualified_unit


        # Promote COMMENT_TS to metadata (repeats identically per row)
        if raw_data and api_metadata is not None:
            comment_ts = next(
                (r.get("COMMENT_TS") for r in raw_data if r.get("COMMENT_TS")),
                None,
            )
            if comment_ts:
                api_metadata["indicator_description"] = comment_ts

        # Strip boilerplate fields for LLM token savings
        raw_data = [_strip_data_row(row) for row in raw_data]

        filter_notes: list[str] = list(validation_errors) if validation_errors else []
        if ref_area_filter == "member_economies_only":
            if ref_area_unpinned:
                from .providers import get_group_hierarchy_manager  # noqa: PLC0415

                _ghm = get_group_hierarchy_manager()
                raw_data = [
                    r for r in raw_data if _ghm.is_country(str(r.get("REF_AREA", "")))
                ]
            else:
                filter_notes.append(
                    "ref_area_filter=member_economies_only applies only when REF_AREA is "
                    "unpinned (omit country_code and do not set REF_AREA to specific codes); "
                    "filter was not applied."
                )

        return IndicatorDataResponse(
            data=raw_data,
            metadata=api_metadata,
            count=len(raw_data),
            total_count=total_count,
            offset=offset,
            has_more=has_more,
            next_offset=api_next_offset,
            error=None,
            failed_validation=filter_notes if filter_notes else None,
        )

    except Data360MCPError:
        # Re-raise our own errors to propagate isError: true to MCP client
        raise
    except Exception as e:
        # # Convert unknown exceptions to Data360MCPError and raise
        # mcp_err = classify_error(e, context="data")
        # return IndicatorDataResponse(data=None, error=mcp_err.detail)
        raise classify_error(e, context="data")


async def get_indicators(database_id: str) -> list[str]:
    """Get all indicator IDs for a specific database.

    Use when you need the full list of indicator IDs for a dataset. For discovery by topic,
    use data360_search_indicators instead. You must know the database_id (e.g. from search or docs).

    Args:
        database_id: The database identifier (e.g., "WB_WDI", "WB_GS").

    Returns:
        List of indicator ID strings for that database. Empty list on error or if none exist.
    """
    url = f"{data360_config.api_url}/indicators"
    params = {"datasetId": database_id}

    try:
        client = get_shared_httpx_client()
        response = await client.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        if isinstance(data, list):
            # Data is a list of indicator ID strings
            return data
        return []

    except Exception as e:
        _logger.error(f"Failed to fetch indicators for {repr(database_id)}: {repr(e)}")
        raise


async def discover_indicators(
    query: str,
    required_country: str | None = None,
    required_dimensions: list[str] | None = None,
    limit: int = 5,
) -> DiscoveryResult:
    """Search for indicators and validate their capabilities.

    .. deprecated::
        This function is deprecated. Use the following workflow instead:
        1. search() with select_fields for candidates
        2. get_disaggregation() to validate availability
        3. get_metadata() with select_fields for specific info

    This is the primary tool that combines:
    1. Search for top K indicators matching the query
    2. Fetch metadata for all K indicators in parallel
    3. Cross-check capabilities (countries, dimensions available)
    4. Return condensed, validated results for LLM to select from

    Args:
        query: Search query string (e.g., "unemployment rate", "poverty")
        required_country: Country name or code to validate (e.g., "Kenya" or "KEN")
        required_dimensions: List of required disaggregations (e.g., ["SEX", "AGE"])
        limit: Maximum number of indicators to search and validate (default: 5)

    Returns:
        DiscoveryResult with list of validated indicator summaries and optional error

    Example:
        discover_indicators(
            query="unemployment rate",
            required_country="Kenya",
            required_dimensions=["SEX", "AGE"]
        )
    """
    import warnings  # noqa: PLC0415

    warnings.warn(
        "discover_indicators is deprecated. Use search() + get_disaggregation() + get_metadata() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    # ... existing implementation ...


# --- Visualization Workflow Tools ---


async def get_data_api_url(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
) -> str:
    """Generate a Data360 API URL for a dataset without fetching data.

    Low-level tool: use only when you need the raw data API URL (e.g. custom clients or debugging).
    For charts, use data360_get_viz_spec instead; it builds the URL, fetches data, and generates the spec.
    Use data360_get_disaggregation to obtain valid filter values.

    Args:
        database_id: Database identifier (e.g., WB_HNP, WB_WDI).
        indicator_id: Indicator ID (e.g., WB_HNP_SP_POP_TOTL).
        country_code: Optional 3-letter code or semicolon-separated list (e.g. "KEN" or "CHN;USA").
        start_year: Optional start year (inclusive).
        end_year: Optional end year (inclusive).
        disaggregation_filters: Optional dict of dimension filters (e.g. {"SEX": "F"}).
            Values are str or None. REF_AREA: comma-separated ISO codes; semicolons normalized.
            If omitted, defaults to totals (_T) for SEX, AGE, URBANISATION where applicable.

    Returns:
        Full Data360 data API URL string (query parameters included).
        Raises ValueError if the indicator is not found in the specified database.
            If this happens, the indicator_id is invalid or stale. Do NOT retry with the same ID.
            Look back at the other indicators already returned by the previous
            data360_search_indicators call in this conversation and try the next best match.
            Only call data360_search_indicators again if no prior search results exist in context.
    """
    settings = get_data360_settings()

    # Fix double-slash: ensure base URL doesn't end with slash before appending /data
    base_url = settings.api_url.rstrip("/")
    base = f"{base_url}/data"

    # Construct query params
    params = {"DATABASE_ID": database_id, "INDICATOR": indicator_id}

    if country_code:
        # Same as get_data(): Data API expects comma-separated REF_AREA; allow semicolons in tool args.
        codes = validated_country_codes(country_code)
        if not codes:
            raise ValueError(
                "country_code must contain ISO-style country codes (letters or digits)."
            )
        params["REF_AREA"] = ",".join(codes)

    if start_year:
        params["timePeriodFrom"] = start_year

    if end_year:
        params["timePeriodTo"] = end_year
    # Fetch metadata and disaggregations FIRST to inform parameter building
    # This ensures we don't apply invalid defaults (like AGE=_T) which cause empty results
    metadata_res = await get_metadata(
        database_id,
        indicator_id,
        select_fields=[],  # We only need disaggregation options, metadata fields not needed
        fetch_disaggregation=True,
    )

    if metadata_res.error:
        if metadata_res.indicator_metadata is None:
            raise ValueError(
                f"Indicator '{indicator_id}' not found: {metadata_res.error}"
            )
        _logger.warning(
            "Non-fatal metadata error for %s in get_data_api_url (proceeding): %s",
            repr(indicator_id),
            repr(metadata_res.error),
        )

    # Process valid disaggregations into {dim: [values]} format
    available_disaggregations = {}
    for d in metadata_res.disaggregation_options or []:
        if d.get("field_name") and d.get("field_value"):
            available_disaggregations[d["field_name"]] = d["field_value"]

    # Validate user filters
    valid_filters, validation_errors = _validate_user_filters(
        disaggregation_filters, available_disaggregations
    )
    if validation_errors:
        # For URL generation, we might still want to raise if anything is invalid
        # or we could just skip invalid ones. Raising is safer for URL consistency.
        raise ValueError("\n\n".join(validation_errors))

    # See _build_disaggregation_params() docstring for behavior
    effective_filters = _build_disaggregation_params(
        valid_filters, available_disaggregations=available_disaggregations
    )

    # Add dimension filters to params
    params.update(effective_filters)

    # Default limit for viz
    limit = 1000
    if country_code:
        # Increase limit based on number of countries requested (max ~60 years per country)
        # Using 1000 as a safe multiplier to cover most time series data including higher frequency
        codes = validated_country_codes(country_code)
        n_countries = max(1, len(codes))
        limit = max(1000, n_countries * 1000)

    params["top"] = limit

    query_string = urlencode(params, safe=",")
    return f"{base}?{query_string}"


# ---------------------------------------------------------------------------
# Data Aggregation Tools (Tier 1 — full implementation)
# ---------------------------------------------------------------------------

_PAGE_SIZE = 100  # rows per page for aggregation fetches


async def _fetch_all_pages(
    database_id: str,
    indicator_id: str,
    country_code: str | None,
    disaggregation_filters: dict[str, str | None] | None,
    start_year: int | None,
    end_year: int | None,
) -> IndicatorDataResponse:
    """SERVER-SIDE pagination consumer for aggregation tools. Not an MCP tool.

    Design rationale — get_data vs _fetch_all_pages:

    data360_get_data (MCP tool, LLM-facing):
        Returns ONE page (limit rows) and exposes has_more + next_offset for the
        LLM to drive pagination. Correct for PATH A (point lookups) and any case
        where the LLM presents data directly to the user. The LLM sees and controls
        the cursor.

    _fetch_all_pages (internal, server-side):
        Loops get_data until has_more=False and merges ALL rows into one response.
        Required for aggregation tools (summarize_data, rank_countries,
        compare_countries) because statistics computed on a partial dataset are
        silently wrong — you cannot rank 48 Sub-Saharan African countries if you
        only have the first 100 rows of a 240-row response.

        This distinction becomes critical when data360_expand_country_group is
        used upstream: a group like SSF (48 countries) × 20 years = 960 rows
        requires 10 pages. Without this helper, any aggregation on large country
        groups would silently truncate at page 1.

    Rule: the LLM should NEVER call this directly. It is satisfied by the
    aggregation MCP tools (data360_summarize_data, data360_rank_countries,
    data360_compare_countries) which call it internally.

    Error handling: first-page errors propagate immediately. Mid-pagination errors
    log a warning and return the rows collected so far (partial > nothing).

    Safety: stops after _MAX_PAGES pages to guard against runaway loops if the API
    incorrectly signals has_more=True indefinitely.
    """
    _MAX_PAGES = 50  # 50 * 100 = 5 000 rows — generous ceiling for any real indicator
    all_rows: list[dict[str, Any]] = []
    metadata = None
    offset = 0
    page_count = 0

    while True:
        page_count += 1
        if page_count > _MAX_PAGES:
            _logger.warning(
                "_fetch_all_pages: hit safety page limit (%d) for %s/%s after %d rows. "
                "Returning partial results.",
                _MAX_PAGES,
                repr(database_id),
                repr(indicator_id),
                len(all_rows),
            )
            break

        page = await get_data(
            database_id=database_id,
            indicator_id=indicator_id,
            country_code=country_code,
            disaggregation_filters=disaggregation_filters,
            start_year=start_year,
            end_year=end_year,
            limit=_PAGE_SIZE,
            offset=offset,
            auto_resolve_time_range=False,
        )

        if page.error:
            # Propagate first-page errors; for later pages keep what we have
            if offset == 0:
                return page
            _logger.warning(
                "Pagination error at offset %d for %s/%s: %s",
                offset,
                repr(database_id),
                repr(indicator_id),
                repr(page.error),
            )
            break

        if page.metadata and metadata is None:
            metadata = page.metadata

        if page.data:
            all_rows.extend(page.data)

        if not page.has_more:
            break

        offset = page.next_offset or (offset + _PAGE_SIZE)

    return IndicatorDataResponse(
        data=all_rows,
        metadata=metadata,
        count=len(all_rows),
        total_count=len(all_rows),
        offset=0,
        has_more=False,
        next_offset=None,
    )


async def _resolve_country_names(codes: list[str]) -> dict[str, str]:
    """Batch-resolve country codes to display names.

    Uses the in-memory CodelistManager to lookup human-readable country names
    from the REF_AREA codelist. Silently returns empty names on any error.
    """
    from .providers import get_codelist_manager  # noqa: PLC0415

    if not codes:
        return {}
    try:
        _cm = get_codelist_manager()
        await _cm._ensure_extdataportal_loaded()
        return {code: _cm.get_label("REF_AREA", code) for code in codes}
    except Exception:
        return {}


# Disaggregation dimensions that may carry meaningful sub-categories.
# UNIT_MEASURE is handled via a separate branch in the helper below.
_DISAGG_DIMS_TO_DETECT: tuple[str, ...] = (
    "SEX",
    "AGE",
    "URBANISATION",
    "COMP_BREAKDOWN_1",
    "COMP_BREAKDOWN_2",
)


async def _auto_detect_disagg_dimensions(
    database_id: str,
    indicator_id: str,
    sample_country: str | None,
    existing_filters: dict[str, str | None],
    *,
    expand_non_trivial: bool,
) -> tuple[dict[str, str | None], list[str]]:
    """Detect disaggregation dimensions and build effective filters.

    Calls get_disaggregation once and inspects SEX, AGE, URBANISATION,
    COMP_BREAKDOWN_1/2, and UNIT_MEASURE for each indicator.

    Two modes, controlled by `expand_non_trivial`:

    expand_non_trivial=True  (summarize_data path):
        For each dim that has values beyond _T and is not already in
        existing_filters, set the filter to None (fetch all values) and
        record it in auto_expanded_dims so the caller can add it to group_by.

    expand_non_trivial=False (rank_countries / compare_countries path):
        For each such dim, pin to _T when available, otherwise to the first
        non-total value, to prevent duplicate rows per country per year which
        would corrupt ranking and comparison statistics.

    UNIT_MEASURE is always pinned to its sole value when exactly one exists
    (regardless of mode), matching the prior behaviour.

    Caller's existing_filters always take precedence — this function never
    overwrites a filter already set by the caller.

    Returns:
        (effective_filters, auto_expanded_dims)
        effective_filters: copy of existing_filters with auto-detected values.
        auto_expanded_dims: lowercase dim names that were expanded into None
            (non-empty only when expand_non_trivial=True).
    """
    effective_filters: dict[str, str | None] = dict(existing_filters)
    auto_expanded_dims: list[str] = []

    try:
        disagg_res = await get_disaggregation(
            database_id=database_id,
            indicator_id=indicator_id,
            required_country=sample_country,
        )
    except Exception:
        return effective_filters, auto_expanded_dims

    if not disagg_res or disagg_res.get("error"):
        return effective_filters, auto_expanded_dims

    for dim in disagg_res.get("dimensions", []):
        field_name: str = dim.get("field_name", "")
        values: list[str] = dim.get("field_value", [])

        if not values or field_name in ("REF_AREA", "TIME_PERIOD"):
            continue

        # Honour the caller's explicit filter for this dimension.
        if field_name in effective_filters:
            continue

        # UNIT_MEASURE: always pin to its single value when unambiguous.
        if field_name == "UNIT_MEASURE":
            if len(values) == 1:
                effective_filters["UNIT_MEASURE"] = values[0]
            continue

        if field_name not in _DISAGG_DIMS_TO_DETECT:
            continue

        # Only act on dimensions that carry values beyond the aggregate total.
        non_total_values = [v for v in values if v not in ("_T", "_Z")]
        if not non_total_values:
            continue

        if expand_non_trivial:
            # Expand: set to None so _fetch_all_pages retrieves all breakdowns.
            effective_filters[field_name] = None
            auto_expanded_dims.append(field_name.lower())
        else:
            # Pin to _T (aggregate total) when available to avoid duplicate rows.
            effective_filters[field_name] = (
                "_T" if "_T" in values else non_total_values[0]
            )

    return effective_filters, auto_expanded_dims


def _compute_trend_direction(values: list[float]) -> str:
    """Determine trend direction from a time-ordered list of values.

    Uses Huber regression (outlier-robust) to fit a linear trend, then
    classifies the result into one of four categories:
    - R² < 0.3  → "volatile"   (no clear linear trend)
    - |slope| < 1% of |mean| per step → "stable"
    - slope > 0 → "increasing"
    - slope < 0 → "decreasing"

    HuberRegressor is preferred over OLS because development indicators can
    contain anomalous years (conflict, crises, revisions) that would skew an
    OLS slope.  The Huber loss function down-weights outliers automatically.

    Expects pre-cleaned, finite float values. The caller (_build_group_summary)
    applies .dropna() upstream; np.isfinite() below drops any remaining
    non-finite values (inf, -inf) before fitting.
    """
    y = np.asarray(values, dtype=float)
    y = y[np.isfinite(y)]  # drop inf/-inf; NaN already removed by caller

    if len(y) < 2:
        return "stable"

    x = np.arange(len(y)).reshape(-1, 1)

    model = HuberRegressor()
    try:
        model.fit(x, y)
    except Exception:
        # Fallback: all identical values or degenerate input
        return "stable"

    slope = float(model.coef_[0])
    r_squared = float(model.score(x, y))

    if r_squared < 0.3:
        return "volatile"

    y_mean = float(np.mean(y))
    if y_mean != 0 and abs(slope / y_mean) < 0.01:
        return "stable"

    return "increasing" if slope > 0 else "decreasing"


def _build_group_summary(
    group_key: dict[str, str],
    rows: list[dict[str, Any]],
) -> GroupSummary:
    """Build a GroupSummary from a list of data rows sharing the same group key.

    Uses pandas for type-safe numeric coercion and descriptive statistics.
    pd.to_numeric with errors='coerce' handles mixed-type OBS_VALUE values
    from the API (strings, None, empty string) without raising exceptions.

    TIME_PERIOD deduplication: indicators with multiple disaggregation dimensions
    (e.g. IPC_IPC_PHASE with COMP_BREAKDOWN_2) can return several rows per period
    when the caller has not fully specified all disaggregation filters.  The
    .drop_duplicates(keep="last") call reduces these to one row per period so that
    trend statistics are computed on a single time series rather than a mix of
    disaggregated values.  A warning is emitted when rows are actually dropped so
    the behaviour is observable; callers should pass disaggregation_filters to
    narrow to a single series and suppress the warning.
    """
    df = pd.DataFrame(rows)

    # Preserve claim_ids before any filtering
    claim_ids: list[str] = (
        df["claim_id"].dropna().tolist() if "claim_id" in df.columns else []
    )

    # Coerce OBS_VALUE — handles strings, None, empty strings from API
    df["OBS_VALUE"] = pd.to_numeric(
        df["OBS_VALUE"] if "OBS_VALUE" in df.columns else pd.Series(dtype=float),
        errors="coerce",
    )
    if "TIME_PERIOD" not in df.columns:
        df["TIME_PERIOD"] = ""
    df["TIME_PERIOD"] = df["TIME_PERIOD"].astype(str)

    # Sort ascending, drop missing values, deduplicate periods (keep last).
    # See docstring for rationale on TIME_PERIOD deduplication.
    df_sorted = df.sort_values("TIME_PERIOD").dropna(subset=["OBS_VALUE"])
    n_before_dedup = len(df_sorted)
    df = df_sorted.drop_duplicates(subset=["TIME_PERIOD"], keep="last").reset_index(
        drop=True
    )
    n_dropped = n_before_dedup - len(df)
    if n_dropped > 0:
        _logger.warning(
            "_build_group_summary: dropped %d duplicate TIME_PERIOD row(s) for group %s. "
            "Pass disaggregation_filters to narrow to a single series and avoid this.",
            n_dropped,
            repr(group_key),
        )

    if df.empty:
        return GroupSummary(group_key=group_key, count=0, claim_ids=claim_ids)

    values_s = df["OBS_VALUE"]
    years = df["TIME_PERIOD"].tolist()

    earliest_val = float(values_s.iloc[0])
    latest_val = float(values_s.iloc[-1])
    earliest_yr: str = years[0]
    latest_yr: str = years[-1]

    total_change = latest_val - earliest_val
    pct_change = (
        round((total_change / abs(earliest_val)) * 100, 2)
        if earliest_val != 0
        else None
    )
    time_range = (
        f"{earliest_yr}-{latest_yr}" if earliest_yr != latest_yr else earliest_yr
    )

    # Use .describe() for a single-pass computation of standard statistics.
    # More efficient than 4 separate aggregation calls on the same series.
    _desc = values_s.describe()

    return GroupSummary(
        group_key=group_key,
        count=len(df),
        latest_value=round(latest_val, 4),
        latest_year=latest_yr,
        earliest_value=round(earliest_val, 4),
        earliest_year=earliest_yr,
        min=round(float(_desc["min"]), 4),
        max=round(float(_desc["max"]), 4),
        mean=round(float(_desc["mean"]), 4),
        median=round(float(_desc["50%"]), 4),
        total_change=round(total_change, 4),
        pct_change=pct_change,
        trend_direction=_compute_trend_direction(values_s.tolist()),
        time_range=time_range,
        claim_ids=claim_ids,
    )


# Mapping from lowercase group_by column names to raw API field names.
_GROUPBY_FIELD_MAP: dict[str, str] = {
    "ref_area": "REF_AREA",
    "region": "REGION",
    "income_group": "INCOME_GROUP",
    "time_period": "TIME_PERIOD",
    "sex": "SEX",
    "age": "AGE",
    "urbanisation": "URBANISATION",
    "residence": "URBANISATION",
    "unit_measure": "UNIT_MEASURE",
    "comp_breakdown_1": "COMP_BREAKDOWN_1",
    "comp_breakdown_2": "COMP_BREAKDOWN_2",
}


async def summarize_data(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    group_by: list[str] | None = None,
) -> DataSummaryResponse:
    """Compute summary statistics for indicator data, grouped by one or more dimensions.

    Call instead of data360_get_data when the user asks about trends, changes over time,
    or general patterns — not specific year values. Particularly useful for PATH C (trend)
    questions like "How has X changed?" or "What is the trend of Y?". Use the default
    group_by=["ref_area"] for all single- or multi-country trend questions to group at the
    country level; this typically yields meaningful multi-point statistics per country, but
    additional auto-detected disaggregation dimensions may create multiple groups per
    country. The LLM should pick group_by columns based on the question's analytical intent.

    Do NOT call this when the user wants a specific data point for a specific year — use
    data360_get_data for that (PATH A). Do NOT call this for visualization — use
    data360_get_viz_spec which fetches data internally.

    Auto-detection of disaggregation dimensions: before fetching data, this tool calls
    get_disaggregation to discover non-trivial dimensions (SEX, AGE, URBANISATION,
    COMP_BREAKDOWN_1/2) that the indicator supports beyond the aggregate total (_T).
    Any such dimension not already specified in disaggregation_filters is automatically
    added to group_by and fetched with all its values. The auto-expanded dimension names
    are reported in ambiguous_dimensions. Caller-provided disaggregation_filters always
    take precedence and suppress auto-expansion for that dimension.

    Args:
        database_id: Database identifier (e.g., "WB_WDI", "WB_GS").
        indicator_id: Indicator ID (e.g., "WB_WDI_NY_GDP_PCAP_KD").
        country_code: Optional 3-letter code or semicolon-separated list (e.g. "KEN;NGA").
        disaggregation_filters: Optional dimension filters (e.g. {"UNIT_MEASURE": "KD"}).
            Filters specified here are honoured as-is and suppress auto-detection for
            that dimension. Pass {"SEX": "_T"} to force totals only, or {"SEX": None}
            to explicitly request all sex breakdowns.
        start_year: Optional start year. Defaults to last 5 years.
        end_year: Optional end year. Defaults to current year.
        group_by: Dimensions to group by. Default ["ref_area"]. Valid columns: ref_area,
            time_period, sex, age, urbanisation, unit_measure, comp_breakdown_1,
            comp_breakdown_2. Non-trivial dimensions found via auto-detection are appended
            automatically when not already present.

            DECISION GUIDE — always match group_by to the analytical intent:
            - "How has Kenya's GDP changed?" → ["ref_area"] (DEFAULT). Produces ONE
              group (KEN) spanning all years — gives real trend direction, min, max,
              mean, pct_change over the full period.
            - "Compare GDP trends: Kenya vs. Nigeria" → ["ref_area"] (DEFAULT).
              Produces TWO groups, one trend line each.
            - "GDP by sex for Kenya" → ["ref_area", "sex"]. One group per (country, sex).

            WARNING — NEVER use ["time_period"] as the sole group_by for trend or
            single-country questions. Grouping by time_period alone typically creates
            ONE GROUP PER YEAR and, when a single series is returned, often leaves each
            group with exactly ONE observation. In that common n=1 case, every group
            shows min=max=mean=median=that single value, change=0 %, trend=stable —
            mathematically degenerate and usually useless for trend analysis.
            ["time_period"] is only valid for cross-country year-over-year aggregates
            (e.g. "global average per year") when country_code is NOT specified.

    Returns:
        DataSummaryResponse:
            groups: List of per-group summaries with count, latest/earliest values,
                min/max/mean/median, total_change, pct_change, trend_direction,
                and source claim_ids.
            metadata: Indicator metadata (name, definition, database_name).
            unit_measure: The unit for interpreting values.
            ambiguous_dimensions: Lowercase names of dimensions that were auto-detected
                as non-trivial and appended to group_by (e.g. ["sex", "age"]). None when
                no auto-expansion occurred.
            error: Error message if request failed; otherwise None.
                Falls back to data360_get_data if this tool encounters an error.
    """
    if group_by is None:
        group_by = ["ref_area"]

    # Validate group_by columns before the disagg fetch to fail fast.
    invalid_cols = [c for c in group_by if c.lower() not in _GROUPBY_FIELD_MAP]
    if invalid_cols:
        return DataSummaryResponse(
            error=f"Invalid group_by columns: {invalid_cols}. "
            f"Valid options: {list(_GROUPBY_FIELD_MAP.keys())}"
        )

    # Pre-fetch: detect non-trivial disaggregation dimensions and auto-expand
    # them into group_by + effective_filters before the data fetch.  This ensures
    # _fetch_all_pages retrieves all breakdown values in a single call, avoiding
    # the silent data loss that occurred when the API defaulted to _T only.
    sample_country: str | None = None
    if country_code:
        codes = validated_country_codes(country_code)
        sample_country = codes[0] if codes else None

    effective_filters, auto_expanded_dims = await _auto_detect_disagg_dimensions(
        database_id=database_id,
        indicator_id=indicator_id,
        sample_country=sample_country,
        existing_filters=dict(disaggregation_filters or {}),
        expand_non_trivial=True,
    )

    # Append auto-detected dims to group_by (preserves caller's explicit list).
    group_by = list(group_by)  # mutable copy
    for dim_lower in auto_expanded_dims:
        if dim_lower not in group_by:
            group_by.append(dim_lower)

    # Validate any auto-appended dims (defensive; _DISAGG_DIMS_TO_DETECT are
    # all in _GROUPBY_FIELD_MAP, but check to avoid silent runtime errors).
    invalid_auto = [c for c in group_by if c.lower() not in _GROUPBY_FIELD_MAP]
    if invalid_auto:
        return DataSummaryResponse(
            error=f"Auto-detected invalid group_by columns: {invalid_auto}. "
            f"Valid options: {list(_GROUPBY_FIELD_MAP.keys())}"
        )

    # Fetch all pages — paginate automatically until has_more=False
    try:
        data_response = await _fetch_all_pages(
            database_id=database_id,
            indicator_id=indicator_id,
            country_code=country_code,
            disaggregation_filters=effective_filters or None,
            start_year=start_year,
            end_year=end_year,
        )
    except Exception as e:
        mcp_err = classify_error(e, context="summarize")
        return DataSummaryResponse(error=mcp_err.detail)

    if data_response.error:
        return DataSummaryResponse(error=data_response.error)

    if not data_response.data:
        return DataSummaryResponse(
            error="No data returned for the specified parameters. "
            "Fall back to data360_get_data with broader filters.",
            metadata=data_response.metadata,
        )

    # Extract unit_measure from first row
    unit_measure = None
    if data_response.data:
        raw_unit = data_response.data[0].get("UNIT_MEASURE")
        if raw_unit:
            from .providers import get_codelist_manager  # noqa: PLC0415
            _cm = get_codelist_manager()
            await _cm._ensure_extdataportal_loaded()
            unit_measure = _cm.get_label("UNIT_MEASURE", str(raw_unit))

    # Group rows by the specified dimensions
    raw_field_names = [_GROUPBY_FIELD_MAP[c.lower()] for c in group_by]
    groups_dict: dict[tuple[str, ...], list[dict[str, Any]]] = {}

    # Region and Income mapping helpers — build an inverted lookup dict once
    # so that each per-row call is O(1) rather than O(n_groups).
    from .providers import get_group_hierarchy_manager  # noqa: PLC0415
    ghm = get_group_hierarchy_manager()
    # _country_to_group: (country_code_upper, group_type) -> group_id
    _country_to_group: dict[tuple[str, str], str] = {}
    for _gid, _ginfo in ghm._groups.items():
        _gtype = _ginfo.get("type", "")
        # Member countries → their containing group
        for _c in _ginfo.get("countries", []):
            _country_to_group[(_c.upper(), _gtype)] = _gid
        # The group code itself maps to itself (e.g. SSF → SSF for REGION)
        _country_to_group[(_gid.upper(), _gtype)] = _gid

    def get_country_group(country: str, group_type: str) -> str:
        return _country_to_group.get((country.upper(), group_type), "_MISSING")

    for row in data_response.data:
        key_parts = []
        for f in raw_field_names:
            val = row.get(f)
            if val is None:
                if f == "REGION":
                    ref_area = row.get("REF_AREA")
                    val = get_country_group(ref_area, "REGION") if ref_area else "_MISSING"
                elif f == "INCOME_GROUP":
                    ref_area = row.get("REF_AREA")
                    val = get_country_group(ref_area, "INCOME") if ref_area else "_MISSING"
                else:
                    # The upstream API omits disaggregation keys when their value is
                    # the default aggregate total (_T).
                    val = (
                        "_T"
                        if f
                        in (
                            "SEX",
                            "AGE",
                            "URBANISATION",
                            "COMP_BREAKDOWN_1",
                            "COMP_BREAKDOWN_2",
                        )
                        else "_MISSING"
                    )
            key_parts.append(str(val))
        key = tuple(key_parts)
        groups_dict.setdefault(key, []).append(row)

    # Build summaries
    group_summaries = []
    for key_tuple, rows in groups_dict.items():
        group_key = {col: val for col, val in zip(group_by, key_tuple)}
        group_summaries.append(_build_group_summary(group_key, rows))

    # Sort by latest_year descending, then by group_key
    group_summaries.sort(
        key=lambda g: (g.latest_year or "", str(g.group_key)),
        reverse=True,
    )

    # Resolve human-readable country names for ref_area groups.
    # Reuses the same _resolve_country_names helper used by rank_countries and
    # compare_countries. Injects ref_area_name directly into the group_key dict
    # so it flows through GroupSummary.to_compact() without a schema change.
    # Silently no-ops on error — country name is optional enrichment.
    _area_codes = sorted(
        {g.group_key["ref_area"] for g in group_summaries if "ref_area" in g.group_key}
    )
    if _area_codes:
        _name_map = await _resolve_country_names(_area_codes)
        for g in group_summaries:
            if "ref_area" in g.group_key:
                name = _name_map.get(g.group_key["ref_area"])
                if name:
                    g.group_key["ref_area_name"] = name

    return DataSummaryResponse(
        groups=group_summaries,
        metadata=data_response.metadata,
        unit_measure=unit_measure,
        ambiguous_dimensions=auto_expanded_dims if auto_expanded_dims else None,
    )


async def rank_countries(
    database_id: str,
    indicator_id: str,
    country_group: str | None = None,
    country_codes: str | None = None,
    year: int | None = None,
    order: str = "desc",
    top_n: int = 10,
    disaggregation_filters: dict[str, str | None] | None = None,
    rank_universe: Literal["explicit", "all_member_economies"] = "explicit",
) -> RankingResponse:
    """Rank countries by indicator value for a specific year.

    Call for PATH B questions involving large country sets: "Top 10 countries by GDP",
    "Which South Asian country has the lowest poverty rate?", "Rank Sub-Saharan African
    countries by life expectancy". Handles ties, missing data, and group expansion internally
    (calls data360_expand_country_group when country_group is provided).

    **Global rankings ("top 10 in the world"):** set ``rank_universe='all_member_economies'``
    and omit both ``country_group`` and ``country_codes``. The tool fetches unpinned
    geographic data from the Data API (full REF_AREA coverage) and keeps only FMR leaf
    member economies—regional aggregates such as ``EAS`` or ``EMU`` are excluded.

    When year is None, the tool selects the ranking year automatically. It considers both
    the latest available year and the year with broadest country coverage, and reports which
    strategy was used in year_selection_note. Both approaches are valid — the LLM should
    evaluate which is more appropriate for the user's question.

    Do NOT call for 2-3 country comparisons — use data360_compare_countries instead.
    Do NOT call for time series or trend analysis — use data360_summarize_data instead.

    Args:
        database_id: Database identifier.
        indicator_id: Indicator ID.
        country_group: Group code to rank within (e.g. "SAS", "LIC", "SSF").
            Expanded internally via data360_expand_country_group.
        country_codes: Semicolon-separated codes. Overrides country_group if both given.
        year: Ranking year. None = auto-select (see year_selection_note in response).
        order: "desc" (highest first) or "asc" (lowest first).
        top_n: Number of top results to return (default 10).
        disaggregation_filters: Optional dimension filters. Do not pin ``REF_AREA`` when
            using ``rank_universe='all_member_economies'`` (it is ignored for the fetch).
        rank_universe: ``explicit`` (default) requires ``country_codes`` or ``country_group``.
            Use ``all_member_economies`` for worldwide leaderboards when both are omitted.

    Returns:
        RankingResponse:
            year: The ranking year used.
            year_selection_note: How the year was chosen (coverage vs recency).
            order: "desc" or "asc".
            total_with_data: Countries that had data.
            total_requested: Countries attempted.
            universe / universe_size: Scope metadata (see field descriptions).
            rankings: Ranked list with rank, ref_area, country_name, obs_value,
                percentile, claim_id. Ties share the same rank.
            excluded: Countries with no data and reason.
            metadata: Indicator metadata.
            unit_measure: Unit string.
            error: Error message if request failed; otherwise None.
                Falls back to data360_get_data if this tool encounters an error.
    """
    from .providers import (  # noqa: PLC0415
        expand_country_group,
        get_group_hierarchy_manager,
    )

    ghm = get_group_hierarchy_manager()
    universe: str | None = None
    universe_size: int | None = None
    global_member_ranking = False

    # Resolve country codes (explicit scope wins over rank_universe)
    resolved_codes: list[str] = []
    if country_codes:
        resolved_codes = validated_country_codes(country_codes)
        if not resolved_codes:
            # Falling through with an empty list would join to country_code="" which
            # reads as "no scope" and ranks the global dataset instead.
            return RankingResponse(
                error=(
                    "country_codes must contain ISO-style country codes "
                    "(letters or digits), separated by ',' or ';'."
                )
            )
        universe = "explicit"
    elif country_group:
        # Gate on is_group() before attempting expansion — this prevents silent
        # failures where an unrecognised code reaches the API and returns 0 rows.
        if not ghm.is_group(country_group):
            return RankingResponse(
                error=(
                    f"'{country_group}' is not a recognised country group. "
                    "Use data360_find_codelist_value('REF_AREA', '<name>') to find "
                    "the correct group code (e.g. 'SSF', 'SAS', 'LIC')."
                )
            )
        try:
            expand_result = await expand_country_group(country_group)
            if isinstance(expand_result, dict) and expand_result.get("country_codes"):
                resolved_codes = [
                    c.strip()
                    for c in expand_result["country_codes"].split(",")
                    if c.strip()
                ]
            elif isinstance(expand_result, dict) and expand_result.get("error"):
                return RankingResponse(error=expand_result["error"])
            else:
                return RankingResponse(
                    error=f"Could not expand country group '{country_group}'."
                )
        except Exception as e:
            return RankingResponse(error=f"Failed to expand country group: {e}")
        universe = "explicit"
    elif rank_universe == "all_member_economies":
        resolved_codes = ghm.list_rankable_country_codes()
        global_member_ranking = True
        universe = "all_member_economies"
    else:
        return RankingResponse(
            error=(
                "No countries specified. Provide country_codes or country_group, "
                "or set rank_universe='all_member_economies' to rank all World Bank "
                "member economies (regional aggregates excluded)."
            )
        )

    total_requested = len(resolved_codes)
    universe_size = total_requested

    filters_for_auto = dict(disaggregation_filters or {})
    if global_member_ranking:
        filters_for_auto.pop("REF_AREA", None)

    # Auto-detect UNIT_MEASURE and pin non-trivial disaggregation dims (SEX, AGE,
    # URBANISATION, COMP_BREAKDOWN_1/2) to their _T (aggregate total) value.
    # This prevents duplicate rows per country per year — which would otherwise
    # corrupt ranking statistics — when an indicator carries sex/age breakdowns.
    # Caller-provided disaggregation_filters always take precedence.
    sample_country = resolved_codes[0] if resolved_codes else "WLD"
    effective_filters, _ = await _auto_detect_disagg_dimensions(
        database_id=database_id,
        indicator_id=indicator_id,
        sample_country=sample_country,
        existing_filters=filters_for_auto,
        expand_non_trivial=False,
    )
    if global_member_ranking and effective_filters:
        effective_filters.pop("REF_AREA", None)

    # Fetch all pages: explicit list uses REF_AREA pin; global uses unpinned full geography.
    try:
        data_response = await _fetch_all_pages(
            database_id=database_id,
            indicator_id=indicator_id,
            country_code=(None if global_member_ranking else ";".join(resolved_codes)),
            disaggregation_filters=effective_filters or None,
            start_year=year - 2 if year else None,
            end_year=year + 1 if year else None,
        )
    except Exception as e:
        mcp_err = classify_error(e, context="rank")
        return RankingResponse(error=mcp_err.detail)

    if data_response.error:
        return RankingResponse(error=data_response.error)

    if not data_response.data:
        return RankingResponse(
            error="No data returned. Fall back to data360_get_data with broader filters.",
            metadata=data_response.metadata,
            total_requested=total_requested,
        )

    # Build {year: {country: (value, claim_id)}} index
    year_country_map: dict[str, dict[str, tuple[float, str]]] = {}
    for row in data_response.data:
        tp = str(row.get("TIME_PERIOD", ""))
        ra = str(row.get("REF_AREA", ""))
        if global_member_ranking and not ghm.is_country(ra):
            continue
        val = row.get("OBS_VALUE")
        cid = row.get("claim_id", "")
        obs = _obs_value_to_float(val)
        if tp and ra and obs is not None:
            year_country_map.setdefault(tp, {})[ra] = (obs, cid)

    # Select ranking year
    year_selection_note = None
    if year:
        ranking_year = str(year)
        year_selection_note = f"User-specified year: {year}"
        # If exact year not available, try closest
        if ranking_year not in year_country_map:
            available = sorted(year_country_map.keys())
            closest = (
                min(available, key=lambda y: abs(int(y) - year)) if available else None
            )
            if closest:
                ranking_year = closest
                year_selection_note = f"Requested {year}, closest available: {closest}"
            else:
                return RankingResponse(
                    error=f"No data available near year {year}.",
                    metadata=data_response.metadata,
                    total_requested=total_requested,
                )
    else:
        # Auto-select: find the latest available year with data
        latest_year = max(year_country_map.keys())
        ranking_year = latest_year

        # Find the year with the broadest coverage for comparison/caveat
        best_year = max(
            year_country_map.keys(),
            key=lambda y: (len(year_country_map[y]), y),
        )

        latest_coverage = len(year_country_map[latest_year])
        best_coverage = len(year_country_map[best_year])

        if latest_year == best_year:
            year_selection_note = (
                f"Latest available year ({latest_year}) with "
                f"{latest_coverage}/{total_requested} countries."
            )
        else:
            year_selection_note = (
                f"Latest available year: {latest_year} "
                f"({latest_coverage}/{total_requested} countries). "
                f"Note: Coverage is partial/incomplete for this year. "
                f"An older year ({best_year}) has broader coverage with "
                f"{best_coverage}/{total_requested} countries."
            )

    # Build ranking from selected year
    year_data = year_country_map.get(ranking_year, {})
    unit_measure = None
    if data_response.data:
        raw_unit = data_response.data[0].get("UNIT_MEASURE")
        if raw_unit:
            from .providers import get_codelist_manager  # noqa: PLC0415
            _cm = get_codelist_manager()
            await _cm._ensure_extdataportal_loaded()
            unit_measure = _cm.get_label("UNIT_MEASURE", str(raw_unit))

    # Batch-resolve country names in a single call
    name_map = await _resolve_country_names(resolved_codes)

    # Sort and rank
    entries = [(code, val, cid) for code, (val, cid) in year_data.items()]
    reverse = order.lower() != "asc"
    entries.sort(key=lambda x: x[1], reverse=reverse)

    # Assign ranks with tie handling.
    # Uses standard competition ranking (1,2,2,4): tied items share the same rank
    # and the next rank skips the number of tied items.  This matches the most
    # common convention for leaderboard rankings.
    rankings: list[RankedCountry] = []
    n_ranked = len(entries)
    for i, (code, val, cid) in enumerate(entries[:top_n]):
        if i == 0:
            rank = 1
        elif entries[i - 1][1] == val:
            # Same value as previous — share its rank
            rank = rankings[-1].rank
        else:
            # Different value — rank is one past the previous entry's position
            rank = i + 1

        percentile = (
            round(((n_ranked - i) / n_ranked) * 100, 1) if n_ranked > 0 else None
        )

        rankings.append(
            RankedCountry(
                rank=rank,
                ref_area=code,
                country_name=name_map.get(code),
                obs_value=round(val, 4),
                percentile=percentile,
                claim_id=cid,
            )
        )

    # Build excluded list
    excluded = []
    countries_with_data = set(year_data.keys())
    for code in resolved_codes:
        if code not in countries_with_data:
            excluded.append(
                ExcludedCountry(
                    ref_area=code,
                    country_name=name_map.get(code),
                    reason=f"No data for {ranking_year}",
                )
            )

    return RankingResponse(
        year=ranking_year,
        year_selection_note=year_selection_note,
        order=order,
        total_with_data=len(year_data),
        total_requested=total_requested,
        universe=universe,
        universe_size=universe_size,
        rankings=rankings,
        excluded=excluded,
        metadata=data_response.metadata,
        unit_measure=unit_measure,
    )


async def compare_countries(
    database_id: str,
    indicator_id: str,
    country_codes: str,
    year: int | None = None,
    include_time_series: bool = False,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
) -> CountryComparisonResponse:
    """Compare an indicator across multiple countries with ranking and gap analysis.

    Call for PATH B (comparison) questions like "Compare GDP between Kenya and Nigeria"
    or "How does Brazil compare to its neighbors on poverty?". Returns a pre-ranked
    snapshot and optional aligned time series with convergence analysis.

    IMPORTANT LIMITATION: This tool supports comparing 2 to 8 countries. If the user asks to
    compare more than 8 countries, you MUST NOT use this tool.
    Instead, use rank_countries or summarize_data.

    The snapshot includes a year_selection_note explaining how the comparison year was
    chosen — either the user-specified year, or the latest year where all compared
    countries have data. Both recency-based and coverage-based year selection are valid
    strategies depending on the analytical intent.

    Do NOT call this for single-country queries — use data360_get_data or
    data360_summarize_data. Do NOT call this for ranking within a large group
    (>8 countries) — use data360_rank_countries instead.

    Args:
        database_id: Database identifier (e.g., "WB_WDI").
        indicator_id: Indicator ID (e.g., "WB_WDI_NY_GDP_PCAP_KD").
        country_codes: Semicolon-separated country codes (e.g. "KEN;NGA;ZAF").
            Supports 2-8 countries.
        year: Comparison year. None = latest year where all countries have data.
        include_time_series: If True, include aligned time series + convergence.
        start_year: For time series mode. Defaults to last 5 years.
        end_year: For time series mode. Defaults to current year.
        disaggregation_filters: Optional dimension filters.

    Returns:
        CountryComparisonResponse:
            snapshot: Single-year ranked comparison with spread statistics.
            time_series: Aligned series + convergence (when include_time_series=True).
            metadata: Indicator metadata.
            unit_measure: Unit string.
            error: Error message if request failed; otherwise None.
                Falls back to data360_get_data if this tool encounters an error.
    """
    codes = validated_country_codes(country_codes)
    if len(codes) < 2:
        return CountryComparisonResponse(
            error="At least 2 country codes required for comparison."
        )

    # Auto-detect UNIT_MEASURE and pin non-trivial disaggregation dims (SEX, AGE,
    # URBANISATION, COMP_BREAKDOWN_1/2) to their _T (aggregate total) value.
    # This prevents duplicate rows per country per year — which would otherwise
    # corrupt comparison statistics — when an indicator carries sex/age breakdowns.
    # Caller-provided disaggregation_filters always take precedence.
    effective_filters, _ = await _auto_detect_disagg_dimensions(
        database_id=database_id,
        indicator_id=indicator_id,
        sample_country=codes[0],
        existing_filters=dict(disaggregation_filters or {}),
        expand_non_trivial=False,
    )

    # Fetch all pages for all countries in one batched call
    try:
        data_response = await _fetch_all_pages(
            database_id=database_id,
            indicator_id=indicator_id,
            country_code=";".join(codes),
            disaggregation_filters=effective_filters or None,
            start_year=start_year or (year - 10 if year else None),
            end_year=end_year or (year + 1 if year else None),
        )
    except Exception as e:
        mcp_err = classify_error(e, context="compare")
        return CountryComparisonResponse(error=mcp_err.detail)

    if data_response.error:
        return CountryComparisonResponse(error=data_response.error)
    if not data_response.data:
        return CountryComparisonResponse(
            error="No data returned. Fall back to data360_get_data with broader filters.",
            metadata=data_response.metadata,
        )

    # Index: {country: {year: (value, claim_id)}}
    country_year_map: dict[str, dict[str, tuple[float, str]]] = {}
    for row in data_response.data:
        ra = str(row.get("REF_AREA", ""))
        tp = str(row.get("TIME_PERIOD", ""))
        val = row.get("OBS_VALUE")
        cid = row.get("claim_id", "")
        obs = _obs_value_to_float(val)
        if ra and tp and obs is not None:
            country_year_map.setdefault(ra, {})[tp] = (obs, cid)

    # Batch-resolve country names in a single call
    name_map = await _resolve_country_names(codes)

    unit_measure = None
    if data_response.data:
        raw_unit = data_response.data[0].get("UNIT_MEASURE")
        if raw_unit:
            from .providers import get_codelist_manager  # noqa: PLC0415
            _cm = get_codelist_manager()
            await _cm._ensure_extdataportal_loaded()
            unit_measure = _cm.get_label("UNIT_MEASURE", str(raw_unit))

    # Determine snapshot year
    all_years = set()
    for yrs in country_year_map.values():
        all_years.update(yrs.keys())
    common_years = sorted(all_years)
    for c in codes:
        if c not in country_year_map:
            common_years = []
            break
        common_years = [y for y in common_years if y in country_year_map[c]]

    year_selection_note = None
    if year:
        snap_year = str(year)
        year_selection_note = f"User-specified year: {year}"
    elif common_years:
        snap_year = common_years[-1]  # Latest common year
        n_countries = sum(
            1
            for c in codes
            if c in country_year_map and snap_year in country_year_map[c]
        )
        year_selection_note = (
            f"Latest year with data for all compared countries: {snap_year} "
            f"({n_countries}/{len(codes)} countries)"
        )
    else:
        # Fallback: latest year from any country
        snap_year = max(all_years) if all_years else None
        if snap_year:
            n_countries = sum(
                1
                for c in codes
                if c in country_year_map and snap_year in country_year_map[c]
            )
            year_selection_note = (
                f"Latest year with data (partial coverage): {snap_year} "
                f"({n_countries}/{len(codes)} countries)"
            )
        else:
            year_selection_note = "No data available for any country."

    # Build snapshot
    snapshot = None
    if snap_year:
        snap_entries = []
        for code in codes:
            entry = country_year_map.get(code, {}).get(snap_year)
            if entry:
                snap_entries.append((code, entry[0], entry[1]))

        snap_entries.sort(key=lambda x: x[1], reverse=True)
        leader_val = snap_entries[0][1] if snap_entries else 0

        ranked = []
        for i, (code, val, cid) in enumerate(snap_entries):
            gap = (
                round(((val - leader_val) / leader_val) * 100, 2)
                if leader_val != 0
                else 0
            )
            ranked.append(
                RankedCountry(
                    rank=i + 1,
                    ref_area=code,
                    country_name=name_map.get(code),
                    obs_value=round(val, 4),
                    percentile=None,
                    claim_id=cid,
                )
            )

        vals_s = pd.Series([e[1] for e in snap_entries], dtype=float)
        spread: dict[str, float | None] = {}
        if not vals_s.empty:
            spread = {
                "min": round(float(vals_s.min()), 4),
                "max": round(float(vals_s.max()), 4),
                "range": round(float(vals_s.max() - vals_s.min()), 4),
                "coefficient_of_variation": (
                    round(float(vals_s.std() / vals_s.mean()), 4)
                    if len(vals_s) > 1 and vals_s.mean() != 0
                    else None
                ),
            }

        snapshot = ComparisonSnapshot(
            year=snap_year,
            year_selection_note=year_selection_note,
            rankings=ranked,
            spread=spread,
        )

    # Build time series (if requested) using pandas for alignment, CAGR, convergence
    ts_response = None
    if include_time_series and common_years:
        # Build a value pivot: rows=TIME_PERIOD, columns=REF_AREA
        # Use country_year_map (already built) to construct the pivot cleanly
        val_records = [
            {"TIME_PERIOD": y, "REF_AREA": c, "OBS_VALUE": v, "claim_id": cid}
            for c, yr_map in country_year_map.items()
            for y, (v, cid) in yr_map.items()
        ]
        ts_df = pd.DataFrame(val_records)
        ts_df["OBS_VALUE"] = pd.to_numeric(ts_df["OBS_VALUE"], errors="coerce")

        # Pivot: rows=TIME_PERIOD, columns=REF_AREA — NaN where country has no data
        val_pivot = ts_df.pivot_table(
            index="TIME_PERIOD", columns="REF_AREA", values="OBS_VALUE", aggfunc="last"
        )
        # Common years = rows where ALL requested countries have data
        aligned_pivot = val_pivot[codes].dropna(axis=0)
        aligned_years = sorted(aligned_pivot.index.tolist())

        # Build series output (with claim_ids from country_year_map)
        series: dict[str, list[dict[str, Any]]] = {}
        for code in codes:
            code_data = country_year_map.get(code, {})
            series[code] = [
                {
                    "time_period": y,
                    "obs_value": code_data[y][0],
                    "claim_id": code_data[y][1],
                }
                for y in aligned_years
                if y in code_data
            ]

        # CAGR: vectorized over aligned pivot
        cagr: dict[str, float | None] = {}
        if len(aligned_years) >= 2:
            first_y, last_y = aligned_years[0], aligned_years[-1]
            n_years = int(last_y) - int(first_y)
            for code in codes:
                if code in aligned_pivot.columns and n_years > 0:
                    v0 = aligned_pivot.loc[first_y, code]
                    v1 = aligned_pivot.loc[last_y, code]
                    # CAGR requires a positive base and a non-negative end value.
                    # Raising a negative ratio to a fractional power raises ValueError,
                    # so guard against both v0 <= 0 and v1 < 0.
                    cagr[code] = (
                        round(float(((v1 / v0) ** (1 / n_years) - 1) * 100), 2)
                        if pd.notna(v0) and pd.notna(v1) and v0 > 0 and v1 >= 0
                        else None
                    )
                else:
                    cagr[code] = None
        else:
            cagr = {code: None for code in codes}

        # Convergence: classify trend in coefficient of variation across aligned years
        convergence = None
        if len(aligned_years) >= 3 and len(codes) > 1:
            row_means = aligned_pivot[codes].mean(axis=1)
            row_stds = aligned_pivot[codes].std(axis=1, ddof=1)
            # CV per year — only where mean != 0
            cvs_s = (
                (row_stds / row_means)
                .replace([float("inf"), float("-inf")], pd.NA)
                .dropna()
            )
            if len(cvs_s) >= 3:
                cv_trend = _compute_trend_direction(cvs_s.tolist())
                convergence = {
                    "decreasing": "converging",
                    "increasing": "diverging",
                }.get(cv_trend, "parallel")

        ts_response = ComparisonTimeSeries(
            aligned_years=aligned_years,
            series=series,
            convergence=convergence,
            cagr=cagr,
        )

    return CountryComparisonResponse(
        snapshot=snapshot,
        time_series=ts_response,
        metadata=data_response.metadata,
        unit_measure=unit_measure,
        country_names=name_map,
    )


# ---------------------------------------------------------------------------
# Data Aggregation Tools (Tier 2 — stubs for future implementation)
# ---------------------------------------------------------------------------

_TIER2_STUB_MSG = (
    "This tool is not yet implemented. "
    "Fall back to data360_get_data for raw data retrieval, "
    "then use data360_summarize_data or data360_rank_countries for analysis."
)


async def compute_derived(
    database_id: str,
    indicator_id: str,
    country_code: str | None = None,
    computation: str = "growth_rate",
    window: int = 5,
    base_year: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    disaggregation_filters: dict[str, str | None] | None = None,
) -> DerivedDataResponse:
    """Compute derived/transformed values from raw indicator data (not yet implemented).

    Will support: growth_rate (YoY %), cagr (compound annual), moving_average,
    period_average, period_change (absolute + %), index (rebase to base_year=100).
    Useful for PATH C (trend) questions requiring arithmetic.

    Current status: stub. Falls back with recoverable error directing the LLM
    to use data360_get_data + data360_summarize_data as alternatives.
    """
    return DerivedDataResponse(error=_TIER2_STUB_MSG)


async def pivot_table(
    entries: list[dict[str, str]],
    country_codes: str,
    year: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    rows: str = "ref_area",
    columns: str = "indicator",
    value_agg: str = "latest",
) -> PivotTableResponse:
    """Build a cross-tabulation table from multiple indicators (not yet implemented).

    Will create a country-by-indicator matrix for PATH D (analytical) questions
    where multiple diagnostic indicators need to be organized into a single table.

    Current status: stub. Falls back with recoverable error directing the LLM
    to call data360_get_data separately per indicator.
    """
    return PivotTableResponse(error=_TIER2_STUB_MSG)


async def diagnostic_summary(
    topic: str,
    country_code: str,
    start_year: int | None = None,
    end_year: int | None = None,
    max_indicators: int = 5,
) -> DiagnosticSummaryResponse:
    """Produce a multi-indicator diagnostic summary for a topic (not yet implemented).

    Will map CONCEPT VOCABULARY categories to search queries, fetch data for the
    most diagnostic indicators, and return structured per-indicator trend analysis.
    Useful for PATH D (analytical) and PATH E (policy bridge) questions.

    Current status: stub. Falls back with recoverable error directing the LLM
    to decompose the question manually using data360_search_indicators + data360_get_data.
    """
    return DiagnosticSummaryResponse(error=_TIER2_STUB_MSG)
