"""Providers for Data360 codelist and reference area data."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

from data360.config import get_data360_settings
from data360.entropy import uniform_jitter
from data360.http_client import (
    RETRY_SAFE_EXTENSION,
    get_background_httpx_client,
    get_shared_httpx_client,
)

data360_config = get_data360_settings()

_logger = logging.getLogger(__name__)


def _describe_error(exc: BaseException | None) -> str:
    """Format an exception for logs; httpx timeouts often carry no message."""
    if exc is None:
        return "unknown error"
    message = str(exc).strip()
    return f"{type(exc).__name__}: {message}" if message else type(exc).__name__


class DatabaseManager:
    """Manages the list of databases dynamically fetched from the Data360 API.

    On startup, loads a complete JSON fallback so the mapping is never empty.
    All subsequent live refreshes happen in a background asyncio task, ensuring
    that get_mapping() is always a sub-millisecond in-memory dict lookup with
    no I/O or network cost on the hot path.
    """

    def __init__(self, ttl_seconds: float = 86400.0):
        self._cache: dict[str, str] = {}
        self._last_fetched: float = 0.0
        self._ttl = ttl_seconds
        self._bg_task: asyncio.Task | None = None
        self._load_fallback()

    def _load_fallback(self):
        """Load the bundled databases.json at startup to guarantee a non-empty cache."""
        fallback_path = Path(__file__).parent / "databases.json"
        try:
            with open(fallback_path, encoding="utf-8") as f:
                self._cache = json.load(f)
            _logger.info("Loaded %d databases from fallback JSON.", len(self._cache))
        except Exception as e:
            _logger.error("Failed to load fallback database mapping: %s", e)
            self._cache = {}

    # ------------------------------------------------------------------
    # Public API — always a pure in-memory read, never blocks on network
    # ------------------------------------------------------------------

    async def get_mapping(self) -> dict[str, str]:
        """Return the cached database_id→name mapping.

        Always returns immediately from in-memory cache. A background task
        is responsible for keeping the cache fresh via periodic API fetches.
        """
        self._ensure_background_sync()
        return self._cache

    def resolve_database_ids(self, query: str | None, strict: bool = False) -> list[str]:
        """Resolve database search terms (semicolon separated) to database IDs from the cache.

        Matches by exact ID, exact acronym suffix, exact name, or substring (if not strict).
        Raises ValueError if any term cannot be resolved.
        """
        if not query:
            return []
        tokens = [t.strip() for t in query.split(';') if t.strip()]
        resolved_ids = []
        for token in tokens:
            resolved = self._resolve_single_database_id(token, strict=strict)
            if resolved:
                if resolved not in resolved_ids:
                    resolved_ids.append(resolved)
            else:
                raise ValueError(f"Database '{token}' could not be resolved.")
        return resolved_ids

    def _resolve_single_database_id(self, query: str, strict: bool = False) -> str | None:
        query_lower = query.lower().strip()

        # 1. Exact match on database ID (key)
        for db_id in self._cache:
            if query_lower == db_id.lower():
                return db_id

        # 2. Exact match on database ID acronym suffix (e.g. "wdi" matches "wb_wdi")
        for db_id in self._cache:
            parts = db_id.split('_')
            if len(parts) > 1 and query_lower == parts[-1].lower():
                return db_id

        # 3. Exact match on database name (value)
        for db_id, name in self._cache.items():
            if query_lower == name.lower():
                return db_id

        if not strict:
            # 4. Substring match on database ID
            for db_id in self._cache:
                if query_lower in db_id.lower():
                    return db_id

            # 5. Substring match on database name
            for db_id, name in self._cache.items():
                if query_lower in name.lower():
                    return db_id

            # 6. Fuzzy prefix/character match on ID and name as fallback
            best_match = None
            best_score = 0.70
            for db_id, name in self._cache.items():
                score_id = self._calculate_similarity(query_lower, db_id.lower())
                score_name = self._calculate_similarity(query_lower, name.lower())
                max_score = max(score_id, score_name)
                if max_score > best_score:
                    best_score = max_score
                    best_match = db_id
            if best_match:
                return best_match

        return None

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0

        if len(s1) <= 3:
            if s1 in s2 or s2.startswith(s1):
                return 0.8
            return 0.0

        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max(len1, len2) * 0.5:
            return 0.0

        s1_chars = set(s1)
        s2_chars = set(s2)
        common = len(s1_chars & s2_chars)
        total = len(s1_chars | s2_chars)
        char_similarity = common / total if total > 0 else 0

        prefix_len = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        prefix_ratio = prefix_len / min(len1, len2)

        return (char_similarity * 0.4) + (prefix_ratio * 0.6)

    def resolve_database_id(self, query: str | None) -> str | None:
        """Resolve a database search term to a single database ID from the cache.

        Deprecated: use resolve_database_ids instead.
        """
        try:
            ids = self.resolve_database_ids(query)
            return ids[0] if ids else None
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Background sync machinery
    # ------------------------------------------------------------------

    def _ensure_background_sync(self) -> None:
        """Spawn the background refresh loop if it is not already running."""
        if os.environ.get("PYTEST_RUNNING"):
            return
        if self._bg_task is None or self._bg_task.done():
            self._bg_task = asyncio.create_task(self._background_sync_loop())

    async def _background_sync_loop(self) -> None:
        """Run forever, refreshing the cache from the API when the TTL expires."""
        while True:
            elapsed = time.monotonic() - self._last_fetched
            if elapsed >= self._ttl:
                try:
                    mapping = await self._fetch_all()
                    if mapping:
                        self._cache = mapping
                        _logger.info(
                            "Background refresh: updated %d databases.", len(mapping)
                        )
                except Exception as e:
                    _logger.error("Background database fetch failed: %s", e)
                    # Keep existing cache; retry after the next full TTL cycle.
                finally:
                    self._last_fetched = time.monotonic()

            sleep_for = max(0.0, self._ttl - (time.monotonic() - self._last_fetched))
            await asyncio.sleep(sleep_for)

    async def _fetch_all(self) -> dict[str, str]:
        """Fetch all datasets from the search endpoint using pagination.

        Runs on the background/bulk client; transient failures are retried by the
        transport (bounded by the background retry budget), so failures propagate
        to the caller which logs and keeps the existing cache.
        """
        url = data360_config.search_url or f"{data360_config.api_url}/searchv2"
        mapping: dict[str, str] = {}
        skip = 0
        limit = 50

        client = get_background_httpx_client()
        while True:
            response = await client.post(
                url,
                headers={
                    "accept": "*/*",
                    "Content-Type": "application/json",
                },
                json={
                    "filter": "type eq 'dataset' and (is_active ne false or is_active eq null)",
                    "orderby": "series_description/name",
                    "select": "series_description/database_id, series_description/name",
                    "skip": skip,
                    "top": limit,
                },
                extensions={RETRY_SAFE_EXTENSION: True},
            )
            response.raise_for_status()
            data = response.json()
            items = data.get("value", [])

            if not items:
                break

            for x in items:
                sd = x.get("series_description", {})
                db_id = sd.get("database_id")
                db_name = sd.get("name")
                if db_id and db_name and db_id not in mapping:
                    mapping[db_id] = db_name

            skip += limit

        return mapping


# Global instance
_database_manager: DatabaseManager | None = None


def get_database_manager() -> DatabaseManager:
    """Get the global DatabaseManager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager


async def get_database_mapping() -> dict[str, str]:
    """Get mapping of database IDs to their actual names (e.g., {'WB_GS': 'Gender Statistics'})."""
    return await get_database_manager().get_mapping()


# ---------------------------------------------------------------------------
# Natural-language aliases for group codes.
#
# This map is intentionally minimal: it contains ONLY phrases where the
# existing fuzzy/substring search in CodelistManager._search_global would
# genuinely fail to return the correct code. The LLM and the fuzzy layer
# already handle most natural-language variations (e.g. "South Asia",
# "low income countries", "sub-saharan africa") without help from this map.
#
# Entries kept here fall into four categories:
#   1. "and" vs "&" variants — substring check fails because the official
#      codelist uses "&" but users type "and".
#   2. Abbreviations not present in the official name (e.g. "mena").
#   3. Common shorthands whose words don't appear in the official name
#      (e.g. "fragile states" vs "Fragile and conflict affected situations").
#   4. One semantic synonym where the variant word is absent from the
#      official name ("lower income" vs "Low income").
#
# All values are FMR group codes that must exist in ref_area_groups.json.
# Enforced by TestDataIntegrity.test_all_alias_values_exist_in_shipped_data.
#
# Future path if coverage proves insufficient: build-time sentence-transformer
# embeddings over all group names, cosine-similarity fallback after alias miss.
# ---------------------------------------------------------------------------
_GROUP_ALIASES: dict[str, str] = {
    # Category 1: "and" vs "&" — fuzzy fails because "&" != "and"
    "east asia and pacific": "EAS",  # official: "East Asia & Pacific"
    "europe and central asia": "ECS",  # official: "Europe & Central Asia"
    "latin america and the caribbean": "LCN",  # official: "Latin America & Caribbean"
    "middle east and north africa": "MEA",  # official: "Middle East, North Africa, Afghanistan & Pakistan"
    "middle east & north africa": "MEA",  # same but omits "Afghanistan & Pakistan"
    "low and middle income": "LMY",  # official: "Low & middle income"
    # Category 2: abbreviation not in official name
    "mena": "MEA",  # common abbreviation for the region
    # Category 3: shorthands whose words don't appear in the official name
    "fragile states": "FCS",  # official: "Fragile and conflict affected situations"
    "small island states": "SST",  # official: "Small states"
    "eastern africa": "AFE",  # official: "Africa Eastern and Southern"
    "western africa": "AFW",  # official: "Africa Western and Central"
    # Category 4: semantic synonym — "lower" absent from "Low income"
    "lower income": "LIC",
}


# FMR endpoint templates (version segment is optional; omitting returns latest).
_FMR_HIERARCHY_URL = (
    "https://fmr.worldbank.org/FMR/sdmx/v2/structure/hierarchy/WB/H_REF_AREA_GROUPS/{version}"
    "?format=sdmx-json"
)
_FMR_CODELIST_URL = (
    "https://fmr.worldbank.org/FMR/sdmx/v2/structure/codelist/WB/CL_REF_GROUPINGS/{version}"
    "?format=sdmx-json"
)
# Group types whose entries are all leaf countries — not expandable groups.
_LEAF_ONLY_TYPES: frozenset[str] = frozenset({"WLD", "_T"})


class GroupHierarchyManager:
    """Manages REF_AREA group-to-country mappings from the FMR hierarchy.

    Startup behaviour (zero-latency):
      Loads the bundled src/data360/ref_area_groups.json immediately on first
      access so the tool is usable without any network I/O on the hot path.
      Re-generate that file with::

          uv run python scripts/build_ref_area_groups.py

    Background sync (TTL = 7 days):
      The first public method call from within an async context spawns a
      long-lived asyncio task that wakes once every 7 days and re-fetches
      the FMR hierarchy and codelist endpoints.  On a successful fetch the
      in-memory state is atomically replaced.

      FMR (https://fmr.worldbank.org) is a VPN-restricted resource.  A
      failed fetch is logged at WARNING level and the existing in-memory
      state is kept intact.  The loop then sleeps another full TTL cycle
      before trying again — non-VPN deployments therefore produce at most
      one warning per week and never retry aggressively.

    Covers all 147 group codes across 6 group types:
      REGION, INCOME, LENDING, OTHER, REGION_UN, CONTINENT.
    """

    # 7-day TTL: income group compositions change at most once a year (July 1).
    _TTL: float = 7 * 24 * 3600.0

    _DATA_FILE = Path(__file__).parent / "ref_area_groups.json"

    def __init__(self, include_types: set[str] | None = None) -> None:
        """Initialize and load the bundled JSON fallback synchronously.

        Args:
            include_types: If provided, only expose groups of these types.
                Defaults to all types. Common subset:
                {"REGION", "INCOME", "LENDING", "OTHER"}
        """
        self._groups: dict[str, dict[str, Any]] = {}
        self._all_countries: set[str] = set()
        self._meta: dict[str, Any] = {}
        self._include_types = include_types
        self._loaded = False
        self._initial_fetch_succeeded = False
        self._last_fetched: float = 0.0
        self._bg_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # SDMX parsing helpers (static — also imported by build_ref_area_groups.py)
    # ------------------------------------------------------------------

    @staticmethod
    def parse_name_map(codelist_data: dict) -> dict[str, str]:
        """Build a {code: name} map from a CL_REF_GROUPINGS SDMX-JSON payload."""
        codes = codelist_data["data"]["codelists"][0]["codes"]
        return {c["id"]: c["name"] for c in codes}

    @staticmethod
    def parse_hierarchy(
        hierarchy_data: dict,
        name_map: dict[str, str],
        include_types: set[str] | None = None,
    ) -> tuple[dict[str, dict], set[str], str]:
        """Parse H_REF_AREA_GROUPS SDMX-JSON into (groups, all_countries, version).

        groups format:
            {
                "SAS": {"name": "South Asia", "type": "REGION", "countries": [...]},
                ...
            }
        all_countries: set of every individual country code that appears in at
            least one group.
        version: hierarchy version string from the SDMX payload (e.g. "38.0").
        """
        hcl = hierarchy_data["data"]["hierarchicalCodelists"][0]
        version = hcl["version"]
        top_level = hcl["hierarchies"][0]["hierarchicalCodes"]

        groups: dict[str, dict] = {}
        all_countries: set[str] = set()

        for type_node in top_level:
            group_type = type_node["id"]
            if group_type in _LEAF_ONLY_TYPES:
                continue
            if include_types and group_type not in include_types:
                continue

            for group_node in type_node.get("hierarchicalCodes", []):
                group_id = group_node["id"]
                country_nodes = group_node.get("hierarchicalCodes", [])
                if not country_nodes:
                    continue

                countries = sorted(c["id"] for c in country_nodes)
                all_countries.update(countries)

                # Strip UTF-8 mojibake from some FMR source names.
                # (e.g. 9WN: "Western Asia\u00c2\u00a0 and ..." — \u00c2\u00a0 is
                # a UTF-8 non-breaking space decoded as Latin-1.)
                raw_name = name_map.get(group_id, group_id)
                clean_name = " ".join(raw_name.replace("\u00c2\u00a0", " ").split())

                groups[group_id] = {
                    "name": clean_name,
                    "type": group_type,
                    "countries": countries,
                }

        return groups, all_countries, version

    # ------------------------------------------------------------------
    # Bundled-JSON loader (synchronous, called once at __init__)
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load the bundled ref_area_groups.json. Called once at __init__."""
        if self._loaded:
            return
        try:
            with open(self._DATA_FILE, encoding="utf-8") as f:
                data = json.load(f)
            self._apply(data)
            _logger.info(
                "GroupHierarchyManager: loaded %d groups (%d countries) from bundled file. Source: %s",
                len(self._groups),
                len(self._all_countries),
                self._meta.get("source", "unknown"),
            )
        except FileNotFoundError:
            _logger.error(
                "GroupHierarchyManager: bundled data file not found at %s. "
                "Run scripts/build_ref_area_groups.py to generate it.",
                self._DATA_FILE,
            )
        except Exception as e:
            _logger.error("GroupHierarchyManager: failed to load bundled data: %s", e)
        finally:
            # Mark as loaded regardless so _ensure_loaded does not retry
            # the synchronous disk read in a loop.
            self._loaded = True

    def _apply(self, data: dict) -> None:
        """Atomically swap the in-memory state from a parsed data dict."""
        raw_groups: dict[str, dict] = data.get("groups", {})
        if self._include_types:
            raw_groups = {
                k: v
                for k, v in raw_groups.items()
                if v.get("type") in self._include_types
            }
        self._groups = raw_groups
        self._all_countries = set(data.get("all_countries", []))
        self._meta = data.get("_meta", {})

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._load()

    # ------------------------------------------------------------------
    # Background sync machinery (mirrors DatabaseManager)
    # ------------------------------------------------------------------

    def _ensure_background_sync(self) -> None:
        """Spawn the background refresh loop if it is not already running.

        No-op when called outside a running event loop (e.g. in sync test code
        or during module import). The bundled JSON is always pre-loaded, so
        the background task is only an update mechanism, not a prerequisite.
        """
        if os.environ.get("PYTEST_RUNNING"):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop — skip task creation.
            return
        if self._bg_task is None or self._bg_task.done():
            self._bg_task = asyncio.create_task(self._background_sync_loop())

    async def _background_sync_loop(self) -> None:
        """Run forever, waking every TTL seconds to refresh from FMR.

        TTL policy:
          - Success: in-memory state is replaced; next wake is _TTL seconds later.
          - Failure: existing state is kept; _last_fetched is still advanced by
            _TTL so the next wake is also _TTL seconds later.  This means a
            non-VPN deployment gets at most one WARNING log per week rather than
            retrying aggressively.

        The loop is safe to cancel: cancellation propagates through
        asyncio.sleep and exits cleanly.
        """
        backoff = 5.0
        max_backoff = 900.0  # 15 minutes
        while True:
            if self._initial_fetch_succeeded:
                elapsed = time.monotonic() - self._last_fetched
                sleep_for = max(0.0, self._TTL - elapsed)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    continue

            try:
                groups, all_countries, version = await self._fetch_fmr_data()
                # Build a minimal data dict compatible with _apply.
                data = {
                    "_meta": {
                        "source": f"FMR H_REF_AREA_GROUPS v{version} + CL_REF_GROUPINGS (live)",
                        "hierarchy_version": version,
                    },
                    "groups": groups,
                    "all_countries": sorted(all_countries),
                }
                self._apply(data)
                self._initial_fetch_succeeded = True
                self._last_fetched = time.monotonic()
                _logger.info(
                    "GroupHierarchyManager: background refresh completed — "
                    "%d groups (%d countries), hierarchy v%s.",
                    len(self._groups),
                    len(self._all_countries),
                    version,
                )
                backoff = 5.0
                sleep_for = self._TTL
            except Exception as e:
                if not self._initial_fetch_succeeded:
                    sleep_for = backoff
                    backoff = min(backoff * 2 + uniform_jitter(0.1, 1.0), max_backoff)
                    _logger.warning(
                        "GroupHierarchyManager: initial background FMR fetch failed (%s). "
                        "Retrying in %.1f seconds.",
                        e,
                        sleep_for,
                    )
                else:
                    sleep_for = self._TTL
                    self._last_fetched = time.monotonic()
                    _logger.warning(
                        "GroupHierarchyManager: background FMR fetch failed (%s). "
                        "Next attempt in %.0f days.",
                        e,
                        self._TTL / 86400,
                    )

            await asyncio.sleep(sleep_for)

    async def _fetch_fmr_data(
        self,
        hierarchy_version: str = "",
        codelist_version: str = "",
    ) -> tuple[dict[str, dict], set[str], str]:
        """Fetch and parse the FMR hierarchy + codelist endpoints.

        Args:
            hierarchy_version: Specific version string (e.g. "38.0").  If empty,
                the FMR API returns the latest version.
            codelist_version: Specific codelist version (e.g. "2.0").  If empty,
                the FMR API returns the latest version.

        Returns:
            Tuple of (groups, all_countries, version) — same shape as
            parse_hierarchy().
        """
        hierarchy_url = _FMR_HIERARCHY_URL.format(version=hierarchy_version)
        codelist_url = _FMR_CODELIST_URL.format(version=codelist_version)

        client = get_background_httpx_client()
        h_resp, cl_resp = await asyncio.gather(
            client.get(hierarchy_url, headers={"Accept": "application/json"}),
            client.get(codelist_url, headers={"Accept": "application/json"}),
        )
        h_resp.raise_for_status()
        cl_resp.raise_for_status()
        hierarchy_data = h_resp.json()
        codelist_data = cl_resp.json()

        name_map = self.parse_name_map(codelist_data)
        return self.parse_hierarchy(hierarchy_data, name_map, self._include_types)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_group(self, code: str) -> bool:
        """Return True if the code is a known country group (not a leaf country)."""
        self._ensure_loaded()
        self._ensure_background_sync()
        return code.upper() in self._groups

    def is_country(self, code: str) -> bool:
        """Return True if the code is an individual country in the hierarchy."""
        self._ensure_loaded()
        return code.upper() in self._all_countries and not self.is_group(code.upper())

    def list_rankable_country_codes(self) -> list[str]:
        """Return sorted leaf economy codes (FMR member countries, excluding group aggregates).

        Used for ranking metadata and for ``member_economies_only`` filtering; the Data API
        may still return additional aggregate codes when ``REF_AREA`` is unpinned—those
        should be dropped with :meth:`is_country` per row.
        """
        self._ensure_loaded()
        self._ensure_background_sync()
        return sorted(c for c in self._all_countries if self.is_country(c))

    def get_group_info(self, code: str) -> dict[str, Any] | None:
        """Return {name, type, countries} for a group code, or None if unknown."""
        self._ensure_loaded()
        self._ensure_background_sync()
        return self._groups.get(code.upper())

    def get_group_type(self, code: str) -> str | None:
        """Return the group type string (REGION, INCOME, ...) or None."""
        info = self.get_group_info(code)
        return info["type"] if info else None

    def expand_group(self, code: str) -> list[str]:
        """Return sorted list of country codes in a group, or [] if unknown."""
        info = self.get_group_info(code)
        return list(info["countries"]) if info else []

    def get_meta(self) -> dict[str, Any]:
        """Return metadata about the loaded hierarchy (version, built_at, etc.)."""
        self._ensure_loaded()
        return self._meta

    def search_groups(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search group names by substring. Returns [{id, name, type, count}]."""
        self._ensure_loaded()
        q = query.lower().strip()
        results = []
        for gid, info in self._groups.items():
            if q in info["name"].lower() or q == gid.lower():
                results.append(
                    {
                        "id": gid,
                        "name": info["name"],
                        "type": info["type"],
                        "count": len(info["countries"]),
                    }
                )
        results.sort(key=lambda x: x["name"])
        return results[:limit]


# Global instance
_group_hierarchy_manager: GroupHierarchyManager | None = None


def get_group_hierarchy_manager() -> GroupHierarchyManager:
    """Get the global GroupHierarchyManager instance."""
    global _group_hierarchy_manager
    if _group_hierarchy_manager is None:
        _group_hierarchy_manager = GroupHierarchyManager()
    return _group_hierarchy_manager


def _make_group_note(group_code: str, info: dict) -> str:
    """Build the standard guidance note for a REF_AREA group result."""
    return (
        f"This is a {info['type'].lower()} group with "
        f"{len(info['countries'])} member countries. "
        f"Use data360_expand_country_group('{group_code}') "
        "to get individual country codes."
    )


_UNIT_MEASURE_FALLBACKS = {
    "ZS": "Percent",
    "PS": "Persons",
    "USD": "US Dollars",
    "LCU": "Local Currency Unit",
    "DY": "Days",
    "YR": "Years",
    "MR": "Meters",
    "KG": "Kilograms",
    "TN": "Tonnes",
}


class CodelistManager:
    """Unified manager for all Data360 codelists.

    Primary data source (lazy startup fetch, then background refresh):
      On first use, it fetches all dimension codelists from the extdataportal
      metadata API in a single HTTP call and populates ``_extdataportal``.
      Covers COMP_BREAKDOWN (5 000+ codes), UNIT_MEASURE (769 codes),
      AGE (173 codes), URBANISATION (16 codes), SEX (7 codes), FREQ (34 codes),
      and REF_AREA (532 codes).

      After the initial fetch a background task wakes every ``_TTL`` seconds
      (default 7 days) and atomically replaces the in-memory mapping so the
      server always has fresh labels without restarting.

    Graceful degradation:
      If the network is unavailable at startup ``_extdataportal`` stays empty
      and ``get_label()`` / ``get_dimension_labels()`` return the raw code
      unchanged.  The background loop retries after the next TTL cycle.

    Legacy paths (kept for backward compatibility):
      ``find_value()`` / ``get_codelist_mapping()`` still work for REF_AREA and
      UNIT_MEASURE via the old per-type API fetch (used by the
      ``find_codelist_value`` MCP tool).
      ``STATIC_MAPPINGS`` (name→code) is kept for the ``find_value()`` search
      path on SEX, AGE, URBANISATION, FREQ.

    New preferred path:
      ``get_label(dimension, code)`` — O(1) code→name lookup.
      ``get_dimension_labels(dimension)`` — full {code: name} dict.
    """

    # 7-day TTL: codelists rarely change; this mirrors GroupHierarchyManager.
    _TTL: float = 7 * 24 * 3600.0

    # COMP_BREAKDOWN_1/2/3 are identical on the API — stored once under this key.
    _COMP_BREAKDOWN_DIMS: frozenset[str] = frozenset(
        {"COMP_BREAKDOWN_1", "COMP_BREAKDOWN_2", "COMP_BREAKDOWN_3"}
    )
    _COMP_BREAKDOWN_KEY = "COMP_BREAKDOWN"

    # Prefix stripped from COMP_BREAKDOWN labels for chart readability.
    _COMP_BREAKDOWN_STRIP_PREFIX = "Metric: "

    # Codelists available via /codelist?type=X API
    GLOBAL_CODELISTS = ["REF_AREA", "UNIT_MEASURE"]

    # Static mappings for codelists without global API endpoints
    # These are based on actual values from disaggregation responses
    # Reference: WB_SSGD_UNEMPLOYMENT_RATE_disaggregation.json
    STATIC_MAPPINGS: dict[str, dict[str, str]] = {
        "FREQ": {
            # Common frequency codes found in Data360
            "annual": "A",
            "yearly": "A",
            "year": "A",
            "monthly": "M",
            "month": "M",
            "quarterly": "Q",
            "quarter": "Q",
            "other": "_O",
            "irregular": "_O",
        },
        "SEX": {
            # Sex codes from actual disaggregation data
            "female": "F",
            "women": "F",
            "woman": "F",
            "girls": "F",
            "girl": "F",
            "male": "M",
            "men": "M",
            "man": "M",
            "boys": "M",
            "boy": "M",
            "total": "_T",
            "all": "_T",
            "both": "_T",
            "overall": "_T",
        },
        "AGE": {
            # Age group codes from actual disaggregation data
            "youth": "Y15T24",
            "young": "Y15T24",
            "15-24": "Y15T24",
            "15 to 24": "Y15T24",
            "young adults": "Y15T29",
            "15-29": "Y15T29",
            "adults": "Y30T59",
            "30-59": "Y30T59",
            "25+": "Y_GE25",
            "25 and over": "Y_GE25",
            "adult": "Y_GE25",
            "elderly": "Y_GE60",
            "60+": "Y_GE60",
            "60 and over": "Y_GE60",
            "seniors": "Y_GE60",
            "total": "_T",
            "all ages": "_T",
        },
        "URBANISATION": {
            # Urbanisation codes from actual disaggregation data
            "urban": "URB",
            "city": "URB",
            "cities": "URB",
            "rural": "RUR",
            "countryside": "RUR",
            "village": "RUR",
            "total": "_T",
            "all": "_T",
        },
    }

    def __init__(self) -> None:
        """Initialise the CodelistManager.

        ``_extdataportal`` starts empty.  It is populated lazily on first use.
        """
        self._cache: dict[str, list[dict[str, Any]]] = {}
        self._loaded: set[str] = set()
        # Runtime-fetched extdataportal data: {dimension → {code → name}}.
        # Populated lazily; stays empty (graceful fallback) if offline.
        self._extdataportal: dict[str, dict[str, str]] = {}
        self._initial_fetch_succeeded = False
        self._last_fetched: float = 0.0
        self._last_fetch_error: Exception | None = None
        self._fetch_task: asyncio.Task | None = None  # type: ignore[type-arg]
        self._bg_task: asyncio.Task | None = None  # type: ignore[type-arg]

    # ------------------------------------------------------------------
    # Extdataportal fetch helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_extdataportal_response(raw: dict) -> dict[str, dict[str, str]]:
        """Parse the extdataportal JSON response into {dimension: {code: name}}.

        Deduplicates COMP_BREAKDOWN_1/2/3 into a single COMP_BREAKDOWN key and
        strips the ``'Metric: '`` prefix from COMP_BREAKDOWN labels.
        """
        built: dict[str, dict[str, str]] = {}
        strip = CodelistManager._COMP_BREAKDOWN_STRIP_PREFIX
        cb_key = CodelistManager._COMP_BREAKDOWN_KEY
        cb_dims = CodelistManager._COMP_BREAKDOWN_DIMS

        for dim, items in raw.items():
            if dim == "_meta" or not isinstance(items, list):
                continue
            # Normalise dimension key
            norm = dim.upper()
            target_key = cb_key if norm in cb_dims else norm
            # Build {id: name} — strip 'Metric: ' prefix from COMP_BREAKDOWN
            mapping: dict[str, str] = {}
            for item in items:
                if not isinstance(item, dict) or "id" not in item:
                    continue
                name = item.get("name", item["id"])
                if target_key == cb_key and name.startswith(strip):
                    name = name[len(strip):]
                mapping[item["id"]] = name
            # Merge into existing entry (COMP_BREAKDOWN_1/2/3 all write to cb_key)
            if target_key in built:
                built[target_key].update(mapping)
            else:
                built[target_key] = mapping

        return built

    async def _fetch_extdataportal(self) -> dict[str, dict[str, str]]:
        """Fetch all dimension codelists from the extdataportal metadata API.

        Runs on the background/bulk client: the payload is ~1.7 MB and nobody is
        waiting on it, so it does not share the interactive pool or timeouts.

        Returns:
            Parsed {dimension: {code: name}} mapping.

        Raises:
            httpx.HTTPStatusError / httpx.RequestError on network failure.
        """
        url = data360_config.codelist_api_base_url
        client = get_background_httpx_client()
        response = await client.get(url)
        response.raise_for_status()
        raw: dict = response.json()
        return self._parse_extdataportal_response(raw)

    def _apply_extdataportal(self, mapping: dict[str, dict[str, str]]) -> None:
        """Atomically replace the in-memory extdataportal mapping."""
        self._extdataportal = mapping
        _logger.info(
            "CodelistManager: extdataportal codelists loaded — %d dimensions, "
            "COMP_BREAKDOWN=%d codes.",
            len(mapping),
            len(mapping.get(self._COMP_BREAKDOWN_KEY, {})),
        )

    async def _fetch_and_apply_extdataportal(self) -> None:
        """Run one catalog fetch and record the outcome instead of raising.

        Failures are stored in ``_last_fetch_error`` so the background refresh loop
        can apply its own backoff, and so a caller that stopped waiting never leaves
        an unretrieved task exception behind.
        """
        try:
            mapping = await self._fetch_extdataportal()
        except Exception as exc:  # noqa: BLE001 - reported via _last_fetch_error
            self._last_fetch_error = exc
            _logger.warning(
                "CodelistManager: extdataportal fetch failed (%s). "
                "Dimension codes will display as raw values until the "
                "next retry.",
                _describe_error(exc),
            )
            return

        self._last_fetch_error = None
        self._apply_extdataportal(mapping)
        self._initial_fetch_succeeded = True
        self._last_fetched = time.monotonic()

    def _start_extdataportal_fetch(self) -> "asyncio.Task[None]":
        """Return the in-flight catalog fetch, starting one if none is running.

        Single-flight: concurrent callers (request paths and the refresh loop)
        share one 1.7 MB fetch instead of each starting their own.
        """
        task = self._fetch_task
        if task is None or task.done():
            task = asyncio.create_task(self._fetch_and_apply_extdataportal())
            self._fetch_task = task
        return task

    # ------------------------------------------------------------------
    # Lazy loader (same pattern as _ensure_loaded for REF_AREA)
    # ------------------------------------------------------------------

    async def _ensure_extdataportal_loaded(self) -> None:
        """Fetch extdataportal codelists on first use, without blocking the request.

        The catalog is bulk background data, so a request waits at most
        ``DATA360_BACKGROUND_INLINE_WAIT_SECONDS`` for it and then continues with
        raw dimension codes while the fetch finishes in the background. Callers
        must therefore tolerate missing labels (``get_label`` already falls back
        to the raw code).
        """
        if not self._extdataportal:
            task = self._start_extdataportal_fetch()
            budget = get_data360_settings().background_inline_wait_seconds
            done, _ = await asyncio.wait({task}, timeout=budget)
            if not done:
                _logger.info(
                    "CodelistManager: catalog still loading after %.1fs; serving raw "
                    "dimension codes while the background fetch completes.",
                    budget,
                )
        self._ensure_background_refresh()

    # ------------------------------------------------------------------
    # Background refresh (mirrors GroupHierarchyManager pattern)
    # ------------------------------------------------------------------

    def _ensure_background_refresh(self) -> None:
        """Spawn the background refresh task if not already running.

        No-op outside a running event loop (e.g. in sync tests).
        """
        if os.environ.get("PYTEST_RUNNING"):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._bg_task is None or self._bg_task.done():
            self._bg_task = asyncio.create_task(self._background_refresh_loop())

    async def _background_refresh_loop(self) -> None:
        """Wake every TTL seconds and refresh from extdataportal.

        Shares the single-flight fetch with request paths. On success the
        in-memory mapping is atomically replaced. On failure the existing mapping
        is kept, a WARNING is logged, and the loop backs off (initial load) or
        waits another full TTL.
        """
        backoff = 5.0
        max_backoff = 900.0  # 15 minutes
        while True:
            if self._initial_fetch_succeeded:
                elapsed = time.monotonic() - self._last_fetched
                sleep_for = max(0.0, self._TTL - elapsed)
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                    continue

            await self._start_extdataportal_fetch()
            if self._last_fetch_error is None:
                _logger.info(
                    "CodelistManager: background refresh completed — "
                    "%d dimensions, COMP_BREAKDOWN=%d codes.",
                    len(self._extdataportal),
                    len(self._extdataportal.get(self._COMP_BREAKDOWN_KEY, {})),
                )
                backoff = 5.0
                sleep_for = self._TTL
            elif not self._initial_fetch_succeeded:
                sleep_for = backoff
                backoff = min(backoff * 2 + uniform_jitter(0.1, 1.0), max_backoff)
                _logger.warning(
                    "CodelistManager: initial background refresh failed (%s). "
                    "Retrying in %.1f seconds.",
                    _describe_error(self._last_fetch_error),
                    sleep_for,
                )
            else:
                sleep_for = self._TTL
                self._last_fetched = time.monotonic()
                _logger.warning(
                    "CodelistManager: background refresh failed (%s). "
                    "Next attempt in %.0f days.",
                    _describe_error(self._last_fetch_error),
                    self._TTL / 86400,
                )

            await asyncio.sleep(sleep_for)

    # ------------------------------------------------------------------
    # Public label-lookup API (synchronous — reads pre-fetched dict)
    # ------------------------------------------------------------------

    def _resolve_extdataportal_key(self, dimension: str) -> str:
        """Normalise dimension name for extdataportal lookup.

        Maps COMP_BREAKDOWN_1/2/3 → COMP_BREAKDOWN; everything else is
        upper-cased and passed through unchanged.
        """
        upper = dimension.upper()
        if upper in self._COMP_BREAKDOWN_DIMS:
            return self._COMP_BREAKDOWN_KEY
        return upper

    def get_label(self, dimension: str, code: str) -> str:
        """Return the human-readable label for a dimension code.

        Reads from the in-memory ``_extdataportal`` dict populated by
        extdataportal mapping (lazy-loaded).  Returns the raw ``code`` unchanged if the
        dimension or code is not found (graceful degradation).

        Args:
            dimension: Dimension name, e.g. ``"COMP_BREAKDOWN_1"``, ``"SEX"``.
            code: The raw code value, e.g. ``"WGI_EST"``, ``"F"``.

        Returns:
            Human-readable label or the original ``code`` if not found.
        """
        key = self._resolve_extdataportal_key(dimension)
        if key == "UNIT_MEASURE" and code in _UNIT_MEASURE_FALLBACKS:
            return _UNIT_MEASURE_FALLBACKS[code]
        return self._extdataportal.get(key, {}).get(code, code)

    def get_dimension_labels(self, dimension: str) -> dict[str, str]:
        """Return a ``{code: name}`` mapping for an entire dimension.

        Suitable for vectorised DataFrame replacement via ``df[col].map(lookup)``.
        Returns an empty dict if the dimension is not found.

        Args:
            dimension: Dimension name, e.g. ``"COMP_BREAKDOWN_1"``, ``"UNIT_MEASURE"``.

        Returns:
            A copy of the code→name dict for the dimension.
        """
        key = self._resolve_extdataportal_key(dimension)
        return dict(self._extdataportal.get(key, {}))

    async def _ensure_loaded(self, codelist_type: str) -> None:
        """Ensure a global codelist is loaded from extdataportal."""
        if codelist_type in self._loaded:
            return
        if codelist_type in self.GLOBAL_CODELISTS:
            await self._ensure_extdataportal_loaded()
            ext_key = self._resolve_extdataportal_key(codelist_type)
            if ext_key in self._extdataportal and self._extdataportal[ext_key]:
                items = [
                    {"Id": code, "Name": name}
                    for code, name in self._extdataportal[ext_key].items()
                ]
                self._cache[codelist_type] = items
                self._loaded.add(codelist_type)

    def _normalize_query(self, query: str) -> str:
        """Normalize query for case-insensitive and whitespace-invariant search."""
        return query.lower().strip()

    async def find_value(
        self, codelist_type: str, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Find values in a codelist matching the query.

        Args:
            codelist_type: Type of codelist (REF_AREA, FREQ, SEX, etc.)
            query: Search query (e.g., "Kenya", "monthly", "female")
            limit: Maximum number of results to return

        Returns:
            List of matches with id, name, and score
        """
        codelist_type = codelist_type.upper()

        # Check for multi-value query (comma-separated)
        if "," in query:
            parts = [p.strip() for p in query.split(",") if p.strip()]
            all_results = []
            seen_ids = set()

            for part in parts:
                part_lower = self._normalize_query(part)
                # Recurse for single value
                matches = await self.find_value(
                    codelist_type, part, limit=1
                )  # find top match for each
                for m in matches:
                    if m["id"] not in seen_ids:
                        all_results.append(m)
                        seen_ids.add(m["id"])
            return all_results

        # Explicitly normalize using helper
        query_lower = self._normalize_query(query)

        # Handle global codelists (API-based)
        if codelist_type in self.GLOBAL_CODELISTS:
            await self._ensure_loaded(codelist_type)
            return self._search_global(codelist_type, query_lower, limit)

        # Handle static codelists
        if codelist_type in self.STATIC_MAPPINGS:
            return self._search_static(codelist_type, query_lower, limit)

        # Unknown codelist
        _logger.warning(f"Unknown codelist type: {codelist_type}")
        return []

    def _search_global(
        self, codelist_type: str, query_lower: str, limit: int
    ) -> list[dict[str, Any]]:
        """Search in a global (API-fetched) codelist.

        For REF_AREA, results are enriched with group metadata (is_group,
        group_type, member_count) sourced from the GroupHierarchyManager.
        """
        items = self._cache.get(codelist_type, [])
        results: list[dict[str, Any]] = []

        # Also search without spaces for cases like "Vietnam" vs "Viet Nam"
        query_no_spaces = query_lower.replace(" ", "")

        for item in items:
            item_id = item.get("Id", "")
            item_name = item.get("Name", "")
            name_lower = item_name.lower()
            name_no_spaces = name_lower.replace(" ", "")

            score = 0

            # Exact ID match
            if query_lower == item_id.lower():
                score = 100
            # Exact name match
            elif query_lower == name_lower:
                score = 100
            # Exact match ignoring spaces (Vietnam == Viet Nam)
            elif query_no_spaces == name_no_spaces:
                score = 100
            # ID starts with query
            elif item_id.lower().startswith(query_lower):
                score = 95
            # Name contains query exactly
            elif query_lower in name_lower:
                score = 90
            # No-space match (vietnam in vietnam)
            elif query_no_spaces in name_no_spaces:
                score = 90
            # Query contains name
            elif name_lower in query_lower:
                score = 85
            else:
                # Fuzzy: prefix matching
                similarity = self._calculate_similarity(query_lower, name_lower)
                if similarity > 0.7:
                    score = int(similarity * 100)

            if score > 0:
                results.append(
                    {
                        "id": item_id,
                        "name": item_name,
                        "score": score,
                    }
                )

        results.sort(key=lambda x: (-x["score"], x["name"]))
        matches = results[:limit]

        # Enrich REF_AREA results with group metadata.
        if codelist_type == "REF_AREA":
            ghm = get_group_hierarchy_manager()
            for match in matches:
                info = ghm.get_group_info(match["id"])
                if info:
                    match["is_group"] = True
                    match["group_type"] = info["type"]
                    match["member_count"] = len(info["countries"])
                    match["note"] = _make_group_note(match["id"], info)
                else:
                    match["is_group"] = False

        return matches

    def _search_static(
        self, codelist_type: str, query_lower: str, limit: int
    ) -> list[dict[str, Any]]:
        """Search in a static (hardcoded) codelist."""
        mapping = self.STATIC_MAPPINGS.get(codelist_type, {})
        results: list[dict[str, Any]] = []

        for name, code in mapping.items():
            name_lower = name.lower()
            score = 0

            # Exact match
            if query_lower == name_lower:
                score = 100
            # Query is the code itself
            elif query_lower == code.lower():
                score = 100
            # Name contains query
            elif query_lower in name_lower:
                score = 90
            # Query contains name
            elif name_lower in query_lower:
                score = 80

            if score > 0:
                results.append(
                    {
                        "id": code,
                        "name": name.capitalize(),
                        "score": score,
                    }
                )

        # Deduplicate by id (keep highest score)
        seen: dict[str, dict[str, Any]] = {}
        for r in results:
            rid = r["id"]
            if rid not in seen or r["score"] > seen[rid]["score"]:
                seen[rid] = r

        deduped = list(seen.values())
        deduped.sort(key=lambda x: (-x["score"], x["name"]))
        return deduped[:limit]

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """Calculate similarity ratio between two strings."""
        if not s1 or not s2:
            return 0.0

        if len(s1) <= 3:
            if s1 in s2 or s2.startswith(s1):
                return 0.8
            return 0.0

        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max(len1, len2) * 0.5:
            return 0.0

        s1_chars = set(s1)
        s2_chars = set(s2)
        common = len(s1_chars & s2_chars)
        total = len(s1_chars | s2_chars)
        char_similarity = common / total if total > 0 else 0

        prefix_len = 0
        for c1, c2 in zip(s1, s2):
            if c1 == c2:
                prefix_len += 1
            else:
                break
        prefix_ratio = prefix_len / min(len1, len2)

        return (char_similarity * 0.4) + (prefix_ratio * 0.6)

    async def get_codelist_mapping(self, codelist_type: str) -> dict[str, str]:
        """Get a dictionary mapping codes to names (e.g., {'KEN': 'Kenya'})."""
        codelist_type = codelist_type.upper()

        # Try from extdataportal first
        try:
            await self._ensure_extdataportal_loaded()
            ext_key = self._resolve_extdataportal_key(codelist_type)
            if ext_key in self._extdataportal and self._extdataportal[ext_key]:
                return dict(self._extdataportal[ext_key])
        except Exception:
            _logger.debug("Failed to load from extdataportal.", exc_info=True)

        # Fallback: derive global codelist from extdataportal via _ensure_loaded cache path
        if codelist_type in self.GLOBAL_CODELISTS:
            try:
                await self._ensure_loaded(codelist_type)
                items = self._cache.get(codelist_type, [])
                return {item.get("Id", ""): item.get("Name", "") for item in items}
            except Exception:
                _logger.warning(
                    "Global codelist load failed for %s.", codelist_type, exc_info=True
                )

        # Static mappings (reverse the value->code mapping to code->name)
        if codelist_type in self.STATIC_MAPPINGS:
            # STATIC_MAPPINGS is Name -> Code. We want Code -> Name.
            # Names in static mapping are lower case keys, we should capitalize for display.
            mapping = {}
            for name, code in self.STATIC_MAPPINGS[codelist_type].items():
                if (
                    code not in mapping
                ):  # First win or preferred name logic could be added
                    mapping[code] = name.capitalize()
            return mapping

        return {}


# Global instance
_codelist_manager: CodelistManager | None = None


def get_codelist_manager() -> CodelistManager:
    """Get the global CodelistManager instance."""
    global _codelist_manager
    if _codelist_manager is None:
        _codelist_manager = CodelistManager()
    return _codelist_manager


async def find_codelist_value(
    codelist_type: str, query: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Find values in a Data360 codelist by name (e.g. country or dimension labels).

    Use when you need to convert a user-friendly name to an API code. Helpful before
    data360_search_indicators (required_country) or when building disaggregation_filters
    for data360_get_data / data360_get_viz_spec. Not required if 3-letter codes are already known.

    For REF_AREA queries, results include group metadata:
      - is_group (bool): whether the code is a country group or an individual country
      - group_type (str): REGION, INCOME, LENDING, OTHER, REGION_UN, or CONTINENT
      - member_count (int): number of countries in the group
      - note (str): guidance on using data360_expand_country_group

    Workflow for country groups (e.g. "South Asian countries", "low income countries"):
      1. Call this function with codelist_type="REF_AREA" and the group name as query.
      2. If the result has is_group=True, you have the group code (e.g. "SAS", "LIC").
      3. To get individual country-level codes, call data360_expand_country_group with
         that code. Use the returned country_codes string directly in data360_get_data
         or data360_search_indicators disaggregation_filters.
      4. To use the group as a regional aggregate instead, pass the group code directly
         (e.g. REF_AREA="SAS") without expanding.

    Args:
        codelist_type: One of REF_AREA (countries/regions), FREQ, SEX, AGE, URBANISATION, UNIT_MEASURE.
        query: Search term (e.g. "Kenya", "female", "annual"). Comma-separated for multiple
            (e.g. "Kenya, Tanzania") returns one match per part. For REF_AREA, natural-language
            group phrases are also recognized (e.g. "South Asian countries", "low income countries").
        limit: Maximum number of matches to return (default 5).

    Returns:
        List of dicts, each with: id (code, e.g. "KEN"), name (e.g. "Kenya"), score (relevance 0–100).
        For REF_AREA, also includes is_group, group_type, member_count, note when applicable.
        Sorted by score descending. Empty list if no matches or unknown codelist_type.
    """
    # Check alias map for REF_AREA before fuzzy search.
    if codelist_type.upper() == "REF_AREA":
        alias_key = query.lower().strip()
        if alias_key in _GROUP_ALIASES:
            group_code = _GROUP_ALIASES[alias_key]
            ghm = get_group_hierarchy_manager()
            info = ghm.get_group_info(group_code)
            if info:
                result = {
                    "id": group_code,
                    "name": info["name"],
                    "score": 100,
                    "is_group": True,
                    "group_type": info["type"],
                    "member_count": len(info["countries"]),
                    "note": _make_group_note(group_code, info),
                }
                return [result]

    manager = get_codelist_manager()
    return await manager.find_value(codelist_type, query, limit)


async def get_codelist_mapping(codelist_type: str) -> dict[str, str]:
    """Get mapping of codes to names for a codelist."""
    manager = get_codelist_manager()
    return await manager.get_codelist_mapping(codelist_type)


# Convenience functions for common codelists
async def find_reference_area(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find reference areas (countries/regions) matching the query."""
    return await find_codelist_value("REF_AREA", query, limit)


async def find_frequency(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find frequency codes matching the query."""
    return await find_codelist_value("FREQ", query, limit)


async def find_sex(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find sex/gender codes matching the query."""
    return await find_codelist_value("SEX", query, limit)


async def find_age_group(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Find age group codes matching the query."""
    return await find_codelist_value("AGE", query, limit)


async def expand_country_group(
    group_code: str,
) -> dict[str, Any]:
    """Expand a REF_AREA group code into its constituent country codes.

    Use this when a user asks about a region, income group, or lending category
    and you need individual country-level data rather than the aggregate.

    Workflow for group discovery and expansion:
      1. If you only have a natural-language name (e.g. "South Asian countries"),
         call data360_find_codelist_value(codelist_type="REF_AREA", query="<name>")
         first to resolve it to a group code (e.g. "SAS").
      2. Pass that code to this function to get the full country list.
      3. Use the returned country_codes string as disaggregation_filters['REF_AREA']
         (already comma-separated ISO codes), or pass the same codes via get_data's
         country_code using semicolons between codes (e.g. 'KEN;MAR').

    Decision guidance — check the returned `count` after calling this function:
    - count <= 20: proceed with country-level data retrieval directly.
    - count > 20: inform the user before fetching all countries. Say:
      "This group contains N countries. Do you want individual country-level
      data for all of them, or would you prefer the regional aggregate?"
      Wait for their answer before proceeding.

    Covers groups sourced from FMR H_REF_AREA_GROUPS v38.0:
      REGION    - WB regional classifications (SAS, SSF, EAS, ECS, LCN, ...)
      INCOME    - LIC, LMC, UMC, HIC, MIC, LMY, MIX
      LENDING   - IDA, IBRD, blend classifications
      OTHER     - FCS, LDC, SST, OED, EUU, ...
      REGION_UN - UN M49 statistical divisions
      CONTINENT - Continental groupings

    Args:
        group_code: REF_AREA group code (e.g. "SAS", "LIC", "EAS", "FCS").
            Case-insensitive. Natural-language names (e.g. "south asian countries")
            are also accepted as a convenience; they are resolved via the alias map.
            For reliable discovery, use data360_find_codelist_value first.

    Returns:
        Dict with:
          - group_code (str): uppercased code
          - group_name (str): human-readable name
          - group_type (str): REGION | INCOME | LENDING | OTHER | REGION_UN | CONTINENT
          - countries (list[dict]): [{code, name}] for each member country
          - country_codes (str): comma-separated codes for direct use in API calls
          - count (int): number of member countries
          - hierarchy_version (str): FMR hierarchy version used
        On failure:
          - error (str): description of what went wrong
    """
    ghm = get_group_hierarchy_manager()
    code_upper = group_code.strip().upper()
    info = ghm.get_group_info(code_upper)

    if not info:
        # Try alias resolution as a fallback
        alias_key = group_code.lower().strip()
        if alias_key in _GROUP_ALIASES:
            code_upper = _GROUP_ALIASES[alias_key]
            info = ghm.get_group_info(code_upper)

    if not info:
        return {
            "error": (
                f"Unknown group code: '{group_code}'. "
                "Use data360_find_codelist_value('REF_AREA', '<name>') to find "
                "the correct code, or check that the group exists in the FMR hierarchy."
            )
        }

    # Build country list with names from the Data360 codelist (best-effort).
    # Codes are always available; names may be missing for some entries.
    cl_manager = get_codelist_manager()
    country_name_map: dict[str, str] = {}
    try:
        country_name_map = await cl_manager.get_codelist_mapping("REF_AREA")
    except Exception as e:
        # Names are a convenience; codes are always returned even without them.
        _logger.warning(
            "expand_country_group: failed to fetch REF_AREA name mapping for '%s': %s. "
            "Country names will fall back to code strings.",
            code_upper,
            e,
        )

    countries = [
        {"code": c, "name": country_name_map.get(c, c)} for c in info["countries"]
    ]

    meta = ghm.get_meta()

    return {
        "group_code": code_upper,
        "group_name": info["name"],
        "group_type": info["type"],
        "countries": countries,
        "country_codes": ",".join(c["code"] for c in countries),
        "count": len(countries),
        "hierarchy_version": meta.get("hierarchy_version", "unknown"),
    }
