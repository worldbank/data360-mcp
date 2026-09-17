"""Transform FMR hierarchy + codelist into a compact lookup for runtime use.

Reads the official FMR SDMX files committed under examples/ and writes a
purpose-built JSON to src/data360/ref_area_groups.json that is shipped
as package data. The output contains all 147 group codes with their country
memberships, drawn from the H_REF_AREA_GROUPS hierarchy.

Parsing logic lives in GroupHierarchyManager (providers.py) so it is shared
between this build-time script and the background sync that runs at runtime.

Usage:
    uv run python scripts/build_ref_area_groups.py

To fetch the latest (or specific version) directly from FMR (requires VPN):
    uv run python scripts/build_ref_area_groups.py --fetch
    uv run python scripts/build_ref_area_groups.py --fetch --hierarchy-version 38.0 --codelist-version 2.0

Re-run this script whenever:
- A new FMR hierarchy version is available (e.g. annual income reclassification)
- The examples/ source files are updated

The hierarchy is versioned (omitting version returns latest from FMR):
  https://fmr.worldbank.org/FMR/sdmx/v2/structure/hierarchy/WB/H_REF_AREA_GROUPS/
The codelist:
  https://fmr.worldbank.org/FMR/sdmx/v2/structure/codelist/WB/CL_REF_GROUPINGS/

Group types included in output (all types with country memberships):
  REGION     - WB regional classifications (SAS, SSF, EAS, ...)
  INCOME     - Income groups (LIC, HIC, LMC, UMC, ...)
  LENDING    - IDA/IBRD/blend classifications
  OTHER      - FCS, LDC, small states, OECD, EU, ...
  REGION_UN  - UN M49 statistical divisions
  CONTINENT  - Numeric continent codes

Excluded:
  WLD, _T    - Leaf-only entries (individual countries, not groups)
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

# Allow importing from the package when run directly via `uv run python scripts/...`
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from data360.providers import GroupHierarchyManager  # noqa: E402

REPO_ROOT = _REPO_ROOT
HIERARCHY_FILE = REPO_ROOT / "examples" / "H_AREA_GROUPS38.json"
CODELIST_FILE = REPO_ROOT / "examples" / "CL_REF_GROUPINGS.json"
OUTPUT_FILE = REPO_ROOT / "src" / "data360" / "ref_area_groups.json"

# CWE-73/918: FMR versions come from CLI args and are concatenated into the
# request URL, so they must match a strict positive pattern. The host is
# pinned — only the official FMR endpoint is ever fetched.
_FMR_VERSION_RE = re.compile(r"^\d+(\.\d+)?$")
_FMR_ALLOWED_HOST = "fmr.worldbank.org"


def validate_fmr_version(value: str, name: str) -> str:
    """Return ``value`` if it is a plain numeric FMR version, else raise."""
    if not _FMR_VERSION_RE.match(value or ""):
        raise ValueError(
            f"Invalid {name} {value!r}: expected a numeric version like '38.0'."
        )
    return value


def validate_fmr_url(url: str) -> str:
    """Return ``url`` if it targets the official FMR host over HTTPS, else raise."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != _FMR_ALLOWED_HOST:
        raise ValueError(f"Refusing to fetch non-FMR URL: {url!r}.")
    return url


def fetch_fmr_data(url: str, output_path: Path) -> None:
    """Download JSON from FMR and save it to output_path."""
    validate_fmr_url(url)
    print(f"Fetching from {url} ...")
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {output_path}")
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        print("  Note: FMR requires VPN access. If you are not on the VPN, this will fail.")
        raise


def build(hierarchy_path: Path, codelist_path: Path, output_path: Path) -> None:
    """Build and write the ref_area_groups.json output file."""
    print(f"Reading hierarchy: {hierarchy_path}")
    print(f"Reading codelist:  {codelist_path}")

    with open(hierarchy_path, encoding="utf-8") as f:
        hierarchy_data = json.load(f)
    with open(codelist_path, encoding="utf-8") as f:
        codelist_data = json.load(f)

    name_map = GroupHierarchyManager.parse_name_map(codelist_data)
    groups, all_countries, version = GroupHierarchyManager.parse_hierarchy(
        hierarchy_data, name_map
    )

    group_types = sorted({v["type"] for v in groups.values()})

    output = {
        "_meta": {
            "source": f"FMR H_REF_AREA_GROUPS v{version} + CL_REF_GROUPINGS",
            "built_at": datetime.now(UTC).strftime("%Y-%m-%d"),
            "hierarchy_version": version,
            "total_groups": len(groups),
            "total_countries": len(all_countries),
            "group_types": group_types,
            "note": (
                "Re-generate with: uv run python scripts/build_ref_area_groups.py"
            ),
        },
        "groups": groups,
        "all_countries": sorted(all_countries),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_path}")
    print(f"  Groups: {len(groups)}")
    print(f"  Countries: {len(all_countries)}")
    print(f"  Group types: {group_types}")
    size_kb = output_path.stat().st_size / 1024
    print(f"  File size: {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(
        description="Transform FMR hierarchy + codelist into a compact lookup for runtime use."
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=(
            "Fetch the latest (or specified version) JSON files from FMR "
            "(VPN required) and overwrite local examples."
        ),
    )
    parser.add_argument(
        "--hierarchy-version",
        type=str,
        default="",
        help="FMR hierarchy version to fetch (e.g. 38.0). If empty, fetches latest.",
    )
    parser.add_argument(
        "--codelist-version",
        type=str,
        default="",
        help="FMR codelist version to fetch (e.g. 2.0). If empty, fetches latest.",
    )

    args = parser.parse_args()

    if args.fetch:
        hierarchy_url = (
            "https://fmr.worldbank.org/FMR/sdmx/v2/structure/hierarchy/WB/H_REF_AREA_GROUPS/"
        )
        if args.hierarchy_version:
            hierarchy_url += validate_fmr_version(
                args.hierarchy_version, "hierarchy version"
            )
        hierarchy_url += "?format=sdmx-json"

        codelist_url = (
            "https://fmr.worldbank.org/FMR/sdmx/v2/structure/codelist/WB/CL_REF_GROUPINGS/"
        )
        if args.codelist_version:
            codelist_url += validate_fmr_version(
                args.codelist_version, "codelist version"
            )
        codelist_url += "?format=sdmx-json"

        print("--- Fetching source data from FMR ---")
        fetch_fmr_data(hierarchy_url, HIERARCHY_FILE)
        fetch_fmr_data(codelist_url, CODELIST_FILE)
        print("-------------------------------------\n")

    build(HIERARCHY_FILE, CODELIST_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()
