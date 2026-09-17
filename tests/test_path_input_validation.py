"""Tests for the CWE-73 remediation (external control of file name or path).

Veracode reports CWE-73 on ``country_code.replace(";", ",")`` (blacklist-style
normalisation of a request value), on ``Series.replace(series_labels)`` and on
``urlopen`` with a URL built from a CLI argument. None of those lines is a
filesystem call in this revision — the only file write, ``save_specs_to_static``,
names its file from ``uuid.uuid4()`` — but the policy's remediation applies all
the same: validate untrusted input against an expected format rather than
stripping characters out of it. These tests pin that validation.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from data360.api import get_data, get_data_api_url
from data360.validation import validated_country_codes
from data360.visualization import (
    _detect_missing_countries,
    _validated_series_labels,
    get_viz_spec,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

# A real FMR URL: the only destination build_ref_area_groups may fetch.
FMR_URL = (
    "https://fmr.worldbank.org/FMR/sdmx/v2/structure/hierarchy/"
    "WB/H_REF_AREA_GROUPS/38.0?format=sdmx-json"
)

# A label that is not a string: the CWE-73 validation must reject it outright.
INVALID_LABEL = 42


class TestValidatedCountryCodes:
    def test_splits_on_either_delimiter_and_normalises_case(self):
        assert validated_country_codes(" usa;  ind; pak ") == ["USA", "IND", "PAK"]
        assert validated_country_codes("USA,IND") == ["USA", "IND"]
        assert validated_country_codes("USA, ind;PAK") == ["USA", "IND", "PAK"]

    def test_traversal_and_markup_tokens_are_dropped(self):
        assert validated_country_codes("USA;../../etc/passwd") == ["USA"]
        assert validated_country_codes("../../etc/passwd") == []
        assert validated_country_codes("<script>alert(1)</script>") == []
        assert validated_country_codes("USA;<img src=x>") == ["USA"]

    def test_empty_input(self):
        assert validated_country_codes(None) == []
        assert validated_country_codes("") == []
        assert validated_country_codes(" ; , ") == []


class TestDetectMissingCountries:
    async def test_only_well_formed_codes_are_reported(self):
        """A traversal-looking token must not reach the caller's message."""
        with patch(
            "data360.providers.get_codelist_mapping",
            new_callable=AsyncMock,
            return_value={},
        ):
            missing = await _detect_missing_countries(
                "USA;../../etc/passwd", {"USA"}
            )

        assert missing == []


class TestCountryCodeBoundary:
    """The six request-parameter sites must never forward a malformed value."""

    @pytest.fixture
    def stub_metadata(self):
        """get_data_api_url consults metadata; keep the test offline."""
        stub = SimpleNamespace(indicator_metadata={}, disaggregation_options=[], error=None)
        with patch("data360.api.get_metadata", new_callable=AsyncMock, return_value=stub):
            yield

    async def test_url_builder_keeps_only_valid_codes(self, stub_metadata):
        url = await get_data_api_url(
            database_id="WB_WDI",
            indicator_id="WB_WDI_NY_GDP_MKTP_KD_ZG",
            country_code="USA;../../etc/passwd",
        )

        assert "REF_AREA=USA" in url
        assert "passwd" not in url
        assert ".." not in url

    async def test_url_builder_rejects_input_with_no_valid_code(self, stub_metadata):
        with pytest.raises(ValueError, match="ISO-style country codes"):
            await get_data_api_url(
                database_id="WB_WDI",
                indicator_id="WB_WDI_NY_GDP_MKTP_KD_ZG",
                country_code="../../etc/passwd",
            )

    def test_ref_area_filter_is_validated_like_country_code(self):
        """The filter path must not smuggle a malformed value past the argument check."""
        from data360.api import _validate_user_filters

        valid, errors = _validate_user_filters(
            {"REF_AREA": "USA;../../etc/passwd", "SEX": "F"}, {}
        )

        assert valid.get("REF_AREA") == "USA"
        assert valid.get("SEX") == "F"
        assert errors == []

    def test_ref_area_filter_with_no_usable_code_is_reported(self):
        from data360.api import _validate_user_filters

        valid, errors = _validate_user_filters({"REF_AREA": "../../etc/passwd"}, {})

        assert "REF_AREA" not in valid
        assert any("REF_AREA" in message for message in errors)

    async def test_multi_indicator_path_validates_series_labels(self, monkeypatch):
        """The multi-indicator pipeline reads series_labels too, so it must validate."""
        from data360 import visualization

        seen: dict = {}
        real = visualization._validated_series_labels

        def spy(labels):
            seen["labels"] = labels
            return real(labels)

        monkeypatch.setattr(visualization, "_validated_series_labels", spy)

        result = await visualization.get_multi_indicator_viz_spec(
            indicator_ids=[{"database_id": "WB_WDI", "indicator_id": "X"}],
            series_labels={"X": 42},
        )

        assert seen.get("labels") == {"X": 42}, "series_labels was not validated"
        assert result.get("error"), "one indicator should be rejected before any fetch"

    async def test_rank_countries_fails_closed_when_no_code_is_usable(self, httpx_mock):
        """An all-malformed scope must not silently rank the global dataset."""
        from data360.api import rank_countries

        result = await rank_countries(
            database_id="WB_WDI",
            indicator_id="WB_WDI_NY_GDP_MKTP_KD_ZG",
            country_codes="../../etc/passwd",
        )

        assert result.error is not None
        assert "ISO-style country codes" in result.error
        assert httpx_mock.get_requests() == [], "an upstream request was attempted"

    async def test_data_fetch_fails_closed_without_calling_upstream(self, httpx_mock):
        result = await get_data(
            database_id="WB_WDI",
            indicator_id="WB_WDI_NY_GDP_MKTP_KD_ZG",
            country_code="../../etc/passwd",
        )

        assert result.error is not None
        assert "ISO-style country codes" in result.error
        assert httpx_mock.get_requests() == []


class TestSeriesLabelValidation:
    def test_keeps_string_labels_and_strips_padding(self):
        assert _validated_series_labels({" WGI_EST ": " Estimate "}) == {
            "WGI_EST": "Estimate"
        }

    def test_drops_non_strings_empties_and_overlong_entries(self):
        assert (
            _validated_series_labels(
                {
                    "SEX": 42,
                    "": "no key",
                    "AGE": "",
                    "K" * 65: "key too long",
                    "URB": "v" * 201,
                    "F": "Female",
                }
            )
            == {"F": "Female"}
        )

    def test_non_mapping_input_is_ignored(self):
        assert _validated_series_labels(None) == {}
        assert _validated_series_labels(["F", "Female"]) == {}


def _values_under_key(spec: dict, key: str) -> list:
    """Collect every scalar stored under *key* in a (possibly nested) spec."""
    found: list = []

    def walk(node) -> None:
        if isinstance(node, dict):
            for node_key, value in node.items():
                if node_key == key:
                    found.extend(value if isinstance(value, list) else [value])
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(spec)
    return found


class TestLabelsReachTheChart:
    """The validated labels are what actually gets applied to the chart data."""

    async def test_valid_label_applied_and_invalid_entry_has_no_effect(self):
        frame = pd.DataFrame(
            {
                "TIME_PERIOD": ["2020-01-01", "2021-01-01", "2022-01-01"],
                "OBS_VALUE": [100, 200, 300],
                "REF_AREA": ["KEN", "KEN", "KEN"],
                "SEX": ["F", "M", "F"],
            }
        )

        with (
            patch(
                "data360.api.get_data_api_url",
                new_callable=AsyncMock,
                return_value="http://fake-api/data?DATABASE_ID=WB_WDI&INDICATOR=FAKE",
            ),
            patch(
                "data360.visualization._fetch_data_internal",
                new_callable=AsyncMock,
                return_value=frame,
            ),
            patch(
                "data360.api.get_metadata",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "data360.visualization.save_specs_to_static",
                return_value="http://localhost:8021/static/viz_specs/test.json",
            ),
            patch(
                "data360.visualization.get_database_mapping",
                new_callable=AsyncMock,
                return_value={"WB_WDI": "World Development Indicators"},
            ),
            patch(
                "data360.providers.get_codelist_mapping",
                new_callable=AsyncMock,
                return_value={},
            ),
        ):
            result = await get_viz_spec(
                database_id="WB_WDI",
                indicator_id="FAKE_IND",
                # "M": 42 is not a usable label, so it must be ignored entirely.
                series_labels={"F": "Female", "M": INVALID_LABEL},
            )

        assert result["error"] is None, result
        spec = result["spec"]
        sex_values = _values_under_key(spec, "sex")

        assert "Female" in sex_values, json.dumps(spec)[:500]
        assert INVALID_LABEL not in sex_values
        assert "M" in sex_values  # untouched code, because its label was rejected


@pytest.fixture(scope="module")
def fmr_script():
    """Load scripts/build_ref_area_groups.py without putting scripts/ on sys.path."""
    module_spec = importlib.util.spec_from_file_location(
        "build_ref_area_groups", REPO_ROOT / "scripts" / "build_ref_area_groups.py"
    )
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class TestFmrVersionArgument:
    def test_accepts_version_like_values(self, fmr_script):
        assert fmr_script.validated_version("38.0", "--hierarchy-version") == "38.0"
        assert fmr_script.validated_version(" 38 ", "--hierarchy-version") == "38"
        assert fmr_script.validated_version("1.2.3.4", "--codelist-version") == "1.2.3.4"
        assert fmr_script.validated_version("", "--hierarchy-version") == ""

    @pytest.mark.parametrize(
        "hostile",
        [
            "38.0/../../../etc/passwd",
            "../../..",
            "file:///etc/passwd",
            "https://evil.example.com/",
            "38.0?format=text",
            "1.2.3.4.5",
            "38 0",
        ],
    )
    def test_rejects_anything_that_is_not_a_version(self, fmr_script, hostile):
        with pytest.raises(SystemExit):
            fmr_script.validated_version(hostile, "--hierarchy-version")


class TestFmrScriptConfiguration:
    def test_repo_root_is_absolute_so_containment_works_from_the_cli(self, fmr_script):
        """`python scripts/build_ref_area_groups.py` leaves __file__ relative."""
        assert fmr_script.REPO_ROOT.is_absolute()

    def test_origin_defaults_to_fmr_and_is_configurable(self, fmr_script, monkeypatch):
        assert fmr_script.allowed_fmr_origin() == "https://fmr.worldbank.org"

        monkeypatch.setenv("FMR_ORIGIN", "https://fmr-mirror.internal")
        assert fmr_script.allowed_fmr_origin() == "https://fmr-mirror.internal"

        # the configured origin is what the guard accepts and refuses against
        assert fmr_script.validated_fmr_url("https://fmr-mirror.internal/x")
        with pytest.raises(ValueError, match="refusing to fetch outside"):
            fmr_script.validated_fmr_url("https://fmr.worldbank.org/x")


class TestFmrDestinationAllowList:
    """CWE-918: only the pinned FMR origin may be fetched."""

    @pytest.fixture
    def fmr_urls(self, fmr_script, monkeypatch):
        """Stub urlopen so the test proves which URLs are even attempted."""
        attempts: list[str] = []

        def _stub(request, *args, **kwargs):
            attempts.append(request.full_url)
            raise RuntimeError("stubbed network")

        monkeypatch.setattr(fmr_script.urllib.request, "urlopen", _stub)
        return attempts

    @pytest.mark.parametrize(
        "hostile",
        [
            "file:///etc/passwd",
            "file:///tmp/anything.json",
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "http://localhost:8021/mcp",
            "http://127.0.0.1:8021/mcp",
            "https://evil.example.com/FMR/sdmx/v2/structure/hierarchy/WB/H_REF_AREA_GROUPS/",
            "https://fmr.worldbank.org.evil.example.com/x",
        ],
    )
    def test_non_fmr_destinations_are_refused_without_a_request(
        self, fmr_script, fmr_urls, hostile
    ):
        with pytest.raises(ValueError, match="refusing to fetch outside"):
            fmr_script.fetch_fmr_data(hostile, REPO_ROOT / "examples" / "x.json")

        assert fmr_urls == [], f"a request was attempted for {hostile!r}"

    def test_the_fmr_origin_is_allowed(self, fmr_script, fmr_urls):
        with pytest.raises(RuntimeError, match="stubbed network"):
            fmr_script.fetch_fmr_data(FMR_URL, REPO_ROOT / "examples" / "x.json")

        assert fmr_urls == [FMR_URL]


class TestFmrOutputContainment:
    def test_refuses_to_write_outside_the_repository(self, fmr_script, tmp_path):
        outside = tmp_path.parent / "outside.json"

        with pytest.raises(ValueError, match="outside the repository"):
            fmr_script.fetch_fmr_data(FMR_URL, outside)

    def test_accepts_a_path_inside_the_repository(self, fmr_script, monkeypatch):
        inside = REPO_ROOT / "examples" / "ignored-by-this-test.json"

        # Stub the fetch so the test stays offline: it must get past the
        # containment check and fail on the download instead.
        def _boom(*args, **kwargs):
            raise RuntimeError("stubbed network")

        monkeypatch.setattr(fmr_script.urllib.request, "urlopen", _boom)

        with pytest.raises(RuntimeError, match="stubbed network"):
            fmr_script.fetch_fmr_data(FMR_URL, inside)
