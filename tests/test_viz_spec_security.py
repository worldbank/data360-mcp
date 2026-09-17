"""Regression tests for CWE-73 (path traversal) and CWE-918 (SSRF) hardening."""

import importlib.util
import re
from pathlib import Path

import pytest
from fastapi import status
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data360.server import VizSpecRequest, app
from data360.visualization import save_specs_to_static

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def _load_build_script():
    spec = importlib.util.spec_from_file_location(
        "build_ref_area_groups", SCRIPTS_DIR / "build_ref_area_groups.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestVizSpecRequestValidation:
    def _valid(self, **overrides):
        payload = {
            "database_id": "WB_WDI",
            "indicator_id": "WB_WDI_NY_GDP_PCAP_KD",
            "country_code": "BRA;ARG;MEX;COL;CHL",
            "start_year": 2018,
            "end_year": 2022,
        }
        payload.update(overrides)
        return VizSpecRequest(**payload)

    def test_dashboard_defaults_accepted(self):
        req = self._valid()
        assert req.indicator_id == "WB_WDI_NY_GDP_PCAP_KD"

    @pytest.mark.parametrize(
        "field,value",
        [
            ("indicator_id", "../../etc/passwd"),
            ("indicator_id", "<script>alert(1)</script>"),
            ("indicator_id", "NY_GDP|PCAP"),
            ("database_id", "..\\windows\\system32"),
            ("country_code", "BRA;../../etc"),
            ("chart_type", "<img src=x onerror=1>"),
        ],
    )
    def test_traversal_and_markup_rejected(self, field, value):
        with pytest.raises(ValidationError):
            self._valid(**{field: value})

    def test_year_bounds_enforced(self):
        with pytest.raises(ValidationError):
            self._valid(start_year=1800)
        with pytest.raises(ValidationError):
            self._valid(end_year=2200)

    def test_viz_spec_endpoint_returns_422_for_traversal(self):
        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/viz-spec",
            json={"database_id": "WB_WDI", "indicator_id": "../../etc/passwd"},
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY


class TestSaveSpecsConfinement:
    def test_writes_uuid_file_inside_viz_specs(self):
        url = save_specs_to_static({"mark": "bar"})
        match = re.search(r"/static/viz_specs/([0-9a-f-]{36}_vega\.json)$", url)
        assert match is not None
        project_root = Path(__file__).parent.parent
        written = project_root / "static" / "viz_specs" / match.group(1)
        assert written.is_file()
        written.unlink()


class TestFmrFetchValidation:
    @pytest.fixture(scope="class")
    def mod(self):
        return _load_build_script()

    @pytest.mark.parametrize("version", ["38.0", "2.0", "38", ""])
    def test_valid_versions_accepted(self, mod, version):
        if version == "":
            return
        assert mod.validate_fmr_version(version, "test") == version

    @pytest.mark.parametrize(
        "version",
        ["../../etc/passwd", "38.0;evil", "$(id)", "38..0", "latest", "v38", "38,0"],
    )
    def test_malicious_versions_rejected(self, mod, version):
        with pytest.raises(ValueError):
            mod.validate_fmr_version(version, "test")

    def test_only_fmr_host_allowed(self, mod):
        mod.validate_fmr_url("https://fmr.worldbank.org/FMR/sdmx/v2/x?format=sdmx-json")
        with pytest.raises(ValueError):
            mod.validate_fmr_url("https://evil.example.com/x")
        with pytest.raises(ValueError):
            mod.validate_fmr_url("http://fmr.worldbank.org/x")
