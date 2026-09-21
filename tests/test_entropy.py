"""Tests for the CWE-331 (insufficient entropy) remediation.

Veracode flags the ``random`` module's Mersenne Twister wherever it appears, even
for backoff jitter. These tests pin the two properties that matter for that
policy: jitter values come from a source that a seeded Mersenne Twister cannot
reproduce, and every reported call site goes through it rather than ``random``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from data360.entropy import uniform_jitter

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTED_FILES = (
    "src/data360/entropy.py",
    "src/data360/http_client.py",
    "src/data360/providers.py",
    "tests/test_rate_limits.py",
)


class TestUniformJitter:
    def test_values_stay_inside_the_band(self):
        for _ in range(500):
            assert 2.0 <= uniform_jitter(2.0, 6.0) < 6.0

    def test_negative_bands_cross_zero(self):
        for _ in range(200):
            assert -1.5 <= uniform_jitter(-1.5, -0.5) < -0.5

    def test_degenerate_band_returns_low(self):
        assert uniform_jitter(3.0, 3.0) == 3.0
        assert uniform_jitter(3.0, 1.0) == 3.0

    def test_covers_the_band_instead_of_returning_a_constant(self):
        draws = [uniform_jitter(0.0, 1.0) for _ in range(200)]

        assert len(set(draws)) > 1
        assert min(draws) < 0.25
        assert max(draws) > 0.75

    def test_values_are_not_reproducible_across_calls(self):
        """The CWE-331 contract: the values are not predictable from a seed."""
        first = [uniform_jitter(0.0, 1.0) for _ in range(5)]
        second = [uniform_jitter(0.0, 1.0) for _ in range(5)]

        assert first != second


@pytest.mark.parametrize("relative_path", REPORTED_FILES)
def test_reported_modules_do_not_use_the_insecure_random_module(relative_path):
    """CWE-331 policy: the reported files must draw randomness from ``secrets``.

    This is deliberately a source check rather than a behaviour check — the flaw
    is "which generator is used", and a reintroduced ``random`` call would only
    be visible to the scanner, not to any runtime assertion.
    """
    tree = ast.parse((REPO_ROOT / relative_path).read_text())

    insecure: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            insecure += [a.name for a in node.names if a.name == "random"]
        elif isinstance(node, ast.ImportFrom) and node.module == "random":
            insecure.append(f"from random import {', '.join(a.name for a in node.names)}")
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "random"
        ):
            insecure.append(f"random.{node.attr}")

    assert not insecure, f"{relative_path} uses the Mersenne Twister: {insecure}"
