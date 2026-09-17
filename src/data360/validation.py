"""Request-input validation helpers (shared, dependency-free).

A leaf module on purpose: both :mod:`data360.api` and
:mod:`data360.visualization` need these checks, and neither should have to import
the other to get them.
"""

from __future__ import annotations

import logging
import re

__all__ = ["validated_country_codes"]

_logger = logging.getLogger(__name__)

# ISO-style country and World Bank aggregate codes: letters/digits, 1-4 of them.
# The character class is what carries the security property (no path separators,
# no dots, no markup); short-but-alphanumeric tokens are intentionally allowed
# through, because callers that validate against the country codelist report them
# as unusable rather than having them silently dropped here.
_COUNTRY_CODE_RE = re.compile(r"[A-Z0-9]{1,4}")


def validated_country_codes(country_code: str | None) -> list[str]:
    """Split a delimited country list into well-formed ISO-style codes (CWE-73).

    Positive validation: every token must match an ISO-style code. Malformed
    tokens are dropped with a warning instead of being forwarded upstream, where
    a path-traversal-looking value is rejected by the Data360 WAF (HTTP 403) and
    would otherwise reach the API query layer.
    """
    if not country_code:
        return []
    codes: list[str] = []
    for token in re.split(r"[;,]", country_code):
        code = token.strip().upper()
        if not code:
            continue
        if _COUNTRY_CODE_RE.fullmatch(code):
            codes.append(code)
        else:
            _logger.warning(
                "Ignoring malformed country code from request: %s", repr(code)
            )
    return codes
