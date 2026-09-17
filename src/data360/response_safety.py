"""HTTP response-safety helpers (CWE-80 prevention).

Veracode's "Improper Neutralization of Script-Related HTML Tags in a Web Page
(Basic XSS)" findings are all about untrusted data reaching an HTTP response
body. The cleansers its Python engine recognizes for CWE-80 are ``jsonify()``,
``flask.jsonify()``, ``html.escape()``, ``markupsafe.escape()``,
``flask.escape()`` and ``bleach.clean()`` — so request-derived values that are
echoed into a response go through :func:`html.escape` on the way in.

Where the untrusted value lands in a *script* context (a value embedded into a
``<script>`` block), HTML escaping is not enough on its own — the value must be
JSON-encoded there. That is handled in the templates with Jinja's ``|tojson``
(see ``templates/vega_lite_renderer.jinja2``); template autoescaping is enabled
in ``templates/render.py`` for every extension this package uses.
"""

from __future__ import annotations

import html
from typing import Any

__all__ = [
    "escape_jsonrpc_id",
    "escape_text",
    "install_response_hardening",
]


def escape_text(value: Any) -> str:
    """HTML-escape untrusted text before it is placed in a response body.

    ``html.escape(..., quote=True)`` escapes ``& < > " '``, covering element-body
    and attribute contexts.
    """
    return html.escape(str(value), quote=True)


def escape_jsonrpc_id(msg_id: Any) -> Any:
    """Echo a JSON-RPC id, HTML-escaping string ids.

    JSON-RPC ids are normally integers; non-string ids pass through unchanged so
    numeric round-tripping is unaffected.
    """
    return html.escape(msg_id, quote=True) if isinstance(msg_id, str) else msg_id


def install_response_hardening(app: Any) -> None:
    """Stop browsers rendering JSON responses as HTML.

    ``X-Content-Type-Options: nosniff`` removes the MIME-sniffing path that makes
    a reflected value in a JSON response exploitable. Defence in depth behind the
    escaping above.
    """

    @app.middleware("http")
    async def _nosniff_middleware(request: Any, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        return response
