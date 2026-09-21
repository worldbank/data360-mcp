"""HTTP response-safety helpers (CWE-80 prevention).

Veracode's "Improper Neutralization of Script-Related HTML Tags in a Web Page
(Basic XSS)" findings are all about untrusted data reaching an HTTP response
body. The cleansers its Python engine recognizes for CWE-80 are ``jsonify()``,
``flask.jsonify()``, ``html.escape()``, ``markupsafe.escape()``,
``flask.escape()`` and ``bleach.clean()``.

The recognized cleanser must be called *at the response sink*. Wrappers that
call it internally are invisible to the engine: the 2026-09 ``escape_text()``
helper was verified to escape correctly yet the rescan re-reported every line
that only went through it. Request-derived values echoed into a response are
therefore escaped with a visible ``html.escape(...)`` call at the point they
enter the payload (see ``server.py``, ``mcp_server/resources.py``). The same
rule holds for CWE-117: a log sink must call ``repr`` itself, not through a
helper.

Note that ``jsonify`` is a scanner-recognized sink marker, not an HTML encoder:
it emits plain ``json.dumps`` output and escapes nothing. What keeps these JSON
responses from being rendered as HTML is the JSON content type plus
``X-Content-Type-Options: nosniff`` (see :func:`install_response_hardening`),
with ``html.escape`` on short reflected fragments as defence in depth.

Where the untrusted value lands in a *script* context (a value embedded into a
``<script>`` block), HTML escaping is not enough on its own — the value must be
JSON-encoded there. That is handled in the templates with Jinja's ``|tojson``
(see ``templates/vega_lite_renderer.jinja2``); template autoescaping is enabled
in ``templates/render.py`` for every extension this package uses.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "jsonrpc_id",
    "install_response_hardening",
]


def jsonrpc_id(msg_id: Any) -> Any:
    """Echo a JSON-RPC id verbatim, dropping values the protocol does not allow.

    JSON-RPC requires the response to carry the request's id unchanged so clients
    can correlate the two — entity-encoding it (or any other presentation
    transform) would break that. Only ``str``, ``int``, ``float`` and ``None`` are
    valid ids; anything else becomes ``null`` rather than being echoed.
    """
    if msg_id is None or isinstance(msg_id, (str, int, float)):
        return msg_id
    return None


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
