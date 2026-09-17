import os
from typing import Any
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))

# Autoescape must cover the template extensions this package actually uses:
# `select_autoescape(["html", "xml"])` leaves *.jinja2 unescaped, so every
# interpolation below would be rendered raw (CWE-80). `default=True` keeps any
# future extension escaped unless it is explicitly listed as safe.
_AUTOESCAPE = select_autoescape(
    enabled_extensions=("html", "htm", "xml", "jinja2", "jinja"),
    default=True,
)

try:
    _env = Environment(
        loader=PackageLoader("data360", "templates"),
        autoescape=_AUTOESCAPE,
    )
except Exception:
    _env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=_AUTOESCAPE,
    )


def render_template(template_name: str, **context: Any) -> str:
    """Render a Jinja2 template file from src/data360/templates/."""
    template = _env.get_template(template_name)
    return template.render(**context)
