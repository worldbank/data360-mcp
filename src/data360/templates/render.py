import os
from typing import Any

from jinja2 import Environment, FileSystemLoader, PackageLoader

TEMPLATES_DIR = os.path.dirname(os.path.abspath(__file__))

# All templates in this directory render HTML, so autoescape is unconditional.
# select_autoescape(["html", "xml"]) must not be used here: template filenames
# end in ".jinja2", which it does not match, silently leaving every {{ }}
# expression unescaped (CWE-80). The only intentional bypasses are the pinned
# first-party vendor JS bundles marked | safe at the use site.
try:
    _env = Environment(
        loader=PackageLoader("data360", "templates"),
        autoescape=True,
    )
except Exception:
    _env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
    )


def render_template(template_name: str, **context: Any) -> str:
    """Render a Jinja2 template file from src/data360/templates/."""
    template = _env.get_template(template_name)
    return template.render(**context)
