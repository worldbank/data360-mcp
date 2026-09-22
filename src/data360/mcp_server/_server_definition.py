from fastmcp import FastMCP, settings

# FastMCP 3.4.3 turned on Host/Origin validation by default with empty allow-lists, which
# 421s every non-loopback Host (public domain, pod IP, proxy service name) — traffic that
# FastMCP 2.x accepted. Widen the lists to the old behaviour; the control stays installed so
# FASTMCP_HTTP_ALLOWED_HOSTS / FASTMCP_HTTP_ALLOWED_ORIGINS can narrow it again per deploy.
# Set here rather than in data360.server so every entry point gets it, including
# `python -m data360.mcp_server`, which never imports that module.
settings.http_allowed_hosts = settings.http_allowed_hosts or ["*"]
settings.http_allowed_origins = settings.http_allowed_origins or ["*"]

# NOTE: base definition to allow for mounting of resources, prompts, tools, independently
mcp = FastMCP(
    "Data360 MCP Server",
    version="0.1.0",
)
