"""Shared httpx clients: connection reuse, split timeouts, bounded retries.

Two clients, one per traffic profile, each with its own connection pool:

* **Interactive** (``get_shared_httpx_client``) — user-facing tool calls, health
  probes and chart writes. Tight timeouts so a stalled dependency cannot hold a
  request open.
* **Background/bulk** (``get_background_httpx_client``) — the dimension catalog
  (~1.7 MB), the dataset catalogue sync and the group-hierarchy fetch. Nobody is
  waiting on these, so they get longer patience and a larger retry budget, and a
  small pool so a slow bulk fetch cannot occupy the interactive pool.

Both share the same resilience policy shape:

* **Split timeouts.** Connect/read/write/pool are configured independently via
  ``Data360Settings``. A single flat 30 s timeout previously let one stalled
  downstream call hold an MCP tool request open for 30 s.
* **Bounded retries with exponential backoff and jitter** for transient failures
  (timeouts, connection errors, ``retry_status_codes``). ``retry_budget_seconds``
  caps the wall-clock time spent retrying a single request, so worst-case latency
  stays bounded instead of multiplying by ``retry_max_attempts``.
* **Retries are opt-in for non-idempotent methods.** GET/HEAD/OPTIONS/TRACE are
  always retried; POST (and other mutating methods) are retried only when the
  caller marks a read-only query with ``extensions={RETRY_SAFE_EXTENSION: True}``.
  The Data360 search/metadata/dimensions endpoints are read-only POSTs and set
  that marker; write endpoints such as the Charts API do not, so a timed-out
  write is never replayed.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

import httpx

from .config import Data360Settings, get_data360_settings
from .entropy import uniform_jitter

_logger = logging.getLogger(__name__)

# Request extension marking a read-only request as safe to retry (see module docstring).
RETRY_SAFE_EXTENSION = "data360_retry_safe"

_IDEMPOTENT_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

# Transport-level failures worth a second attempt. TimeoutException covers
# connect/read/write/pool timeouts; the rest cover refused/dropped connections.
_RETRYABLE_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.RemoteProtocolError,
)

# httpx defaults for the interactive pool; bulk work is rare and must not be able
# to occupy many sockets, so its pool is deliberately small (bulkhead).
_INTERACTIVE_LIMITS = httpx.Limits(max_connections=100, max_keepalive_connections=20)
_BULK_LIMITS = httpx.Limits(max_connections=8, max_keepalive_connections=4)

_client: httpx.AsyncClient | None = None
_client_lock = threading.Lock()
_bulk_client: httpx.AsyncClient | None = None
_bulk_client_lock = threading.Lock()


async def _sleep(delay: float) -> None:
    await asyncio.sleep(delay)


# Indirection so tests can assert retry timing without sleeping.
_RETRY_SLEEP = _sleep


@dataclass(frozen=True)
class _RetryPolicy:
    """Retry/backoff configuration for one client profile."""

    max_attempts: int
    budget_seconds: float
    backoff_base: float
    backoff_max: float
    retry_status_codes: frozenset[int]


class _RetryingTransport(httpx.AsyncHTTPTransport):
    """Retry transient failures for requests that are safe to replay."""

    def __init__(self, *, policy: _RetryPolicy, limits: httpx.Limits) -> None:
        super().__init__(limits=limits)
        self._policy = policy

    @staticmethod
    def _retry_allowed(request: httpx.Request) -> bool:
        return (
            request.method in _IDEMPOTENT_METHODS
            or request.extensions.get(RETRY_SAFE_EXTENSION) is True
        )

    async def _backoff(
        self, request: httpx.Request, attempt: int, reason: BaseException | None
    ) -> None:
        policy = self._policy
        delay = min(policy.backoff_max, policy.backoff_base * (2 ** (attempt - 1)))
        # Equal jitter keeps clients from retrying in lockstep without giving up
        # the exponential growth between consecutive delays. Drawn from the OS
        # CSPRNG (see data360.entropy) to satisfy the CWE-331 policy.
        delay = uniform_jitter(delay / 2, delay)
        if reason is None:
            detail = "retryable status"
        else:
            message = str(reason).strip()
            detail = (
                f"{type(reason).__name__}: {message}"
                if message
                else type(reason).__name__
            )
        _logger.warning(
            "Retrying %s %s (attempt %d/%d) in %.2fs after %s",
            request.method,
            request.url.path,
            attempt + 1,
            policy.max_attempts,
            delay,
            detail,
        )
        await _RETRY_SLEEP(delay)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        policy = self._policy
        attempts = policy.max_attempts if self._retry_allowed(request) else 1
        deadline = time.monotonic() + policy.budget_seconds

        for attempt in range(1, attempts + 1):
            # Upper bound on this attempt's backoff delay (jitter is <= this value).
            max_delay = min(
                policy.backoff_max, policy.backoff_base * (2 ** (attempt - 1))
            )
            try:
                response = await super().handle_async_request(request)
            except _RETRYABLE_EXCEPTIONS as exc:
                if attempt >= attempts or time.monotonic() + max_delay >= deadline:
                    raise
                await self._backoff(request, attempt, exc)
                continue

            if response.status_code in policy.retry_status_codes and attempt < attempts:
                # Only schedule a retry if the backoff delay itself fits in the budget.
                if time.monotonic() + max_delay >= deadline:
                    return response
                await response.aclose()
                await self._backoff(request, attempt, None)
                continue

            return response
        raise AssertionError("retry loop exited without a response")  # pragma: no cover


def _build_client(
    *,
    timeout: httpx.Timeout,
    limits: httpx.Limits,
    policy: _RetryPolicy,
) -> httpx.AsyncClient:
    transport = _RetryingTransport(policy=policy, limits=limits)
    return httpx.AsyncClient(timeout=timeout, transport=transport)


def _policy(
    *,
    max_attempts: int,
    budget_seconds: float,
    settings: Data360Settings,
) -> _RetryPolicy:
    return _RetryPolicy(
        max_attempts=max(1, max_attempts),
        budget_seconds=budget_seconds,
        backoff_base=settings.retry_backoff_base,
        backoff_max=settings.retry_backoff_max,
        retry_status_codes=frozenset(settings.retry_status_codes),
    )


def _interactive_client() -> httpx.AsyncClient:
    settings = get_data360_settings()
    return _build_client(
        timeout=httpx.Timeout(
            connect=settings.connect_timeout,
            read=settings.read_timeout,
            write=settings.write_timeout,
            pool=settings.pool_timeout,
        ),
        limits=_INTERACTIVE_LIMITS,
        policy=_policy(
            max_attempts=settings.retry_max_attempts,
            budget_seconds=settings.retry_budget_seconds,
            settings=settings,
        ),
    )


def _bulk_http_client() -> httpx.AsyncClient:
    settings = get_data360_settings()
    return _build_client(
        timeout=httpx.Timeout(
            connect=settings.background_connect_timeout,
            read=settings.background_read_timeout,
            write=settings.background_write_timeout,
            pool=settings.background_pool_timeout,
        ),
        limits=_BULK_LIMITS,
        policy=_policy(
            max_attempts=settings.background_retry_max_attempts,
            budget_seconds=settings.background_retry_budget_seconds,
            settings=settings,
        ),
    )


def get_shared_httpx_client() -> httpx.AsyncClient:
    """Return the process-wide client for interactive (user-facing) calls."""
    global _client  # noqa: PLW0603
    client = _client
    if client is not None and not client.is_closed:
        return client

    with _client_lock:
        client = _client
        if client is None or client.is_closed:
            client = _interactive_client()
            _client = client
        return client


def get_background_httpx_client() -> httpx.AsyncClient:
    """Return the process-wide client for background/bulk calls.

    Separate instance and connection pool from the interactive client, so a slow
    catalog fetch cannot starve user-facing requests.
    """
    global _bulk_client  # noqa: PLW0603
    client = _bulk_client
    if client is not None and not client.is_closed:
        return client

    with _bulk_client_lock:
        client = _bulk_client
        if client is None or client.is_closed:
            client = _bulk_http_client()
            _bulk_client = client
        return client


async def _aclose(client: httpx.AsyncClient | None) -> None:
    if client is not None and not client.is_closed:
        await client.aclose()


async def aclose_shared_httpx_client() -> None:
    """Close the interactive client (call from ASGI shutdown)."""
    global _client  # noqa: PLW0603
    with _client_lock:
        client = _client
        _client = None

    await _aclose(client)


async def aclose_background_httpx_client() -> None:
    """Close the background/bulk client (call from ASGI shutdown)."""
    global _bulk_client  # noqa: PLW0603
    with _bulk_client_lock:
        client = _bulk_client
        _bulk_client = None

    await _aclose(client)


async def aclose_all_httpx_clients() -> None:
    """Close every per-process client (ASGI shutdown / test teardown)."""
    await aclose_shared_httpx_client()
    await aclose_background_httpx_client()
