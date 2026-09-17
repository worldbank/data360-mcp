"""
Manual MCP load / rate-limit harness for Data360 MCP (JSON-RPC over SSE).

Requires the `requests` package (listed under [dependency-groups] dev in pyproject.toml).

Environment:
  DATA360_MCP_URL or MCP_SERVER_URL — base URL for MCP POST (default: QA APIM).
  APIM_SUBSCRIPTION_KEY — if set, sent as Ocp-Apim-Subscription-Key (Azure API Management).

Example:
  uv run python scripts/test_rate_limits.py --phase all --url https://example/mcp
"""

from __future__ import annotations

import json
import os
import secrets
import statistics
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Literal

import requests
import typer

from data360.entropy import uniform_jitter

# ── Defaults ───────────────────────────────────────────────────
DEFAULT_MCP_URL = "https://azapimqa.worldbank.org/public/data360/mcp"

BASE_SSE_HEADERS: dict[str, str] = {
    "Accept": "application/json, text/event-stream",
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
}

DEFAULT_REQUEST_TIMEOUT = (5, 30)  # connect, read
DEFAULT_MAX_WORKERS = 20
DEFAULT_REQUESTS_PER_PHASE = 100
DEFAULT_SOAK_DURATION_SECONDS = 300
DEFAULT_SOAK_CONCURRENCY = DEFAULT_MAX_WORKERS
DEFAULT_RETRY_ATTEMPTS = 2
DEFAULT_BACKOFF_BASE = 0.5
DEFAULT_RAMP_MAX_FAILURE_RATE = 0.10
DEFAULT_RAMP_MIN_429_COUNT = 1

Phase = Literal["all", "warmup", "ramp", "soak"]

LOAD_TEST_CASES: list[dict[str, Any]] = [
    {
        "label": "initialize",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "load-test-client", "version": "1.0"},
        },
    },
    {"label": "tools/list", "method": "tools/list", "params": {}},
    {
        "label": "data360_get_data",
        "method": "tools/call",
        "params": {
            "name": "data360_get_data",
            "arguments": {
                "database_id": "WB_WDI",
                "indicator_id": "WB_WDI_NY_GDP_PCAP_KD",
                "disaggregation_filters": {"REF_AREA": "KEN"},
                "start_year": 2015,
                "end_year": 2022,
                "limit": 10,
            },
        },
    },
    {
        "label": "data360_search_indicators",
        "method": "tools/call",
        "params": {
            "name": "data360_search_indicators",
            "arguments": {"query": "GDP per capita"},
        },
    },
    {
        "label": "data360_get_metadata",
        "method": "tools/call",
        "params": {
            "name": "data360_get_metadata",
            "arguments": {
                "database_id": "WB_WDI",
                "indicator_id": "WB_WDI_NY_GDP_PCAP_KD",
                "fetch_disaggregation": True,
            },
        },
    },
]


@dataclass
class LoadTestConfig:
    """Per-run settings passed through the request path."""

    mcp_url: str
    request_headers: dict[str, str]
    request_timeout: tuple[float, float]
    retry_attempts: int
    backoff_base: float
    retry_on_429: bool


@dataclass
class RampOptions:
    """Parameters for the ramp phase."""

    requests_per_phase: int
    ramp_concurrencies: list[int]
    min_429_count: int
    max_failure_rate: float
    json_phases: list[dict[str, Any]] | None


def resolve_mcp_url(url: str | None) -> str:
    if url:
        return url
    return (
        os.environ.get("DATA360_MCP_URL")
        or os.environ.get("MCP_SERVER_URL")
        or DEFAULT_MCP_URL
    )


def build_request_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    h = dict(BASE_SSE_HEADERS)
    key = os.environ.get("APIM_SUBSCRIPTION_KEY")
    if key:
        h["Ocp-Apim-Subscription-Key"] = key
    if extra:
        h.update(extra)
    return h


def make_thread_local_session_factory(headers: dict[str, str]):
    """Return a callable that yields a thread-local requests.Session (safe for ThreadPoolExecutor)."""

    local = threading.local()

    def get_session() -> requests.Session:
        session = getattr(local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(headers)
            local.session = session
        return session

    return get_session


def classify_result(status_code: int | None, err: str | None, payload: Any) -> str:  # noqa: PLR0911
    if err:
        err_l = err.lower()
        if "timeout" in err_l:
            return "timeout"
        if "connection" in err_l:
            return "connection_error"
        if "No data in SSE stream" in err:
            return "empty_sse"
        return "client_error"

    if status_code == HTTPStatus.TOO_MANY_REQUESTS:
        return "rate_limited"
    if status_code is not None and status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
        return "server_error"
    if status_code is not None and status_code >= HTTPStatus.BAD_REQUEST:
        return f"http_{status_code}"

    if payload is None:
        return "no_payload"
    if isinstance(payload, dict) and "error" in payload:
        return "jsonrpc_error"
    if (
        isinstance(payload, dict)
        and "result" in payload
        and isinstance(payload["result"], dict)
        and payload["result"].get("isError")
    ):
        return "tool_isError"
    return "success"


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * p
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def mcp_request(
    session: requests.Session,
    cfg: LoadTestConfig,
    method: str,
    params: dict[str, Any] | None,
    req_id: int = 1,
) -> dict[str, Any]:
    payload = {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params or {}}

    start = time.perf_counter()
    try:
        response = session.post(
            cfg.mcp_url,
            json=payload,
            headers=cfg.request_headers,
            stream=True,
            timeout=cfg.request_timeout,
        )
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "status_code": None,
            "err": "timeout",
            "payload": None,
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "status_code": None,
            "err": f"connection error: {e}",
            "payload": None,
        }
    except Exception as e:
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "status_code": None,
            "err": str(e),
            "payload": None,
        }

    status_code = response.status_code

    if status_code != HTTPStatus.OK:
        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "status_code": status_code,
            "err": response.text[:500],
            "payload": None,
        }

    try:
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8") if isinstance(line, bytes) else line
            if decoded.startswith("data:"):
                data_str = decoded[5:].strip()
                if not data_str:
                    continue
                try:
                    body = json.loads(data_str)
                    return {
                        "ok": True,
                        "latency": time.perf_counter() - start,
                        "status_code": status_code,
                        "err": None,
                        "payload": body,
                    }
                except json.JSONDecodeError:
                    continue

        return {
            "ok": False,
            "latency": time.perf_counter() - start,
            "status_code": status_code,
            "err": "No data in SSE stream",
            "payload": None,
        }
    finally:
        response.close()


def request_with_retry(
    session: requests.Session,
    test_case: dict[str, Any],
    req_id: int,
    cfg: LoadTestConfig,
) -> dict[str, Any]:
    attempt = 0
    last: dict[str, Any] | None = None

    while attempt <= cfg.retry_attempts:
        result = mcp_request(
            session, cfg, test_case["method"], test_case["params"], req_id=req_id
        )
        result["label"] = test_case["label"]
        result["attempt"] = attempt + 1

        category = classify_result(
            result["status_code"], result["err"], result["payload"]
        )
        result["category"] = category

        if category == "success":
            return result

        last = result
        retryable = {
            "timeout",
            "server_error",
            "empty_sse",
        }
        if cfg.retry_on_429:
            retryable.add("rate_limited")

        if attempt < cfg.retry_attempts and category in retryable:
            sleep_s = cfg.backoff_base * (2**attempt) + uniform_jitter(0, 0.2)
            time.sleep(sleep_s)
        else:
            return result

        attempt += 1

    return last  # type: ignore[return-value]


def run_batch(
    concurrency: int,
    total_requests: int,
    cfg: LoadTestConfig,
    get_session,
    mix_cases: bool = True,
) -> tuple[list[dict[str, Any]], float]:
    results: list[dict[str, Any]] = []
    start_batch = time.perf_counter()

    def worker(i: int) -> dict[str, Any]:
        case = secrets.choice(LOAD_TEST_CASES) if mix_cases else LOAD_TEST_CASES[0]
        session = get_session()
        return request_with_retry(session, case, i + 1, cfg)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, i) for i in range(total_requests)]
        for future in as_completed(futures):
            results.append(future.result())

    duration = time.perf_counter() - start_batch
    return results, duration


def summarize_results(
    name: str,
    results: list[dict[str, Any]],
    duration: float,
) -> dict[str, Any]:
    latencies = [float(r["latency"]) for r in results]
    categories = Counter(r["category"] for r in results)
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    for r in results:
        by_label[str(r["label"])][str(r["category"])] += 1

    success_count = categories.get("success", 0)
    total = len(results)
    rps = total / duration if duration > 0 else 0

    print(f"\n{'=' * 70}")
    print(name)
    print(f"{'=' * 70}")
    print(f"Total requests      : {total}")
    print(f"Duration (s)        : {duration:.2f}")
    print(f"Throughput (req/s)  : {rps:.2f}")
    print(
        f"Success rate        : {success_count}/{total} ({(100 * success_count / total):.2f}%)"
    )

    if latencies:
        print(f"Latency mean (s)    : {statistics.mean(latencies):.3f}")
        print(f"Latency median (s)  : {statistics.median(latencies):.3f}")
        p95 = percentile(latencies, 0.95)
        p99 = percentile(latencies, 0.99)
        print(
            f"Latency p95 (s)     : {p95:.3f}"
            if p95 is not None
            else "Latency p95 (s)     : n/a"
        )
        print(
            f"Latency p99 (s)     : {p99:.3f}"
            if p99 is not None
            else "Latency p99 (s)     : n/a"
        )
        print(f"Latency max (s)     : {max(latencies):.3f}")

    print("\nOutcome counts:")
    for k, v in categories.most_common():
        print(f"  - {k}: {v}")

    print("\nBy endpoint/test case:")
    for label, counter in sorted(by_label.items()):
        print(f"  {label}:")
        for k, v in counter.most_common():
            print(f"    - {k}: {v}")

    return {
        "name": name,
        "total_requests": total,
        "duration_seconds": duration,
        "throughput_rps": rps,
        "success_count": success_count,
        "success_rate": success_count / total if total else 0.0,
        "outcome_counts": dict(categories),
        "by_label": {lbl: dict(ct) for lbl, ct in by_label.items()},
        "latency": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95": percentile(latencies, 0.95),
            "p99": percentile(latencies, 0.99),
            "max": max(latencies) if latencies else None,
        },
    }


def ramp_should_stop(
    results: list[dict[str, Any]],
    *,
    min_429_count: int,
    max_failure_rate: float,
) -> bool:
    rate_limited = sum(1 for r in results if r["category"] == "rate_limited")
    success_n = sum(1 for r in results if r["category"] == "success")
    total = len(results)
    failure_rate = 1 - (success_n / total) if total else 0.0
    return rate_limited >= min_429_count or failure_rate > max_failure_rate


def ramp_test(cfg: LoadTestConfig, get_session, options: RampOptions) -> None:
    print("\n=== RAMP TEST ===")
    for concurrency in options.ramp_concurrencies:
        print(f"\nRunning ramp step with concurrency={concurrency}")
        results, duration = run_batch(
            concurrency=concurrency,
            total_requests=options.requests_per_phase,
            cfg=cfg,
            get_session=get_session,
            mix_cases=True,
        )
        summary = summarize_results(
            f"Ramp step @ concurrency={concurrency}", results, duration
        )
        if options.json_phases is not None:
            options.json_phases.append(summary)

        if ramp_should_stop(
            results,
            min_429_count=options.min_429_count,
            max_failure_rate=options.max_failure_rate,
        ):
            print(
                "\nStopping ramp: threshold reached "
                f"(min_429_count={options.min_429_count}, "
                f"max_failure_rate={options.max_failure_rate})."
            )
            break


def soak_test(
    cfg: LoadTestConfig,
    get_session,
    concurrency: int,
    duration_seconds: float,
    json_phases: list[dict[str, Any]] | None,
) -> None:
    print("\n=== SOAK TEST ===")
    wall_start = time.perf_counter()
    deadline = time.time() + duration_seconds
    all_results: list[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: set[Future[dict[str, Any]]] = set()
        req_id = 1

        while len(futures) < concurrency and time.time() < deadline:
            case = secrets.choice(LOAD_TEST_CASES)
            futures.add(
                executor.submit(
                    _soak_once,
                    get_session,
                    case,
                    req_id,
                    cfg,
                )
            )
            req_id += 1

        while futures:
            done, _pending = wait(futures, return_when=FIRST_COMPLETED)
            for f in done:
                futures.discard(f)
                all_results.append(f.result())

                if time.time() < deadline:
                    case = secrets.choice(LOAD_TEST_CASES)
                    futures.add(
                        executor.submit(
                            _soak_once,
                            get_session,
                            case,
                            req_id,
                            cfg,
                        )
                    )
                    req_id += 1

    wall_elapsed = time.perf_counter() - wall_start
    summary = summarize_results(
        f"Soak test @ concurrency={concurrency} for ~{duration_seconds:.0f}s scheduled "
        f"(wall {wall_elapsed:.2f}s)",
        all_results,
        wall_elapsed,
    )
    if json_phases is not None:
        json_phases.append(summary)


def _soak_once(
    get_session,
    case: dict[str, Any],
    req_id: int,
    cfg: LoadTestConfig,
) -> dict[str, Any]:
    session = get_session()
    return request_with_retry(session, case, req_id, cfg)


app = typer.Typer(no_args_is_help=True)


@app.command()
def main(  # noqa: PLR0913
    phase: Phase = typer.Option(
        "all",
        "--phase",
        "-p",
        help="Which phases to run: all, warmup, ramp, or soak.",
    ),
    url: str | None = typer.Option(
        None,
        "--url",
        "-u",
        envvar="DATA360_MCP_URL",
        help="MCP base URL (overrides DATA360_MCP_URL / MCP_SERVER_URL).",
    ),
    warmup_concurrency: int = typer.Option(2, "--warmup-concurrency"),
    warmup_requests: int = typer.Option(10, "--warmup-requests"),
    ramp_requests: int = typer.Option(
        DEFAULT_REQUESTS_PER_PHASE,
        "--ramp-requests",
        help="Requests per ramp step.",
    ),
    ramp_concurrency: str = typer.Option(
        "1,2,5,10,20,30",
        "--ramp-concurrency",
        help="Comma-separated concurrency steps for the ramp.",
    ),
    ramp_min_429: int = typer.Option(
        DEFAULT_RAMP_MIN_429_COUNT,
        "--ramp-min-429",
        help="Stop ramp when rate_limited count reaches this (use >1 to ignore sporadic 429s).",
    ),
    ramp_max_failure_rate: float = typer.Option(
        DEFAULT_RAMP_MAX_FAILURE_RATE,
        "--ramp-max-failure-rate",
        help="Stop ramp when non-success fraction exceeds this.",
    ),
    soak_concurrency: int = typer.Option(
        DEFAULT_SOAK_CONCURRENCY,
        "--soak-concurrency",
        help="Worker threads for soak (high values stress the client OS; start near 20).",
    ),
    soak_duration: float = typer.Option(
        float(DEFAULT_SOAK_DURATION_SECONDS),
        "--soak-duration",
        help="Scheduled soak duration in seconds (wall time includes draining in-flight work).",
    ),
    retry_attempts: int = typer.Option(DEFAULT_RETRY_ATTEMPTS, "--retry-attempts"),
    backoff_base: float = typer.Option(DEFAULT_BACKOFF_BASE, "--backoff-base"),
    retry_429: bool = typer.Option(
        False,
        "--retry-429/--no-retry-429",
        help="Retry on HTTP 429 (off by default; retries add load under throttle).",
    ),
    json_out: str | None = typer.Option(
        None,
        "--json-out",
        help="Write a JSON report (config + per-phase summaries) to this path.",
    ),
) -> None:
    mcp_url = resolve_mcp_url(url)
    headers = build_request_headers()
    cfg = LoadTestConfig(
        mcp_url=mcp_url,
        request_headers=headers,
        request_timeout=DEFAULT_REQUEST_TIMEOUT,
        retry_attempts=retry_attempts,
        backoff_base=backoff_base,
        retry_on_429=retry_429,
    )
    get_session = make_thread_local_session_factory(headers)

    ramp_steps = [int(x.strip()) for x in ramp_concurrency.split(",") if x.strip()]

    json_phases: list[dict[str, Any]] | None = [] if json_out else None

    report: dict[str, Any] = {
        "mcp_url": mcp_url,
        "phase": phase,
        "config": {
            "retry_attempts": retry_attempts,
            "backoff_base": backoff_base,
            "retry_on_429": retry_429,
            "ramp_requests": ramp_requests,
            "ramp_concurrency_steps": ramp_steps,
            "ramp_min_429": ramp_min_429,
            "ramp_max_failure_rate": ramp_max_failure_rate,
            "soak_concurrency": soak_concurrency,
            "soak_duration_seconds": soak_duration,
        },
        "phases": json_phases if json_phases is not None else [],
    }

    print(f"MCP Server URL: {mcp_url}")

    if phase in ("all", "warmup"):
        warmup_results, warmup_duration = run_batch(
            warmup_concurrency,
            warmup_requests,
            cfg,
            get_session,
            mix_cases=True,
        )
        wsum = summarize_results("Warmup", warmup_results, warmup_duration)
        if json_phases is not None:
            json_phases.append(wsum)

    if phase in ("all", "ramp"):
        ramp_test(
            cfg,
            get_session,
            RampOptions(
                requests_per_phase=ramp_requests,
                ramp_concurrencies=ramp_steps,
                min_429_count=ramp_min_429,
                max_failure_rate=ramp_max_failure_rate,
                json_phases=json_phases,
            ),
        )

    if phase in ("all", "soak"):
        soak_test(
            cfg,
            get_session,
            concurrency=soak_concurrency,
            duration_seconds=soak_duration,
            json_phases=json_phases,
        )

    if json_out:
        out_path = os.path.expanduser(json_out)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to {out_path}")


if __name__ == "__main__":
    app()
