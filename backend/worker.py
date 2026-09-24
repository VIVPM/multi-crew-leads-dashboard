"""
worker.py — background processor for lead-scoring jobs.

Run alongside the API server:  python backend/worker.py
Scale throughput by running more worker processes: job claiming is guarded
by a conditional status update so workers never double-process, and a
starting worker only reaps jobs that have outlived their own time budget,
so it cannot kill work another worker still has in flight. That second
half was added after load testing measured a second worker destroying
100% of the first one's running jobs (see fail_stale_running_jobs).
"""

import os
import signal
import sys
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional


if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


from dotenv import load_dotenv  # noqa: E402
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

from logging_setup import configure_logging, job_context, current_correlation_ids  # noqa: E402

configure_logging()
logger = logging.getLogger("worker")


_have_langfuse = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
_have_grafana = bool(os.getenv("GRAFANA_OTLP_ENDPOINT") and os.getenv("GRAFANA_OTLP_AUTH"))

if _have_langfuse or _have_grafana:
    try:
        import base64
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from openinference.instrumentation.langchain import LangChainInstrumentor

        _tracer_provider = TracerProvider(resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "sales-pipeline-backend"),
            "service.namespace": "lead-coordinator",
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
        }))
        # Copies correlation IDs onto both Langfuse traces and observations.
        class _CorrelationSpanProcessor(SpanProcessor):
            def on_start(self, span, parent_context=None):
                for key, value in current_correlation_ids().items():
                    if value is not None:
                        span.set_attribute(f"langfuse.trace.metadata.{key}", value)
                        span.set_attribute(f"langfuse.observation.metadata.{key}", value)

        _tracer_provider.add_span_processor(_CorrelationSpanProcessor())
        _enabled = []

        if _have_langfuse:
            _lf_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
            _creds = f'{os.environ["LANGFUSE_PUBLIC_KEY"]}:{os.environ["LANGFUSE_SECRET_KEY"]}'
            _auth = base64.b64encode(_creds.encode()).decode()
            _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{_lf_host}/api/public/otel/v1/traces",


                headers={
                    "Authorization": f"Basic {_auth}",
                    "x-langfuse-ingestion-version": "4",
                },
            )))
            _enabled.append(f"Langfuse ({_lf_host})")

        if _have_grafana:
            _tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            )))
            _enabled.append("Grafana Cloud")


        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        logger.info("LLM tracing enabled via OTLP: %s", ", ".join(_enabled))
    except Exception:
        logger.exception("Failed to initialize LLM tracing (non-fatal)")
else:
    logger.info("No tracing backend configured (Langfuse/Grafana) — LLM tracing disabled")


_jobs_processed_counter = None


_tokens_counter = None
_cost_counter = None


_ttft_histogram = None
_tokens_per_s_histogram = None
_queue_depth_gauge = None
_concurrency_gauge = None
if _have_grafana:
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

        _metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(
                endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/metrics",
                headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
            ),
            export_interval_millis=15000,
        )
        _meter_provider = MeterProvider(
            resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "sales-pipeline-backend")}),
            metric_readers=[_metric_reader],
        )
        _meter = _meter_provider.get_meter("worker")
        _jobs_processed_counter = _meter.create_counter(
            "jobs_processed_total", description="Jobs finished, by status (done/failed)",
        )
        _tokens_counter = _meter.create_counter(
            "llm_tokens_total", unit="token",
            description="LLM tokens spent, by provider",
        )
        _cost_counter = _meter.create_counter(
            "llm_cost_usd_total", unit="USD",
            description="Estimated LLM spend, by provider",
        )
        _ttft_histogram = _meter.create_histogram(
            "llm_ttft_seconds", unit="s",
            description="Time to first token, by provider (streaming providers only)",
        )
        _tokens_per_s_histogram = _meter.create_histogram(
            "llm_tokens_per_second", unit="token/s",
            description="Generation throughput per model call, by provider",
        )
        _queue_depth_gauge = _meter.create_gauge(
            "queue_pending_jobs",
            description="Jobs waiting to be claimed — the autoscaling signal",
        )
        _concurrency_gauge = _meter.create_gauge(
            "worker_target_concurrency",
            description="Job slots the worker is currently willing to fill",
        )
        logger.info("Job metrics enabled via OTLP (Grafana Cloud)")
    except Exception:
        logger.exception("Failed to initialize job metrics (non-fatal)")


try:
    from backend.backend import supabase, persist_results  # noqa: E402
except ImportError:
    from backend import supabase, persist_results  # noqa: E402
from pipeline import LLM_MODEL, process_leads, PIPELINE_TIMEOUT_S, is_retryable  # noqa: E402
from queue_policy import choose_round_robin_job, next_concurrency  # noqa: E402

POLL_INTERVAL_S = 3


PIPELINE_MAX_ATTEMPTS = 3

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))
if MAX_CONCURRENT_JOBS < 1:
    raise RuntimeError("MAX_CONCURRENT_JOBS must be at least 1")


WORKER_MIN_CONCURRENCY = int(os.getenv("WORKER_MIN_CONCURRENCY", "2"))
SCALE_UP_QUEUE_DEPTH = int(os.getenv("SCALE_UP_QUEUE_DEPTH", "5"))
SCALE_COOLDOWN_S = float(os.getenv("SCALE_COOLDOWN_S", "30"))
if not 1 <= WORKER_MIN_CONCURRENCY <= MAX_CONCURRENT_JOBS:
    raise RuntimeError(
        f"WORKER_MIN_CONCURRENCY must be between 1 and MAX_CONCURRENT_JOBS "
        f"({MAX_CONCURRENT_JOBS}), got {WORKER_MIN_CONCURRENCY}"
    )
if SCALE_UP_QUEUE_DEPTH < 1:
    raise RuntimeError("SCALE_UP_QUEUE_DEPTH must be at least 1")

FAIR_CLAIM_SCAN_LIMIT = int(os.getenv("FAIR_CLAIM_SCAN_LIMIT", "100"))
if FAIR_CLAIM_SCAN_LIMIT < 2:
    raise RuntimeError("FAIR_CLAIM_SCAN_LIMIT must be at least 2")
_last_claimed_user_id: Optional[str] = None


COMPANY_CACHE_TTL_DAYS = int(os.getenv("COMPANY_CACHE_TTL_DAYS", "7"))


WORKER_SHUTDOWN_GRACE_S = int(os.getenv("WORKER_SHUTDOWN_GRACE_S", "25"))


BREAKER_THRESHOLD = int(os.getenv("BREAKER_THRESHOLD", "5"))
BREAKER_COOLDOWN_S = int(os.getenv("BREAKER_COOLDOWN_S", "60"))

_consecutive_failures = 0
_breaker_open_until = 0.0


def _record_job_outcome(exc: Optional[BaseException]) -> None:
    """Feed one job's result to the breaker."""
    global _consecutive_failures, _breaker_open_until
    if exc is None:
        _consecutive_failures = 0
        return
    if not is_retryable(exc):
        return
    _consecutive_failures += 1
    if _consecutive_failures >= BREAKER_THRESHOLD:
        _breaker_open_until = time.monotonic() + BREAKER_COOLDOWN_S
        _consecutive_failures = 0
        logger.error(
            "Circuit breaker open: %d consecutive transport failures. "
            "Not claiming for %ds; queued jobs stay queued.",
            BREAKER_THRESHOLD, BREAKER_COOLDOWN_S,
        )


def _breaker_is_open() -> bool:
    return time.monotonic() < _breaker_open_until

_stopping = False


def _request_stop(signum, _frame) -> None:
    """Stop claiming new jobs; let the ones already running finish."""
    global _stopping
    if _stopping:
        return
    _stopping = True
    logger.info("Signal %s received; draining, no new jobs will be claimed", signum)


def _cache_row(key: str) -> Optional[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=COMPANY_CACHE_TTL_DAYS)).isoformat()
    resp = (
        supabase.table("company_research_cache")
        .select("company_info,cultural_fit_score,cultural_fit_notes")
        .eq("company_key", key)
        .gte("cached_at", cutoff)
        .gte("cultural_fit_score", 0)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def cache_get_company(key: str) -> Optional[dict]:
    """Return cached company research, or None if the caller should research it.

    On a miss, claims the key first so concurrent misses for the same company
    don't all pay for the same research. company_key is unique, so exactly one
    caller wins the claim; the rest wait for its result.
    """
    hit = _cache_row(key)
    if hit is not None:
        return hit

    try:
        supabase.table("company_research_cache").insert({
            "company_key": key,
            "company_name": key.split(":", 1)[0],
            "company_info": {},
            "cultural_fit_score": -1,
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return None
    except Exception:
        pass


    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(2)
        hit = _cache_row(key)
        if hit is not None:
            return hit
    return None


def cache_set_company(key: str, company_name: str, data: dict) -> None:
    row = {
        "company_key": key,
        "company_name": company_name,
        "company_info": data["company_info"],
        "cultural_fit_score": data["cultural_fit_score"],
        "cultural_fit_notes": data.get("cultural_fit_notes"),
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }
    try:

        supabase.table("company_research_cache").upsert(row, on_conflict="company_key").execute()
    except Exception:
        logger.exception("Failed to write company research cache (non-fatal)")


def _job_time_budget_s(job: dict) -> float:
    """Longest a job can legitimately stay 'running' before it must be dead."""
    return PIPELINE_TIMEOUT_S * max(1, len(job.get("leads") or [])) * PIPELINE_MAX_ATTEMPTS


def fail_stale_running_jobs():
    """Fail jobs abandoned by a crashed worker — but only those.

    A job is reaped only once it has been running longer than it could possibly
    need, so a booting worker can't kill another worker's in-flight jobs.
    """
    rows = supabase.table("jobs").select("id,leads,started_at").eq(
        "status", "running").execute().data or []
    now = datetime.now(timezone.utc)

    stale_ids = []
    for job in rows:
        started = job.get("started_at")
        if not started:

            stale_ids.append(job["id"])
            continue
        started_dt = datetime.fromisoformat(started.replace("Z", "+00:00"))
        if (now - started_dt).total_seconds() > _job_time_budget_s(job):
            stale_ids.append(job["id"])

    if not stale_ids:
        logger.info("No stale jobs to clean up (%d still running elsewhere)", len(rows))
        return

    supabase.table("jobs").update({
        "status": "failed",
        "error": "Worker stopped while the job was running. Please re-process the lead.",
        "gemini_api_key": None,
        "tavily_api_key": None,
    }).in_("id", stale_ids).execute()
    logger.warning("Marked %d stale running job(s) as failed", len(stale_ids))


def pending_depth() -> int:
    """How many jobs are waiting to be claimed.

    A HEAD count, not a row fetch: the claim scan already pulls rows, but it
    only runs when the worker has a free slot, so depth read from it would
    freeze exactly when the backlog matters — at capacity.
    """
    res = (
        supabase.table("jobs")
        .select("id", count="exact")
        .eq("status", "pending")
        .limit(1)
        .execute()
    )
    return res.count or 0


def claim_next_job():
    """Fairly choose and atomically claim a pending job.

    The database remains the durable queue.  Within the oldest candidate
    window, tenants are served round-robin and each tenant remains FIFO.  The
    cursor is process-local by design: this deployment uses one worker, while
    the conditional update below still prevents double processing if another
    worker is ever added.
    """
    global _last_claimed_user_id
    pending = (
        supabase.table("jobs")
        .select("*")
        .eq("status", "pending")
        .order("created_at")
        .limit(FAIR_CLAIM_SCAN_LIMIT)
        .execute()
    )
    if not pending.data:
        return None
    job = choose_round_robin_job(pending.data, _last_claimed_user_id)
    claimed = (
        supabase.table("jobs")
        .update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", job["id"])
        .eq("status", "pending")
        .execute()
    )
    if not claimed.data:
        return None
    _last_claimed_user_id = str(job.get("user_id") or "")
    return job


async def run_job(job: dict) -> list:
    leads = job["leads"]
    start = time.time()


    progress: dict = {}

    def _on_stage(stage: str, state: str) -> None:
        progress[stage] = state
        supabase.table("jobs").update({"progress": progress}).eq("id", job["id"]).execute()

    llm_stats: list = []
    scores, emails, agent_times, cache_hits = await process_leads(
        leads, job["gemini_api_key"], job["tavily_api_key"],
        our_company_context=job.get("our_company_context") or "",
        cache_get=cache_get_company,
        cache_set=cache_set_company,
        force_refresh=job.get("force_refresh", False),
        on_stage=_on_stage,
        llm_stats=llm_stats,
    )
    elapsed = round(time.time() - start, 1)
    results = persist_results(leads, scores, emails, agent_times, cache_hits, elapsed)


    if _tokens_counter is not None:
        tokens = sum((getattr(s.token_usage, "total_tokens", 0) or 0) for s in scores)
        tokens += sum((getattr(e.token_usage, "total_tokens", 0) or 0) for e in emails if e)
        prompt = sum((getattr(s.token_usage, "prompt_tokens", 0) or 0) for s in scores)
        prompt += sum((getattr(e.token_usage, "prompt_tokens", 0) or 0) for e in emails if e)
        completion = tokens - prompt

        cost = round(prompt * 0.15 / 1_000_000 + completion * 0.60 / 1_000_000, 6)
        attrs = {"provider": LLM_MODEL}
        _tokens_counter.add(tokens, attrs)
        _cost_counter.add(cost, attrs)


    if _ttft_histogram is not None:
        for call in llm_stats:
            call_attrs = {"provider": call["provider"]}
            if call["ttft_s"] is not None:
                _ttft_histogram.record(call["ttft_s"], call_attrs)
            if call["tokens_per_s"] is not None:
                _tokens_per_s_histogram.record(call["tokens_per_s"], call_attrs)
    return results


def finish_job(job_id: str, **fields):
    """Mark a job done or failed, wiping its stored API keys."""
    fields.update({"gemini_api_key": None, "tavily_api_key": None})
    supabase.table("jobs").update(fields).eq("id", job_id).execute()


async def process_one_job(job: dict) -> None:
    """Run one job to completion. Failures are contained to this job."""
    with job_context(job["id"]):
        logger.info("Claimed job (%d lead(s))", len(job["leads"]))
        try:
            results = await run_job(job)
            finish_job(job["id"], status="done", results=results)
            logger.info("Job done")
            _record_job_outcome(None)
            if _jobs_processed_counter:
                _jobs_processed_counter.add(1, {"status": "done"})
        except Exception as exc:
            logger.exception("Job failed")
            finish_job(job["id"], status="failed", error=str(exc)[:500])
            _record_job_outcome(exc)
            if _jobs_processed_counter:
                _jobs_processed_counter.add(1, {"status": "failed"})


async def main():
    logger.info(
        "Worker started (poll interval %ds, up to %d job(s) concurrently)",
        POLL_INTERVAL_S, MAX_CONCURRENT_JOBS,
    )


    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="pipeline")
    )
    _default_pool = min(32, (os.cpu_count() or 1) + 4)
    if _default_pool < MAX_CONCURRENT_JOBS:
        logger.info(
            "Pipeline executor sized to %d; the interpreter default here would "
            "have been %d, capping real concurrency below the setting.",
            MAX_CONCURRENT_JOBS, _default_pool,
        )

    try:


        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _request_stop)
    except ValueError:
        logger.info("Not on the main thread; uvicorn owns signal handling here")

    try:
        fail_stale_running_jobs()
    except Exception:
        logger.exception("Failed to clean up stale jobs on startup")

    in_flight: set = set()
    target = WORKER_MIN_CONCURRENCY
    last_change = time.monotonic()
    while not _stopping:


        try:
            depth = pending_depth()
        except Exception:
            logger.exception("Failed to read queue depth; holding concurrency at %d", target)
            depth = None
        if depth is not None:
            new_target = next_concurrency(
                target, depth, time.monotonic() - last_change,
                minimum=WORKER_MIN_CONCURRENCY, maximum=MAX_CONCURRENT_JOBS,
                scale_up_depth=SCALE_UP_QUEUE_DEPTH, cooldown_s=SCALE_COOLDOWN_S,
            )
            if new_target != target:
                logger.info(
                    "Scaling concurrency %d -> %d (queue depth %d)", target, new_target, depth,
                )
                target, last_change = new_target, time.monotonic()
            if _queue_depth_gauge is not None:
                _queue_depth_gauge.set(depth)
                _concurrency_gauge.set(target)


        while not _stopping and not _breaker_is_open() and len(in_flight) < target:
            try:
                job = claim_next_job()
            except Exception:
                logger.exception("Failed to poll jobs table; retrying")
                break
            if job is None:
                break
            task = asyncio.create_task(process_one_job(job))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

        if not in_flight:
            await asyncio.sleep(POLL_INTERVAL_S)
        else:

            await asyncio.wait(in_flight, timeout=POLL_INTERVAL_S, return_when=asyncio.FIRST_COMPLETED)

    if in_flight:
        logger.info(
            "Draining %d in-flight job(s), up to %ds", len(in_flight), WORKER_SHUTDOWN_GRACE_S,
        )
        _, pending = await asyncio.wait(in_flight, timeout=WORKER_SHUTDOWN_GRACE_S)
        if pending:


            logger.warning(
                "%d job(s) outlasted the grace period; the next worker will reap them",
                len(pending),
            )
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
