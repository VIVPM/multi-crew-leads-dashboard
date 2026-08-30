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

# cp1252 consoles can't print the emoji that show up in agent output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Loaded here because the tracing setup below runs before backend.backend is imported
from dotenv import load_dotenv  # noqa: E402
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

from logging_setup import configure_logging, job_context, current_correlation_ids  # noqa: E402

configure_logging()  # idempotent — no-op if backend.py already configured it in-process
logger = logging.getLogger("worker")

# Optional OTLP tracing of agent/task/LLM calls, exported to Langfuse and/or
# Grafana Cloud. Must run before pipeline.py pulls in langchain — the
# instrumentor patches LangChain's callback manager at import time.
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
        # Langfuse v4 queries observations directly, so an attribute that sits
        # only on the root span can't filter or aggregate its children. Copy the
        # job/lead IDs onto every span. Reads the same contextvars the JSON logs
        # use, so a span and a log line for one job carry matching IDs.
        #
        # Both prefixes on purpose, confirmed against a real canary trace:
        # `langfuse.trace.metadata.*` is folded into the trace's metadata and
        # does NOT appear on the individual observations, so on its own it
        # gives trace-level filtering only. `langfuse.observation.metadata.*`
        # is what lands on each observation and makes it filterable by itself,
        # which is the v4 requirement. Plain unprefixed attributes fall into
        # the metadata.attributes catch-all, which isn't queryable at all.
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
                # v4 ingestion: without this header directly-ingested OTEL data
                # can lag the v4 data model and the v2 APIs by up to 10 minutes.
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

        # One instrumentor covers both levels now: LangGraph runs on LangChain's
        # callback manager, so the graph nodes and the model calls underneath
        # them come out of the same hook. Under CrewAI this took two.
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
        logger.info("LLM tracing enabled via OTLP: %s", ", ".join(_enabled))
    except Exception:
        logger.exception("Failed to initialize LLM tracing (non-fatal)")
else:
    logger.info("No tracing backend configured (Langfuse/Grafana) — LLM tracing disabled")

# Job outcomes as a counter metric, so alerting can query it directly
_jobs_processed_counter = None
# Spend, as counters rather than a value read out of analysis_runs one lead at
# a time. Keys are operator-held, so cost is the number that actually needs a
# trend line and an alert on it — a run of unusually expensive leads is
# invisible in a per-lead table until the bill arrives.
_tokens_counter = None
_cost_counter = None
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
        logger.info("Job metrics enabled via OTLP (Grafana Cloud)")
    except Exception:
        logger.exception("Failed to initialize job metrics (non-fatal)")

# Import path differs between uvicorn (repo root) and standalone
try:
    from backend.backend import supabase, persist_results  # noqa: E402
except ImportError:
    from backend import supabase, persist_results  # noqa: E402
from pipeline import LLM_MODEL, process_leads, PIPELINE_TIMEOUT_S, is_retryable  # noqa: E402

POLL_INTERVAL_S = 3

# Mirrors process_leads' max_retries — bounds how long a job can legitimately run
PIPELINE_MAX_ATTEMPTS = 3

MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))

# Short TTL: a company can shut down or get acquired between lookups
COMPANY_CACHE_TTL_DAYS = int(os.getenv("COMPANY_CACHE_TTL_DAYS", "7"))

# How long a shutdown waits for in-flight jobs before giving up on them.
#
# The usual advice is "longer than p99 job duration", which here would be
# minutes — a lead takes ~50s and a job can hold several. That's not
# achievable on Render, which sends SIGTERM and then SIGKILLs about 30s
# later, so the default sits inside that window rather than pretending to
# outlast it. Raise it (and the platform's own termination grace period)
# together if you ever want a whole job to survive a redeploy.
#
# Even at 25s this is strictly better than no drain at all: the worker stops
# claiming immediately, so a redeploy no longer kills jobs it accepted
# seconds earlier, and single-lead jobs already past their model calls get
# to write their results.
WORKER_SHUTDOWN_GRACE_S = int(os.getenv("WORKER_SHUTDOWN_GRACE_S", "25"))

# Circuit breaker. Consecutive transport-shaped job failures before the worker
# stops claiming, and how long it waits before trying again.
#
# Without this, a provider outage is worse than downtime: the worker keeps
# claiming, each job burns its retries, and the whole backlog converts itself
# into failed jobs that a user has to resubmit — paying for the retries on the
# way. Pausing leaves the queue intact so the work survives the outage.
#
# Only transport failures count. A run of malformed leads failing validation is
# not an outage and must not stop the queue.
BREAKER_THRESHOLD = int(os.getenv("BREAKER_THRESHOLD", "5"))
BREAKER_COOLDOWN_S = int(os.getenv("BREAKER_COOLDOWN_S", "60"))

_consecutive_failures = 0
_breaker_open_until = 0.0


def _record_job_outcome(exc: Optional[BaseException]) -> None:
    """Feed one job's result to the breaker."""
    global _consecutive_failures, _breaker_open_until
    if exc is None:
        _consecutive_failures = 0  # one success is enough to call it recovered
        return
    if not is_retryable(exc):
        return  # the job's own fault, not the provider's
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
        return  # second signal — the drain is already under way
    _stopping = True
    logger.info("Signal %s received; draining, no new jobs will be claimed", signum)


def _cache_row(key: str) -> Optional[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=COMPANY_CACHE_TTL_DAYS)).isoformat()
    resp = (
        supabase.table("company_research_cache")
        .select("company_info,cultural_fit_score,cultural_fit_notes")
        .eq("company_key", key)
        .gte("cached_at", cutoff)
        .gte("cultural_fit_score", 0)  # excludes in-flight claim placeholders (sentinel -1)
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
            "cultural_fit_score": -1,  # sentinel: claimed, research in flight
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        return None  # we won the claim — caller does the research
    except Exception:
        pass  # someone else already claimed this key — wait on them instead

    # Wait for the winner; if it crashed, fall through and research it ourselves
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        time.sleep(2)
        hit = _cache_row(key)
        if hit is not None:
            return hit
    return None  # winner still not done — research it ourselves as a fallback


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
        # Upsert rather than select-then-write, so two writers can't create two rows
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
            # No timestamp to age it against, so treat it as abandoned
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


def claim_next_job():
    """Atomically claim the oldest pending job; returns None if there is none."""
    pending = (
        supabase.table("jobs")
        .select("*")
        .eq("status", "pending")
        .order("created_at")
        .limit(1)
        .execute()
    )
    if not pending.data:
        return None
    job = pending.data[0]
    claimed = (
        supabase.table("jobs")
        .update({"status": "running", "started_at": datetime.now(timezone.utc).isoformat()})
        .eq("id", job["id"])
        .eq("status", "pending")  # conditional update: loses the race harmlessly
        .execute()
    )
    if not claimed.data:
        return None
    return job


async def run_job(job: dict) -> list:
    leads = job["leads"]
    start = time.time()

    # Stage updates for the UI's progress tracker, written as each agent finishes
    progress: dict = {}

    def _on_stage(stage: str, state: str) -> None:
        progress[stage] = state
        supabase.table("jobs").update({"progress": progress}).eq("id", job["id"]).execute()

    scores, emails, agent_times, cache_hits = await process_leads(
        leads, job["gemini_api_key"], job["tavily_api_key"],
        our_company_context=job.get("our_company_context") or "",
        cache_get=cache_get_company,
        cache_set=cache_set_company,
        force_refresh=job.get("force_refresh", False),
        on_stage=_on_stage,
    )
    elapsed = round(time.time() - start, 1)
    results = persist_results(leads, scores, emails, agent_times, cache_hits, elapsed)

    # Emitted here rather than inside persist_results so pipeline/DB code stays
    # free of metrics wiring, and so cost is counted once per job even when a
    # lead's analysis row is later overwritten by a re-run.
    if _tokens_counter is not None:
        tokens = sum((getattr(s.token_usage, "total_tokens", 0) or 0) for s in scores)
        tokens += sum((getattr(e.token_usage, "total_tokens", 0) or 0) for e in emails if e)
        prompt = sum((getattr(s.token_usage, "prompt_tokens", 0) or 0) for s in scores)
        prompt += sum((getattr(e.token_usage, "prompt_tokens", 0) or 0) for e in emails if e)
        completion = tokens - prompt
        # Same gemini-2.5-flash rates persist_results prices with.
        cost = round(prompt * 0.15 / 1_000_000 + completion * 0.60 / 1_000_000, 6)
        attrs = {"provider": LLM_MODEL}
        _tokens_counter.add(tokens, attrs)
        _cost_counter.add(cost, attrs)
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
    # pipeline.py runs each job's synchronous work through asyncio.to_thread,
    # which hands it to the loop's default executor. That default is
    # ThreadPoolExecutor(min(32, cpu_count + 4)) — on a 1-vCPU box, 5 threads.
    # So MAX_CONCURRENT_JOBS=10 was a number the runtime could not honour: the
    # loop claimed ten jobs, ran five, and left the rest sitting in `running`
    # doing nothing while their time budget ticked down toward being reaped.
    #
    # Sizing the executor to the setting makes the setting mean what it says.
    # One thread per in-flight job is exact, not an estimate — the tools inside
    # a job (requests, tavily) run on that same thread.
    #
    # Scoped to this loop, so uvicorn's own threadpool is untouched when
    # RUN_WORKER_IN_PROCESS=1 puts this loop on a worker thread.
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
        # Handlers can only be installed from the main thread. With
        # RUN_WORKER_IN_PROCESS=1 this runs in a uvicorn worker thread, where
        # signal.signal raises ValueError — uvicorn owns the signals there and
        # this loop just never sees one.
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, _request_stop)
    except ValueError:
        logger.info("Not on the main thread; uvicorn owns signal handling here")

    try:
        fail_stale_running_jobs()
    except Exception:
        logger.exception("Failed to clean up stale jobs on startup")

    in_flight: set = set()
    while not _stopping:
        # Only claim what we can start now, so no job sits claimed but unworked
        while not _stopping and not _breaker_is_open() and len(in_flight) < MAX_CONCURRENT_JOBS:
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
            # Wake when a slot frees up or the poll interval elapses
            await asyncio.wait(in_flight, timeout=POLL_INTERVAL_S, return_when=asyncio.FIRST_COMPLETED)

    if in_flight:
        logger.info(
            "Draining %d in-flight job(s), up to %ds", len(in_flight), WORKER_SHUTDOWN_GRACE_S,
        )
        _, pending = await asyncio.wait(in_flight, timeout=WORKER_SHUTDOWN_GRACE_S)
        if pending:
            # Left running rather than cancelled: a cancel mid-model-call would
            # abandon a job we've already paid for, with nothing written back.
            # fail_stale_running_jobs on the next boot reaps them.
            logger.warning(
                "%d job(s) outlasted the grace period; the next worker will reap them",
                len(pending),
            )
    logger.info("Worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
