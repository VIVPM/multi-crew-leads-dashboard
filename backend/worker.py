"""
worker.py — background processor for lead-scoring jobs.

Run alongside the API server:  python backend/worker.py
Scale throughput by running more worker processes; job claiming is
guarded by a conditional status update so workers never double-process.
"""

import os
import sys
import time
import asyncio
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# The API module is `backend.backend` when served by uvicorn from the repo
# root, but plain `backend` when this file runs standalone (python backend/worker.py).
try:
    from backend.backend import supabase, persist_results  # noqa: E402
except ImportError:
    from backend import supabase, persist_results  # noqa: E402
from pipeline import process_leads  # noqa: E402

POLL_INTERVAL_S = 3

# Jobs from different users share nothing (own leads, own API keys) and have
# no legitimate reason to serialize — this bounds how many run at once in
# this process, not how many CAN run (build_crews() makes fresh Crew objects
# per job, so there's no shared-state reason to keep this at 1). Real ceiling
# is Gemini/Tavily rate limits on whichever key each job brings, which this
# worker has no visibility into — so this is a blunt cap, not a tuned one.
MAX_CONCURRENT_JOBS = int(os.getenv("MAX_CONCURRENT_JOBS", "10"))

# Short deliberately: company status can change (shut down, acquired) between
# lookups, and a wrong "still viable" read is worse than the cost of a re-fetch.
COMPANY_CACHE_TTL_DAYS = int(os.getenv("COMPANY_CACHE_TTL_DAYS", "7"))


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
    hit = _cache_row(key)
    if hit is not None:
        return hit

    # Cache miss — claim this key so concurrent requests for the same
    # brand-new company don't all pay for the same Tavily + Gemini research.
    # company_key is unique (migrations.sql), so this insert is atomic:
    # exactly one concurrent caller wins it.
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

    # ponytail: if the winner crashes mid-research, every request for this
    # key pays this one 30s wait before falling back to researching it
    # itself, which also heals the stuck placeholder row via cache_set's
    # upsert. Rare, and self-healing, so not worth a lease/heartbeat.
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
        # Atomic upsert (company_key is unique) — replaces the old
        # select-then-insert-or-update, which could race two concurrent
        # writers into two rows for the same key.
        supabase.table("company_research_cache").upsert(row, on_conflict="company_key").execute()
    except Exception:
        logger.exception("Failed to write company research cache (non-fatal)")


def fail_stale_running_jobs():
    """Jobs left 'running' by a crashed/killed worker are marked failed on startup."""
    # ponytail: assumes workers restart together; use a started_at lease when running many workers
    stale = supabase.table("jobs").update({
        "status": "failed",
        "error": "Worker restarted while the job was running. Please re-process the lead.",
        "gemini_api_key": None,
        "tavily_api_key": None,
    }).eq("status", "running").execute()
    if stale.data:
        logger.warning("Marked %d stale running job(s) as failed", len(stale.data))


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
        .update({"status": "running"})
        .eq("id", job["id"])
        .eq("status", "pending")  # conditional update: loses the race harmlessly
        .execute()
    )
    if not claimed.data:
        return None
    return job


async def run_job(job: dict) -> list:
    leads = job["leads"]
    # pipeline.py does its own {"lead_data": lead} wrapping per kickoff call
    # (it needs the raw dict itself for cache keys / correlation IDs)
    start = time.time()
    scores, emails, agent_times, cache_hits = await process_leads(
        leads, job["gemini_api_key"], job["tavily_api_key"],
        our_company_context=job.get("our_company_context") or "",
        cache_get=cache_get_company,
        cache_set=cache_set_company,
        force_refresh=job.get("force_refresh", False),
    )
    elapsed = round(time.time() - start, 1)
    return persist_results(leads, scores, emails, agent_times, cache_hits, elapsed)


def finish_job(job_id: str, **fields):
    # API keys are wiped as soon as the job leaves the running state
    fields.update({"gemini_api_key": None, "tavily_api_key": None})
    supabase.table("jobs").update(fields).eq("id", job_id).execute()


async def process_one_job(job: dict) -> None:
    """Runs one job to completion; failures here are per-job, not per-process
    — one job's CrewAI error, timeout, or exception never touches the others
    running alongside it."""
    with job_context(job["id"]):
        logger.info("Claimed job (%d lead(s))", len(job["leads"]))
        try:
            results = await run_job(job)
            finish_job(job["id"], status="done", results=results)
            logger.info("Job done")
        except Exception as exc:
            logger.exception("Job failed")
            finish_job(job["id"], status="failed", error=str(exc)[:500])


async def main():
    logger.info(
        "Worker started (poll interval %ds, up to %d job(s) concurrently)",
        POLL_INTERVAL_S, MAX_CONCURRENT_JOBS,
    )
    try:
        fail_stale_running_jobs()
    except Exception:
        logger.exception("Failed to clean up stale jobs on startup")

    in_flight: set = set()
    while True:
        # Only claim as many jobs as we can actually start right now — a job
        # claimed but not yet running would sit at status="running" without
        # being worked on, invisible to other workers that could take it.
        while len(in_flight) < MAX_CONCURRENT_JOBS:
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
            # Wake as soon as a slot frees up (to claim more work sooner) or
            # after the normal poll interval, whichever happens first.
            await asyncio.wait(in_flight, timeout=POLL_INTERVAL_S, return_when=asyncio.FIRST_COMPLETED)


if __name__ == "__main__":
    asyncio.run(main())
