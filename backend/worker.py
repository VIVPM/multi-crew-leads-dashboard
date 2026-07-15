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

logger = logging.getLogger("worker")

POLL_INTERVAL_S = 3


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


def run_job(job: dict) -> list:
    leads = job["leads"]
    raw_inputs = [{"lead_data": lead} for lead in leads]
    start = time.time()
    scores, emails, agent_times = asyncio.run(
        process_leads(raw_inputs, job["gemini_api_key"], job["tavily_api_key"])
    )
    elapsed = round(time.time() - start, 1)
    return persist_results(leads, scores, emails, agent_times, elapsed)


def finish_job(job_id: str, **fields):
    # API keys are wiped as soon as the job leaves the running state
    fields.update({"gemini_api_key": None, "tavily_api_key": None})
    supabase.table("jobs").update(fields).eq("id", job_id).execute()


def main():
    logger.info("Worker started (poll interval %ds)", POLL_INTERVAL_S)
    try:
        fail_stale_running_jobs()
    except Exception:
        logger.exception("Failed to clean up stale jobs on startup")
    while True:
        try:
            job = claim_next_job()
        except Exception:
            # transient DB/network error — keep the loop alive
            logger.exception("Failed to poll jobs table; retrying")
            time.sleep(POLL_INTERVAL_S)
            continue
        if job is None:
            time.sleep(POLL_INTERVAL_S)
            continue
        logger.info("Claimed job %s (%d lead(s))", job["id"], len(job["leads"]))
        try:
            results = run_job(job)
            finish_job(job["id"], status="done", results=results)
            logger.info("Job %s done", job["id"])
        except Exception as exc:
            logger.exception("Job %s failed", job["id"])
            finish_job(job["id"], status="failed", error=str(exc)[:500])


if __name__ == "__main__":
    main()
