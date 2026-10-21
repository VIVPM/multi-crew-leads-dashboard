"""Network-free proof that worker database calls don't stall the event loop."""

import asyncio
import os
import sys
import threading
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
for name, value in {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_KEY": "test-key",
    "DAILY_LEAD_CAP": "5",
    "LLM_MODEL": "GEMINI",
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "ALLOWED_ORIGINS": "http://localhost:5173",
}.items():
    os.environ.setdefault(name, value)
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""

import worker  # noqa: E402


async def model(*args, **kwargs):
    await asyncio.sleep(0.01)
    return [], [], {}, []


save_barrier = threading.Barrier(4)


def slow_save(*args):
    save_barrier.wait(timeout=3)
    return []


jobs = [{"id": str(i), "leads": [{"id": i}], "gemini_api_key": "key",
         "tavily_api_key": "key", "our_company_context": "test"} for i in range(4)]


async def run_all(fn):
    await asyncio.gather(*(fn(j) for j in jobs))


with patch.object(worker, "process_leads", side_effect=model), patch.object(worker, "persist_results", side_effect=slow_save):
    asyncio.run(run_all(worker.run_job))


finished = []
finish_barrier = threading.Barrier(4)


def slow_finish(job_id, **fields):
    finish_barrier.wait(timeout=3)
    finished.append((job_id, fields["status"]))


async def done(job):
    await asyncio.sleep(0.01)
    return []


with patch.object(worker, "run_job", side_effect=done), patch.object(worker, "finish_job", side_effect=slow_finish):
    asyncio.run(run_all(worker.process_one_job))
assert len(finished) == 4 and all(status == "done" for _, status in finished)


async def check_loop(slow_depth):
    worker._stopping = False
    resume = threading.Event()
    observed = []

    async def heartbeat():
        await asyncio.sleep(0.04)
        resume.set()

    def depth():
        if slow_depth:
            observed.append(resume.wait(timeout=2))
            worker._stopping = True
            return 0
        return 1

    def claim():
        observed.append(resume.wait(timeout=2))
        worker._stopping = True
        return None

    with patch.object(worker, "judge_config"), patch.object(worker, "fail_stale_running_jobs"), \
         patch.object(worker.signal, "signal"), patch.object(worker, "POLL_INTERVAL_S", 0.01), \
         patch.object(worker, "pending_depth", side_effect=depth), \
         patch.object(worker, "claim_next_job", side_effect=claim):
        await asyncio.gather(worker.main(), heartbeat())
    return observed == [True]


assert asyncio.run(check_loop(True)), "depth lookup blocked the event loop"
assert asyncio.run(check_loop(False)), "job claim blocked the event loop"
worker._stopping = False
print("test_worker_io.py: all checks passed")
