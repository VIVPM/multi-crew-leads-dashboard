"""
load_test_api.py — does the API stay responsive while the worker is saturated?

Test B (load_test.py) measured the queue. This measures the other half: the
user-facing API. The question that matters is not "how many requests per
second can /leads/process absorb" — that endpoint just inserts a row, so it
would report a huge and meaningless number. It is whether logging in and
browsing leads stays fast while the pipeline is flat out.

That risk is real on this deployment. RUN_WORKER_IN_PROCESS=1 puts the worker
in the API process (backend.py:736). It gets its own thread and its own event
loop, so it cannot block uvicorn's loop directly, but it still shares a GIL and
a thread pool with every sync `def` endpoint in backend.py. Whether that is
survivable is an empirical question, so:

    idle phase      -> hammer the API with an empty queue
    saturated phase -> hammer it again with the worker at full concurrency

and compare. Same LLM stub as Test B, so it costs nothing.

    python backend/load_test_api.py
    python backend/load_test_api.py --no-in-process-worker   # separate-worker deploy

Latency is scaled (--lead-seconds), so read the idle-vs-saturated *ratio*, not
absolute milliseconds.
"""

import os
import sys
import time
import json
import uuid
import asyncio
import argparse
import subprocess
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

import httpx
from supabase import create_client

from load_test import TAG_PREFIX, pctl, preflight

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

LOAD_USER = "loadtest@local"
LOAD_PASS = "loadtest-pw-9137"


# =============================================================================
# Serve mode — the real API, with the LLM stubbed
# =============================================================================

def serve_mode(port: int, lead_seconds: float, in_process_worker: bool) -> None:
    """Run the real FastAPI app, stubbing only the LLM call.

    Patch order matters. worker.py binds `process_leads` and `persist_results`
    by value at import time, and backend.py imports worker lazily inside its
    startup hook — so patching both modules before importing backend is what
    makes the worker thread pick up the stubs.
    """
    os.environ["RUN_WORKER_IN_PROCESS"] = "1" if in_process_worker else "0"

    import pipeline

    async def stub_process_leads(leads, *_a, **_k):
        await asyncio.sleep(lead_seconds * len(leads))
        n = len(leads)
        return [None] * n, [None] * n, {}, [False] * n

    pipeline.process_leads = stub_process_leads

    import backend as backend_mod
    backend_mod.persist_results = lambda leads, *_a, **_k: [
        {"lead_id": lead.get("id"), "stub": True} for lead in leads
    ]

    import uvicorn
    uvicorn.run(backend_mod.app, host="127.0.0.1", port=port, log_level="error")


# =============================================================================
# Client
# =============================================================================

async def _hammer(base: str, token: str, user_id: str, job_id: str,
                  concurrency: int, duration: float) -> dict:
    """Fire the read mix a real browser makes, and record every latency."""
    results: dict = {"health": [], "leads": [], "job": [], "login": []}
    errors = {"count": 0, "samples": []}
    stop = time.monotonic() + duration
    auth = {"Authorization": f"Bearer {token}"}

    async def one_client(client: httpx.AsyncClient) -> None:
        while time.monotonic() < stop:
            for name, method, url, kwargs in (
                ("health", "GET", "/", {}),
                ("leads", "GET", f"/leads/{user_id}", {"headers": auth}),
                ("job", "GET", f"/jobs/{job_id}", {"headers": auth}),
                ("login", "POST", "/auth/login",
                 {"json": {"username": LOAD_USER, "password": LOAD_PASS}}),
            ):
                t = time.monotonic()
                try:
                    r = await client.request(method, url, timeout=30, **kwargs)
                    dt = (time.monotonic() - t) * 1000
                    if r.status_code >= 400:
                        errors["count"] += 1
                        if len(errors["samples"]) < 5:
                            errors["samples"].append(f"{name} {r.status_code} {r.text[:70]}")
                    else:
                        results[name].append(dt)
                except Exception as exc:
                    errors["count"] += 1
                    if len(errors["samples"]) < 5:
                        errors["samples"].append(f"{name} {type(exc).__name__}")
                if time.monotonic() >= stop:
                    return

    async with httpx.AsyncClient(base_url=base) as client:
        await asyncio.gather(*[one_client(client) for _ in range(concurrency)])

    return {"lat": results, "errors": errors}


def wait_for_health(base: str, timeout: float = 90) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{base}/", timeout=3).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ensure_user(base: str) -> tuple:
    s = httpx.post(f"{base}/auth/signup",
                   json={"username": LOAD_USER, "password": LOAD_PASS}, timeout=30)
    r = httpx.post(f"{base}/auth/login",
                   json={"username": LOAD_USER, "password": LOAD_PASS}, timeout=30)
    if r.status_code != 200:
        sys.exit(f"Could not authenticate the load-test user.\n"
                 f"  signup -> {s.status_code} {s.text[:200]}\n"
                 f"  login  -> {r.status_code} {r.text[:200]}")
    d = r.json()
    return d["token"], d["user_id"]


def seed_jobs(tag: str, user_id: str, n_jobs: int, leads_per_job: int) -> None:
    rows = [{
        "user_id": user_id, "status": "pending",
        "leads": [{"id": str(uuid.uuid4()), "name": f"API Load {j}-{i}",
                   "company": f"Loadco {j}", "job_title": "VP Engineering"}
                  for i in range(leads_per_job)],
        "our_company_context": tag, "force_refresh": False,
        "gemini_api_key": "stub", "tavily_api_key": "stub",
    } for j in range(n_jobs)]
    for i in range(0, len(rows), 50):
        supabase.table("jobs").insert(rows[i:i + 50]).execute()


def wait_until_running(tag: str, want: int, timeout: float = 120) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = supabase.table("jobs").select("status").eq(
            "our_company_context", tag).execute().data or []
        running = sum(1 for r in rows if r["status"] == "running")
        if running >= want:
            return running
        time.sleep(1)
    return running


def _row(label: str, idle: list, sat: list) -> str:
    if not idle or not sat:
        return f"  {label:10} {'no data':>52}"
    i50, i95 = pctl(idle, 50), pctl(idle, 95)
    s50, s95 = pctl(sat, 50), pctl(sat, 95)
    ratio = s95 / i95 if i95 else float("nan")
    flag = "  <-- degraded" if ratio >= 2 else ""
    return (f"  {label:10} p50 {i50:7.0f} -> {s50:7.0f}ms   "
            f"p95 {i95:7.0f} -> {s95:7.0f}ms   x{ratio:.1f}{flag}")


def report(idle: dict, sat: dict, args, running: int, tag: str) -> None:
    print("\n" + "=" * 78)
    print(f"API UNDER LOAD  (worker {'in-process' if args.in_process_worker else 'separate'}, "
          f"{args.concurrency} clients, {running} jobs running)")
    print("=" * 78)
    print(f"  {'':10} {'idle':>21}   {'saturated':>24}")
    for key, label in (("health", "GET /"), ("leads", "GET /leads"),
                       ("job", "GET /jobs"), ("login", "POST /login")):
        print(_row(label, idle["lat"][key], sat["lat"][key]))

    n_idle = sum(len(v) for v in idle["lat"].values())
    n_sat = sum(len(v) for v in sat["lat"].values())
    print(f"\n  Requests           {n_idle} idle, {n_sat} saturated")
    print(f"  Throughput         {n_idle / args.duration:.0f} -> {n_sat / args.duration:.0f} req/s")
    print(f"  Errors             {idle['errors']['count']} idle, {sat['errors']['count']} saturated")
    for s in (idle["errors"]["samples"] + sat["errors"]["samples"])[:5]:
        print(f"    {s}")

    out_dir = os.path.join(ROOT_DIR, "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"api_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(), "tag": tag,
            "config": vars(args), "jobs_running": running,
            "idle": {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "n": len(v)}
                     for k, v in idle["lat"].items()},
            "saturated": {k: {"p50": pctl(v, 50), "p95": pctl(v, 95), "n": len(v)}
                          for k, v in sat["lat"].items()},
            "errors": {"idle": idle["errors"]["count"], "saturated": sat["errors"]["count"]},
        }, f, indent=2)
    print(f"\n  Report: {path}")
    print("  Simulated lead latency — compare the ratio, not absolute ms.")


def cleanup(tag: str) -> None:
    deleted = supabase.table("jobs").delete().eq("our_company_context", tag).execute().data or []
    print(f"Cleaned up {len(deleted)} load-test job(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--duration", type=float, default=20)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--leads-per-job", type=int, default=10)
    ap.add_argument("--lead-seconds", type=float, default=6.0)
    ap.add_argument("--no-in-process-worker", dest="in_process_worker",
                    action="store_false", default=True)
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()

    if args.serve:
        return serve_mode(args.port, args.lead_seconds, args.in_process_worker)

    preflight()
    base = f"http://127.0.0.1:{args.port}"
    tag = f"{TAG_PREFIX}-api-{uuid.uuid4().hex[:8]}"

    env = dict(os.environ)
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "GRAFANA_OTLP_ENDPOINT", "GRAFANA_OTLP_AUTH"):
        env.pop(k, None)
    cmd = [sys.executable, os.path.abspath(__file__), "--serve",
           "--port", str(args.port), "--lead-seconds", str(args.lead_seconds)]
    if not args.in_process_worker:
        cmd.append("--no-in-process-worker")

    os.makedirs(os.path.join(ROOT_DIR, "load_test_results"), exist_ok=True)
    api_log_path = os.path.join(ROOT_DIR, "load_test_results", "api_server.log")
    api_log = open(api_log_path, "w", encoding="utf-8")
    api = subprocess.Popen(cmd, env=env, cwd=ROOT_DIR,
                           stdout=api_log, stderr=subprocess.STDOUT)
    running = 0
    try:
        if not wait_for_health(base):
            sys.exit("API did not come up — run with stderr visible to see why.")
        print(f"API up on {base} (worker {'in-process' if args.in_process_worker else 'separate'})")

        token, user_id = ensure_user(base)
        seed_jobs(tag, user_id, 1, 1)  # one throwaway job so GET /jobs/{id} is a real lookup
        probe = (supabase.table("jobs").select("id").eq("our_company_context", tag)
                 .limit(1).execute().data or [{}])[0].get("id", str(uuid.uuid4()))

        print(f"Phase 1/2: idle, {args.concurrency} clients for {args.duration}s...")
        idle = asyncio.run(_hammer(base, token, user_id, probe, args.concurrency, args.duration))

        print(f"Seeding {args.jobs} jobs and waiting for the worker to pick them up...")
        seed_jobs(tag, user_id, args.jobs, args.leads_per_job)
        running = wait_until_running(tag, min(args.jobs, 5))
        print(f"Phase 2/2: saturated ({running} jobs running), {args.duration}s...")
        sat = asyncio.run(_hammer(base, token, user_id, probe, args.concurrency, args.duration))

        report(idle, sat, args, running, tag)
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()
        api_log.close()
        print(f"  Server log: {api_log_path}")
        cleanup(tag)


if __name__ == "__main__":
    main()
