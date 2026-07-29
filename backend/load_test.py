"""
load_test.py — how fast can the worker drain the queue, and does it stay
correct when more than one worker is running?

The LLM is stubbed, so this costs nothing and runs in minutes instead of
hours. That is the whole point: at the measured ~52s and ~$0.029 per lead
(--calibrate 5, cache bypassed), draining 400 leads for real would be ~$11
and ~6 hours of wall clock, too slow to re-run after every fix. Everything this test actually
targets — job claiming, concurrency bounding, queue wait, multi-worker
safety — is our code and is unaffected by whether Gemini really answers.

What stays real: the jobs table, claim_next_job's SELECT-then-conditional-
UPDATE across separate processes, MAX_CONCURRENT_JOBS, the worker main loop.
What is stubbed: process_leads (sleeps instead of calling Gemini/Tavily) and
persist_results (so a load run never writes to the leads table).

    python backend/load_test.py --jobs 40 --workers 2
    python backend/load_test.py --cleanup        # if a run died mid-way

Latency is scaled down by default (--lead-seconds 1.0 vs ~106 real). Ratios
hold, wall clock does not — read throughput as "jobs per unit of lead time",
not as leads/hour.
"""

import os
import sys
import math
import json
import statistics
import time
import uuid
import argparse
import asyncio
import subprocess
from typing import Optional
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
from supabase import create_client

TAG_PREFIX = "LOADTEST"
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


# =============================================================================
# Worker mode — a real worker with the LLM boundary stubbed out
# =============================================================================

def worker_mode(lead_seconds: float, start_delay: float = 0) -> None:
    # Delaying inside the worker (rather than between spawns) keeps the
    # coordinator free to start watching at t=0 — otherwise a staggered run
    # only observes the tail of each job and reports impossibly short service
    # times.
    if start_delay:
        time.sleep(start_delay)
    import worker

    async def stub_process_leads(leads, *_a, **_k):
        # Sequential per lead, matching how the real pipeline bills time, so a
        # 10-lead job takes 10x a 1-lead job here too.
        await asyncio.sleep(lead_seconds * len(leads))
        n = len(leads)
        return [None] * n, [None] * n, {}, [False] * n

    def stub_persist_results(leads, *_a, **_k):
        return [{"lead_id": l.get("id"), "stub": True} for l in leads]

    real_claim = worker.claim_next_job

    def counting_claim():
        _t = time.time()
        job = real_claim()
        _dur = time.time() - _t
        if job:
            outcome = "win"
        else:
            # claim_next_job returns None for two very different reasons: it
            # lost the conditional update to another worker, or there was
            # nothing pending at all. Counting both as "lost race" massively
            # overstates contention, since an idle worker polls an empty queue
            # every 3s forever.
            still_pending = (worker.supabase.table("jobs").select("id")
                             .eq("status", "pending").limit(1).execute().data)
            outcome = "lost" if still_pending else "empty"
        # Wall clock (not monotonic) so the coordinator can compare across
        # processes — the first attempt also dates worker boot, which otherwise
        # gets misread as queue contention (importing crewai is slow).
        print(f"LT_CLAIM {outcome} {time.time():.3f} {_dur:.3f}", flush=True)
        return job

    worker.process_leads = stub_process_leads
    worker.persist_results = stub_persist_results
    worker.claim_next_job = counting_claim
    asyncio.run(worker.main())


# =============================================================================
# Coordinator
# =============================================================================

def preflight() -> None:
    """Refuse to run if real work is in the queue.

    Two ways this test would otherwise corrupt real data: the stub would write
    fake results over a real user's job, and every worker calls
    fail_stale_running_jobs() on startup, which marks *every* running job
    failed with no worker filter.
    """
    rows = supabase.table("jobs").select("id,status,our_company_context").in_(
        "status", ["pending", "running"]).execute().data or []
    foreign = [r for r in rows if not (r.get("our_company_context") or "").startswith(TAG_PREFIX)]
    if foreign:
        sys.exit(
            f"ABORT: {len(foreign)} real job(s) are pending/running.\n"
            "This test would overwrite them with stub results, and worker startup\n"
            "would mark them failed. Wait for them to finish, then re-run."
        )


def seed(tag: str, n_jobs: int, leads_per_job: int) -> tuple:
    users = supabase.table("users").select("id").limit(1).execute().data
    if not users:
        sys.exit("No users in the database to attribute load-test jobs to.")
    user_id = users[0]["id"]

    rows = [{
        "user_id": user_id,
        "status": "pending",
        "leads": [{"id": str(uuid.uuid4()), "name": f"Load Test {j}-{i}",
                   "company": f"Loadco {j}", "job_title": "VP Engineering"}
                  for i in range(leads_per_job)],
        "our_company_context": tag,
        "force_refresh": False,
        # real keys are never needed — process_leads is stubbed
        "gemini_api_key": "stub",
        "tavily_api_key": "stub",
    } for j in range(n_jobs)]

    t0 = time.monotonic()
    for i in range(0, len(rows), 50):
        supabase.table("jobs").insert(rows[i:i + 50]).execute()
    print(f"Seeded {n_jobs} jobs x {leads_per_job} leads = {n_jobs * leads_per_job} leads")
    return t0, time.time()


def spawn_workers(n: int, lead_seconds: float, max_concurrent: int, stagger: float = 0) -> list:
    env = dict(os.environ)
    env["MAX_CONCURRENT_JOBS"] = str(max_concurrent)
    # Keep synthetic jobs out of Langfuse/Grafana — otherwise a load run
    # inflates jobs_processed_total and can fire the real alerts. Also cuts
    # worker startup from ~13s to ~1s.
    for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
              "GRAFANA_OTLP_ENDPOINT", "GRAFANA_OTLP_AUTH"):
        env.pop(k, None)

    procs = []
    for idx in range(n):
        procs.append(subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--worker-mode",
             "--lead-seconds", str(lead_seconds),
             "--start-delay", str(idx * stagger)],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1,
        ))
    print(f"Started {n} worker process(es), MAX_CONCURRENT_JOBS={max_concurrent}"
          + (f", staggered {stagger}s apart" if stagger else ""))
    return procs


def watch(tag: str, n_jobs: int, t0: float, timeout_s: float) -> dict:
    """Poll the jobs table and record when each job changes state."""
    running_at, finished_at, status, errors = {}, {}, {}, {}
    peak_running = 0
    deadline = time.monotonic() + timeout_s

    while time.monotonic() < deadline:
        rows = supabase.table("jobs").select("id,status,error").eq(
            "our_company_context", tag).execute().data or []
        now = time.monotonic()
        live = 0
        for r in rows:
            jid, st = r["id"], r["status"]
            status[jid] = st
            if st == "running":
                running_at.setdefault(jid, now)
                live += 1
            elif st in ("done", "failed"):
                running_at.setdefault(jid, now)  # finished between polls
                finished_at.setdefault(jid, now)
                if st == "failed" and r.get("error"):
                    errors[jid] = r["error"]
        peak_running = max(peak_running, live)
        if len(finished_at) >= n_jobs:
            break
        time.sleep(0.5)

    return {
        "wait":    sorted(running_at[j] - t0 for j in running_at),
        "service": sorted(finished_at[j] - running_at[j] for j in finished_at),
        "drain_s": (max(finished_at.values()) - t0) if finished_at else None,
        "done":    sum(1 for s in status.values() if s == "done"),
        "failed":  sum(1 for s in status.values() if s == "failed"),
        "stuck":   n_jobs - len(finished_at),
        "peak_running": peak_running,
        "errors": errors,
    }


def pctl(xs: list, p: float) -> float:
    """Nearest-rank percentile. Small n here, so interpolation would invent
    precision the sample doesn't have."""
    if not xs:
        return float("nan")
    xs = sorted(xs)
    rank = max(1, math.ceil(p / 100 * len(xs)))
    return xs[min(rank, len(xs)) - 1]


def cleanup(tag: Optional[str] = None) -> None:
    q = supabase.table("jobs").delete()
    q = q.eq("our_company_context", tag) if tag else q.like("our_company_context", f"{TAG_PREFIX}%")
    deleted = q.execute().data or []
    print(f"Cleaned up {len(deleted)} load-test job(s)")


def report(m: dict, claims: dict, boots: list, claim_ms: list, args, tag: str) -> None:
    n = args.jobs * args.leads_per_job
    print("\n" + "=" * 74)
    print(f"WORKER DRAIN  ({args.jobs} jobs x {args.leads_per_job} leads, "
          f"{args.workers} worker(s), {args.lead_seconds}s/lead simulated)")
    print("=" * 74)

    if m["drain_s"]:
        print(f"  Drained in         {m['drain_s']:.1f}s "
              f"({m['drain_s'] / args.jobs:.2f}s per job)")
        print(f"  Throughput         {args.jobs / m['drain_s'] * 60:.1f} jobs/min "
              f"| {n / m['drain_s'] * 60:.0f} leads/min")
    print(f"  Completed          {m['done']} done, {m['failed']} failed, {m['stuck']} never finished")
    print(f"  Peak concurrent    {m['peak_running']}  (cap {args.max_concurrent} "
          f"x {args.workers} worker(s) = {args.max_concurrent * args.workers})")

    if boots:
        print(f"\n  Worker boot        {min(boots):.1f}-{max(boots):.1f}s to first claim "
              f"(importing crewai; subtract from queue wait)")
    if m["wait"]:
        # min, not max: a job can be claimed as soon as the FIRST worker is up,
        # so a late-joining worker's boot time is not queueing time.
        contention = pctl(m["wait"], 50) - (min(boots) if boots else 0)
        print(f"  Queue wait         p50 {pctl(m['wait'], 50):6.1f}s   "
              f"p95 {pctl(m['wait'], 95):6.1f}s   max {max(m['wait']):6.1f}s")
        print(f"    of which        ~{contention:.1f}s is post-boot, i.e. actually queueing")
    if m["service"]:
        ideal = args.lead_seconds * args.leads_per_job
        print(f"  Service time       p50 {pctl(m['service'], 50):6.1f}s   "
              f"p95 {pctl(m['service'], 95):6.1f}s   (ideal {ideal:.1f}s)")

    if claim_ms:
        print(f"  Claim round trip   p50 {pctl(claim_ms, 50):6.0f}ms  p95 {pctl(claim_ms, 95):6.0f}ms"
              f"   (SELECT + conditional UPDATE, one job per call)")

    if m.get("errors"):
        from collections import Counter
        print("\n  FAILURE REASONS:")
        for msg, c in Counter(m["errors"].values()).most_common():
            print(f"    {c:3}x  {msg[:88]}")

    # Only win/lost say anything about contention. Polls of an empty queue are
    # just an idle worker ticking every POLL_INTERVAL_S and would swamp the
    # ratio if counted.
    contested = claims["win"] + claims["lost"]
    if contested:
        print(f"\n  Claim attempts     {claims['win']} won, {claims['lost']} lost to another "
              f"worker, {claims['empty']} on an empty queue")
        if claims["lost"]:
            print(f"    -> {claims['lost'] / contested * 100:.0f}% of contested claims lost and slept 3s instead")
            print(f"       of taking the next job (claim_next_job only ever looks at the")
            print(f"       single oldest pending row).")

    out_dir = os.path.join(os.path.dirname(BASE_DIR), "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"drain_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(), "tag": tag,
                   "config": vars(args), "claims": claims,
                   "worker_boot_s": boots,
                   "claim_ms_p50": pctl(claim_ms, 50), "claim_ms_p95": pctl(claim_ms, 95),
                   "drain_s": m["drain_s"], "done": m["done"], "failed": m["failed"],
                   "stuck": m["stuck"], "peak_running": m["peak_running"],
                   "queue_wait_p50": pctl(m["wait"], 50), "queue_wait_p95": pctl(m["wait"], 95),
                   "service_p50": pctl(m["service"], 50), "service_p95": pctl(m["service"], 95)},
                  f, indent=2)
    print(f"\n  Report: {path}")
    print("  Simulated lead latency — treat throughput as relative, not absolute.")


def calibrate(n_leads: int, base: str) -> None:
    """Run real leads through the real pipeline to check the stub's timing model.

    Tests A and B both assume a job costs lead_seconds * lead_count, based on
    ~106s/lead from stored analysis runs. _process_batch scores leads in a plain
    for-loop (pipeline.py:338), so the shape is right by construction; this
    checks the magnitude, which is the part that could drift with prompt or
    model changes. force_refresh bypasses the company cache so the number is a
    clean worst case rather than a function of what happened to be cached.

    This is the only part of load testing that spends money: n_leads * ~$0.019.
    """
    import httpx
    from load_test_api import LOAD_USER, ensure_user

    # Real companies, all distinct, so no two leads share cached research.
    companies = [
        ("Twilio", "Director of Revenue Operations"),
        ("Snowflake", "VP Sales Operations"),
        ("Atlassian", "Head of Growth"),
        ("Okta", "Director of Demand Generation"),
        ("Datadog", "VP Marketing"),
        ("Zoom", "Director of Sales Enablement"),
        ("Cloudflare", "Head of Revenue Operations"),
        ("Twist Bioscience", "VP Commercial Operations"),
        ("Grammarly", "Director of Growth Marketing"),
        ("Notion", "Head of Sales Operations"),
    ][:n_leads]

    donor = (supabase.table("users").select("company_context")
             .not_.is_("company_context", "null").limit(1).execute().data or [])
    if not donor:
        sys.exit("No user has a company_context/ICP saved — set one in the UI first.")
    token, _user_id = ensure_user(base)
    supabase.table("users").update(
        {"company_context": donor[0]["company_context"]}).eq("username", LOAD_USER).execute()

    auth = {"Authorization": f"Bearer {token}"}
    lead_ids = []
    for company, title in companies:
        r = httpx.post(f"{base}/leads", headers=auth, timeout=30, json={
            "name": f"Calibration {company}", "company": company, "job_title": title,
            "email": f"calibration@{company.split()[0].lower()}.example",
            "use_case": "Evaluating tooling for outbound lead qualification",
            "industry": "Technology", "location": "United States", "source": "Website",
        })
        r.raise_for_status()
        lead_ids.append(r.json()["id"])  # create_lead returns the inserted row

    leads_payload = [{"id": i} for i in lead_ids if i]
    print(f"Created {len(leads_payload)} leads; processing for real (~${0.019 * len(leads_payload):.2f})...")

    t0 = time.time()
    r = httpx.post(f"{base}/leads/process", headers=auth, timeout=60,
                   json={"leads": leads_payload, "force_refresh": True})
    r.raise_for_status()
    job_id = r.json()["job_id"]

    status, last = "pending", 0.0
    while time.time() - t0 < 3600:
        time.sleep(10)
        j = httpx.get(f"{base}/jobs/{job_id}", headers=auth, timeout=30).json()
        status = j.get("status", "?")
        if time.time() - t0 - last > 60:
            last = time.time() - t0
            print(f"  {last / 60:.0f} min elapsed, status={status}", flush=True)
        if status in ("done", "failed"):
            break
    wall = time.time() - t0

    runs = (supabase.table("analysis_runs").select("lead_id,duration_seconds,total_tokens,total_cost")
            .in_("lead_id", [i for i in lead_ids if i]).execute().data or [])
    per_lead = [r["duration_seconds"] for r in runs if r.get("duration_seconds")]
    costs = [r["total_cost"] for r in runs if r.get("total_cost")]

    print("\n" + "=" * 74)
    print(f"REAL-COST CALIBRATION  ({len(leads_payload)} leads, one job, cache bypassed)")
    print("=" * 74)
    print(f"  Job status         {status}")
    print(f"  Wall clock         {wall:.0f}s ({wall / 60:.1f} min)")
    n = len(leads_payload)
    # analysis_runs.duration_seconds is the whole job's elapsed time written
    # onto every lead (backend.py:742), not a per-lead measurement — so divide,
    # never average. Single-lead jobs from the UI make the two look identical,
    # which is exactly how that gets misread.
    pipeline_s = per_lead[0] if per_lead else wall
    print(f"  Pipeline time      {pipeline_s:.0f}s for {n} leads "
          f"-> {pipeline_s / n:.0f}s per lead")
    print(f"  (analysis_runs.duration_seconds is per-job, duplicated across leads)")
    if costs:
        print(f"  Cost               ${sum(costs):.4f} total, ${sum(costs) / n:.4f}/lead (cache bypassed)")
    print(f"\n  Feed {pipeline_s / n:.0f} to --lead-seconds for an unscaled run; the scaled")
    print(f"  default only preserves ratios.")

    out_dir = os.path.join(os.path.dirname(BASE_DIR), "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"calibration_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.now(timezone.utc).isoformat(),
                   "n_leads": len(leads_payload), "status": status, "wall_s": round(wall, 1),
                   "per_lead_s": per_lead, "costs": costs,
                   "median_per_lead_s": statistics.median(per_lead) if per_lead else None,
                   "total_cost": sum(costs) if costs else None}, f, indent=2)
    print(f"\n  Report: {path}")

    for lid in lead_ids:
        if lid:
            httpx.delete(f"{base}/leads/{lid}", headers=auth, timeout=30)
    print(f"  Cleaned up {len(lead_ids)} calibration lead(s)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--leads-per-job", type=int, default=10)
    ap.add_argument("--lead-seconds", type=float, default=1.0)
    ap.add_argument("--max-concurrent", type=int, default=10)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--stagger", type=float, default=0,
                    help="seconds between worker starts (exposes startup races)")
    ap.add_argument("--cleanup", action="store_true")
    ap.add_argument("--calibrate", type=int, metavar="N",
                    help="run N real leads through the real pipeline (costs ~$0.019/lead)")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--worker-mode", action="store_true")
    ap.add_argument("--start-delay", type=float, default=0)
    args = ap.parse_args()

    if args.worker_mode:
        return worker_mode(args.lead_seconds, args.start_delay)
    if args.cleanup:
        return cleanup()
    if args.calibrate:
        return calibrate(args.calibrate, args.base)

    preflight()
    tag = f"{TAG_PREFIX}-{uuid.uuid4().hex[:8]}"
    procs = []
    try:
        t0, wall0 = seed(tag, args.jobs, args.leads_per_job)
        procs = spawn_workers(args.workers, args.lead_seconds, args.max_concurrent, args.stagger)
        m = watch(tag, args.jobs, t0, args.timeout)

        claims = {"win": 0, "lost": 0, "empty": 0}
        boots, claim_ms = [], []
        for p in procs:
            p.terminate()
            try:
                out, _ = p.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                out = ""
            first = None
            for line in (out or "").splitlines():
                if not line.startswith("LT_CLAIM "):
                    continue
                parts = line.split()
                claims[parts[1]] = claims.get(parts[1], 0) + 1
                if first is None and len(parts) > 2:
                    first = float(parts[2])
                if len(parts) > 3:
                    claim_ms.append(float(parts[3]) * 1000)
            if first is not None:
                boots.append(first - wall0)

        report(m, claims, boots, claim_ms, args, tag)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
        cleanup(tag)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
        assert pctl([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95) == 10
        assert pctl([5], 50) == 5
        assert math.isnan(pctl([], 50))
        print("selftest ok")
    else:
        main()
