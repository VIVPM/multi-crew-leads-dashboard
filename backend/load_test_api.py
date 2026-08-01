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
    # httpx defaults to max_connections=100 — below that silently caps what
    # this function can actually send, so a "500 concurrent" run would really
    # measure the client's own pool, not the server. Scale it to concurrency.
    limits = httpx.Limits(max_connections=concurrency + 20, max_keepalive_connections=concurrency)

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

    async with httpx.AsyncClient(base_url=base, limits=limits) as client:
        await asyncio.gather(*[one_client(client) for _ in range(concurrency)])

    return {"lat": results, "errors": errors}


RAMP_LEVELS = [10, 25, 50, 100, 200, 300, 400, 500]


def _agg(lat: dict, keys: tuple) -> list:
    out = []
    for k in keys:
        out.extend(lat.get(k, []))
    return out


def run_ramp(base: str, token: str, user_id: str, job_id: str,
             levels: list, duration: float) -> list:
    """Step concurrency up and watch where it breaks — this is the answer to
    "how many concurrent users can use this without breaking".

    Runs against an IDLE queue on purpose (worker forced off in main()): this
    isolates one variable, the sync-def-endpoint concurrency ceiling itself
    (backend.py's routes are all `def`, so FastAPI runs each in a worker
    thread via anyio.to_thread.run_sync, which defaults to a hardcoded
    CapacityLimiter(40) — verified in anyio 4.12.1's source, not something
    this repo set). Whether the worker being busy makes things worse is the
    separate idle-vs-saturated test above, at one fixed concurrency.

    "health"/"leads"/"job" are cheap reads (~one Supabase round trip) and are
    combined into one "fast" figure; "login" is reported separately since
    bcrypt makes it slow by design even with zero contention, and mixing it
    in would make the fast-path number look worse than it is.
    """
    rows = []
    baseline_p95 = None
    for level in levels:
        print(f"  concurrency {level:4} ...", end="", flush=True)
        res = asyncio.run(_hammer(base, token, user_id, job_id, level, duration))
        fast = _agg(res["lat"], ("health", "leads", "job"))
        login = res["lat"]["login"]
        n_ok = sum(len(v) for v in res["lat"].values())
        n_err = res["errors"]["count"]
        n_total = n_ok + n_err
        err_pct = (n_err / n_total * 100) if n_total else 0.0
        p50, p95 = pctl(fast, 50), pctl(fast, 95)
        lp95 = pctl(login, 95)
        if baseline_p95 is None and fast:
            baseline_p95 = p95
        degraded = bool(baseline_p95) and p95 > 3 * baseline_p95
        flag = "  <-- ERRORS" if err_pct > 1 else ("  <-- latency degrading" if degraded else "")
        print(f"\r  {level:4} users   {n_total:5} req   {err_pct:5.1f}% err   "
              f"fast p50 {p50:6.0f}ms p95 {p95:6.0f}ms   login p95 {lp95:6.0f}ms{flag}")
        rows.append({
            "concurrency": level, "requests": n_total, "errors": n_err,
            "error_pct": round(err_pct, 1), "fast_p50_ms": p50, "fast_p95_ms": p95,
            "login_p95_ms": lp95, "req_per_s": round(n_total / duration, 1),
        })
        if err_pct > 25:
            print(f"  Error rate over 25% at {level} concurrent users — stopping the ramp here.")
            break
    return rows


def ramp_report(rows: list, args, tag: str) -> None:
    print("\n" + "=" * 78)
    print(f"CONCURRENCY RAMP  ({args.ramp_duration:.0f}s per level, worker idle/off — "
          f"isolates the API's own ceiling)")
    print("=" * 78)

    baseline = rows[0]["fast_p95_ms"] if rows else float("nan")
    healthy = [r for r in rows if r["error_pct"] < 1 and r["fast_p95_ms"] < 3 * baseline]
    ceiling = healthy[-1]["concurrency"] if healthy else 0
    print(f"\n  Estimated healthy ceiling: ~{ceiling} concurrent users")
    print(f"  SLO: <1% errors AND fast-path p95 < 3x the {rows[0]['concurrency'] if rows else '?'}-user baseline")
    print("  This measured the sync-def-endpoint / anyio-threadpool ceiling on this")
    print("  machine — a different box (more CPU/cores) or Render's actual instance")
    print("  size will shift the number; re-run there before trusting it as final.")

    out_dir = os.path.join(ROOT_DIR, "load_test_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ramp_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(), "tag": tag,
            "config": vars(args), "levels": rows, "estimated_ceiling": ceiling,
        }, f, indent=2)
    print(f"\n  Report: {path}")
    print("  LLM fully stubbed (pipeline.process_leads never runs) — zero Gemini/Tavily calls, $0.")


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


def insert_done_probe(tag: str, user_id: str) -> str:
    """A GET /jobs/{id} target for a --base-url run, safe against a real
    deployment: inserted directly as already-'done', never 'pending', so no
    worker anywhere (local or on the deployment, since they share one
    Supabase project) can ever pick it up and spend real money on it."""
    row = {"user_id": user_id, "status": "done", "leads": [],
           "our_company_context": tag, "force_refresh": False}
    return supabase.table("jobs").insert(row).execute().data[0]["id"]


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
    ap.add_argument("--ramp", action="store_true",
                    help="step concurrency up through --ramp-levels to find where it breaks, "
                         "instead of the fixed-concurrency idle/saturated test")
    ap.add_argument("--ramp-levels", default=",".join(str(n) for n in RAMP_LEVELS))
    ap.add_argument("--ramp-duration", type=float, default=8)
    ap.add_argument("--base-url", default=None,
                    help="hit an already-running server (e.g. a Render deployment) instead of "
                         "spawning a local stub. Only valid with --ramp: the idle/saturated test's "
                         "Phase 2 seeds a real 'pending' job, which a real worker on a live "
                         "deployment could pick up and actually spend money on.")
    args = ap.parse_args()
    if args.ramp:
        # A different question than idle-vs-saturated (worker busy or not) —
        # keep the worker out of it so the ramp measures one variable.
        args.in_process_worker = False

    if args.serve:
        return serve_mode(args.port, args.lead_seconds, args.in_process_worker)

    if args.base_url and not args.ramp:
        sys.exit("--base-url only supports --ramp — the idle/saturated test's Phase 2 seeds a "
                 "real 'pending' job, unsafe against a live deployment with a real worker.")

    preflight()  # DB-level check either way: refuses to run if real work is queued
    tag = f"{TAG_PREFIX}-api-{uuid.uuid4().hex[:8]}"

    api = None
    api_log = None
    if args.base_url:
        base = args.base_url.rstrip("/")
    else:
        base = f"http://127.0.0.1:{args.port}"
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
            sys.exit(f"Could not reach {base} — "
                     + ("check the URL is right and reachable." if args.base_url
                        else "run with stderr visible to see why the local server didn't come up."))
        print(f"API up on {base}"
              + ("" if args.base_url else f" (worker {'in-process' if args.in_process_worker else 'separate'})"))

        token, user_id = ensure_user(base)
        if args.base_url:
            # Never seed a 'pending' job against a live deployment — see the
            # safety note on --base-url above. This row is inert by construction.
            probe = insert_done_probe(tag, user_id)
        else:
            seed_jobs(tag, user_id, 1, 1)  # one throwaway job so GET /jobs/{id} is a real lookup
            probe = (supabase.table("jobs").select("id").eq("our_company_context", tag)
                     .limit(1).execute().data or [{}])[0].get("id", str(uuid.uuid4()))

        if args.ramp:
            levels = [int(x) for x in args.ramp_levels.split(",") if x.strip()]
            print(f"Ramping concurrency through {levels}, {args.ramp_duration:.0f}s each "
                  f"(worker off — isolating the API's own ceiling)...")
            rows = run_ramp(base, token, user_id, probe, levels, args.ramp_duration)
            ramp_report(rows, args, tag)
        else:
            print(f"Phase 1/2: idle, {args.concurrency} clients for {args.duration}s...")
            idle = asyncio.run(_hammer(base, token, user_id, probe, args.concurrency, args.duration))

            print(f"Seeding {args.jobs} jobs and waiting for the worker to pick them up...")
            seed_jobs(tag, user_id, args.jobs, args.leads_per_job)
            running = wait_until_running(tag, min(args.jobs, 5))
            print(f"Phase 2/2: saturated ({running} jobs running), {args.duration}s...")
            sat = asyncio.run(_hammer(base, token, user_id, probe, args.concurrency, args.duration))

            report(idle, sat, args, running, tag)
    finally:
        if api is not None:
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
