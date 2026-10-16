"""
load_test_k6.py — step realistic k6 users through load levels and measure the API and the job queue separately.

Starts the stubbed local API from load_test_api.py (real FastAPI, real Supabase,
real in-process worker; the model call is a sleep and the queued keys are fake,
so it costs nothing). Test accounts are inserted directly, because signup is
rate-limited per IP. For each level, k6 runs load_test_k6.js and reports API
p50/p95/p99 and errors per endpoint; meanwhile this script polls the jobs table
every POLL_S seconds and reports queue wait, run time and failures, because k6
only sees the request that queues a job, not the job finishing. Each level's
queue drains before the next starts. Everything the run created is deleted.

    python backend/load_test_k6.py                        # 10..150 users, 3 min each
    python backend/load_test_k6.py --levels 5 --hold 30   # smoke run
    python backend/load_test_k6.py --spike 10,150

Read absolute numbers as this machine with a --lead-seconds stub, not production.
"""

import os
import sys
import json
import time
import shutil
import argparse
import threading
import subprocess
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from load_test import TAG_PREFIX, pctl, preflight  # noqa: E402
from load_test_api import supabase, wait_for_health, seed_leads  # noqa: E402
from security import hash_password  # noqa: E402

OUT_DIR = os.path.join(ROOT_DIR, "results", "load_test")
PASSWORD = "k6-load-pw-4821"
POLL_S = 2
JOB_LIMITS_S = {"p50": 5, "p95": 30, "p99": 60}
ENDPOINTS = ("login", "leads", "credits", "company_context", "create_lead", "process", "job_status")
PHASE_ENDPOINTS = ("login", "leads", "job_status", "process")


def spike_plan(base_users: int, peak_users: int) -> tuple:
    """k6 stages and phase boundaries for a spike: settle at base, jump to peak in 10 s, hold, drop, watch recovery."""
    stages = [{"seconds": 30, "target": base_users}, {"seconds": 60, "target": base_users},
              {"seconds": 10, "target": peak_users}, {"seconds": 60, "target": peak_users},
              {"seconds": 10, "target": base_users}, {"seconds": 180, "target": base_users}]
    phases = [{"name": "before", "end_s": 90}, {"name": "spike", "end_s": 160}, {"name": "after", "end_s": 350}]
    return stages, phases


def phase_stats(summary: dict, phases: list) -> dict:
    """p50/p95/p99 per spike phase for the endpoints that show a spike: login, leads, job status, submit."""
    out = {}
    for ph in phases:
        out[ph["name"]] = {}
        for e in PHASE_ENDPOINTS:
            v = summary["metrics"].get(f"phase_{ph['name']}_{e}", {}).get("values", {})
            if v.get("count"):
                out[ph["name"]][e] = {"count": int(v["count"]), "p50": round(v["p(50)"]),
                                      "p95": round(v["p(95)"]), "p99": round(v["p(99)"])}
    return out


def find_k6() -> str:
    """k6 from PATH, or the default winget install location for a shell opened before the install."""
    path = shutil.which("k6") or r"C:\Program Files\k6\k6.exe"
    if not os.path.exists(path):
        sys.exit("k6 not found. Install it: winget install k6 --source winget")
    return path


def create_accounts(tag: str, n: int, seed: int) -> list:
    """Insert n users with an ICP and seeded leads; the ICP starts with TAG_PREFIX so preflight knows the jobs."""
    hashed = hash_password(PASSWORD)
    rows = [{"username": f"{tag}-{i}", "password": hashed,
             "company_context": f"{TAG_PREFIX} k6 ICP: B2B software teams of 50-5000 people."}
            for i in range(n)]
    users = supabase.table("users").insert(rows).execute().data
    for u in users:
        seed_leads(str(u["id"]), seed)
    return users


def cleanup(users: list) -> None:
    ids = [str(u["id"]) for u in users]
    if not ids:
        return
    supabase.table("jobs").delete().in_("user_id", ids).execute()
    supabase.table("leads").delete().in_("user_id", ids).execute()
    supabase.table("refresh_tokens").delete().in_("user_id", ids).execute()
    supabase.table("audit_events").delete().in_("actor_user_id", ids).execute()
    supabase.table("users").delete().in_("id", ids).execute()
    print(f"Cleaned up {len(ids)} test account(s) with their leads, jobs and sessions")


class JobWatcher:
    """Polls the test accounts' jobs and records when each one is first seen finished."""

    def __init__(self, user_ids: list):
        self.user_ids = user_ids
        self.jobs: dict = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rows = (supabase.table("jobs").select("id,status,created_at,started_at")
                        .in_("user_id", self.user_ids).execute().data) or []
            except Exception:
                rows = []
            now = datetime.now(timezone.utc)
            for r in rows:
                seen = self.jobs.setdefault(r["id"], {"finished": None})
                seen.update(r)
                if r["status"] in ("done", "failed") and seen["finished"] is None:
                    seen["finished"] = now
            self._stop.wait(POLL_S)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def created_since(self, since: datetime) -> list:
        return [j for j in self.jobs.values() if _ts(j["created_at"]) >= since]


def _ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def job_stats(jobs: list) -> dict:
    wait = [(_ts(j["started_at"]) - _ts(j["created_at"])).total_seconds() for j in jobs if j.get("started_at")]
    run = [(j["finished"] - _ts(j["started_at"])).total_seconds()
           for j in jobs if j["finished"] and j.get("started_at")]
    total = [(j["finished"] - _ts(j["created_at"])).total_seconds() for j in jobs if j["finished"]]
    stats = {"submitted": len(jobs),
             "done": sum(j["status"] == "done" for j in jobs),
             "failed": sum(j["status"] == "failed" for j in jobs),
             "unfinished": sum(j["status"] in ("pending", "running") for j in jobs)}
    for name, xs in (("queue_wait_s", wait), ("run_s", run), ("total_s", total)):
        stats[name] = {p: round(pctl(xs, int(p[1:])), 1) if xs else None for p in ("p50", "p95", "p99")}
    return stats


def api_stats(summary: dict) -> dict:
    metrics = summary["metrics"]
    out = {}
    for e in ENDPOINTS:
        lat = metrics.get(f"api_{e}", {}).get("values", {})
        err = metrics.get(f"err_{e}", {}).get("values", {})
        if lat.get("count"):
            out[e] = {"count": int(lat["count"]), "p50": round(lat["p(50)"]), "p95": round(lat["p(95)"]),
                      "p99": round(lat["p(99)"]), "error_pct": round(100 * err.get("rate", 0), 2)}
    failed = metrics.get("http_req_failed", {}).get("values", {}).get("rate", 0)
    return {"endpoints": out, "error_pct": round(100 * failed, 2),
            "requests": int(metrics.get("http_reqs", {}).get("values", {}).get("count", 0)),
            "thresholds_failed": sorted(name for name, m in metrics.items()
                                        if any(not t["ok"] for t in m.get("thresholds", {}).values()))}


def run_level(k6: str, base: str, vus: int, args, accounts_path: str, watcher: JobWatcher) -> dict:
    summary_path = os.path.join(OUT_DIR, f"k6_summary_{vus}.json")
    started = datetime.now(timezone.utc)
    env = dict(os.environ, BASE=base, VUS=str(vus), RAMP_S=str(args.ramp_up), HOLD_S=str(args.hold),
               MAX_JOBS_PER_USER=str(args.jobs_per_user), ACCOUNTS=accounts_path, SUMMARY=summary_path)
    phases = []
    if args.spike:
        base_users, peak_users = (int(x) for x in args.spike.split(","))
        stages, phases = spike_plan(base_users, peak_users)
        env.update(STAGES=json.dumps(stages), PHASES=json.dumps(phases), SUBMIT_WINDOW_S="30")
        print(f"\n== spike: {base_users} users, jump to {peak_users} in 10s, hold 60s, back to {base_users} for 180s")
    else:
        print(f"\n== {vus} users: {args.ramp_up}s ramp-up + {args.hold}s hold")
    subprocess.run([k6, "run", "--quiet", os.path.join(BASE_DIR, "load_test_k6.js")], env=env, check=False)
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    api = api_stats(summary)
    os.remove(summary_path)

    deadline = time.monotonic() + args.drain_timeout
    while time.monotonic() < deadline:
        jobs = watcher.created_since(started)
        if all(j["finished"] for j in jobs):
            break
        time.sleep(POLL_S)
    jobs = job_stats(watcher.created_since(started))
    over = [p for p, limit in JOB_LIMITS_S.items()
            if jobs["queue_wait_s"][p] is not None and jobs["queue_wait_s"][p] > limit]
    passed = not api["thresholds_failed"] and not over and jobs["failed"] == 0 and jobs["unfinished"] == 0
    result = {"users": vus, "passed": passed, "api": api, "jobs": jobs,
              "job_limits_failed": [f"queue_wait_{p}" for p in over]}
    if phases:
        bounds = [0] + [ph["end_s"] for ph in phases]
        result["phases"] = phase_stats(summary, phases)
        result["jobs_by_phase"] = {
            ph["name"]: job_stats([j for j in watcher.created_since(started)
                                   if bounds[i] <= (_ts(j["created_at"]) - started).total_seconds() < bounds[i + 1]])
            for i, ph in enumerate(phases)
        }
    return result


def print_level(r: dict) -> None:
    a, j = r["api"], r["jobs"]
    print(f"   {'endpoint':16}{'count':>7}{'p50':>8}{'p95':>8}{'p99':>8}{'err%':>7}")
    for name, s in a["endpoints"].items():
        print(f"   {name:16}{s['count']:>7}{s['p50']:>6}ms{s['p95']:>6}ms{s['p99']:>6}ms{s['error_pct']:>7}")
    print(f"   requests {a['requests']}, errors {a['error_pct']}%")
    print(f"   jobs: {j['submitted']} submitted, {j['done']} done, {j['failed']} failed, {j['unfinished']} unfinished")
    for name in ("queue_wait_s", "run_s", "total_s"):
        s = j[name]
        print(f"   {name:13} p50 {s['p50']}s  p95 {s['p95']}s  p99 {s['p99']}s")
    for phase, endpoints in r.get("phases", {}).items():
        jp = r["jobs_by_phase"][phase]
        print(f"   [{phase}] " + "  ".join(f"{e} {s['p50']}/{s['p95']}/{s['p99']}ms" for e, s in endpoints.items())
              + f"  | jobs {jp['submitted']}, wait {jp['queue_wait_s']['p50']}/{jp['queue_wait_s']['p95']}"
              + f"/{jp['queue_wait_s']['p99']}s")
    failed = a["thresholds_failed"] + r["job_limits_failed"]
    print(f"   {'PASS' if r['passed'] else 'FAIL'}" + (f" ({', '.join(failed)})" if failed else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="10,25,50,75,100,150")
    ap.add_argument("--ramp-up", type=int, default=30)
    ap.add_argument("--hold", type=int, default=180)
    ap.add_argument("--accounts", type=int, default=20)
    ap.add_argument("--seed-leads", type=int, default=25)
    ap.add_argument("--jobs-per-user", type=int, default=1, help="leads each virtual user submits per level")
    ap.add_argument("--lead-seconds", type=float, default=41,
                    help="stubbed pipeline time per lead; 41s is the golden evaluation's average")
    ap.add_argument("--drain-timeout", type=int, default=600)
    ap.add_argument("--stop-error-pct", type=float, default=25)
    ap.add_argument("--port", type=int, default=8011)
    ap.add_argument("--workers", type=int, default=0,
                    help="separate stub worker processes; 0 runs the worker inside the API process")
    ap.add_argument("--report", default="k6.json", help="report file name under results/load_test/")
    ap.add_argument("--spike", default=None, metavar="BASE,PEAK",
                    help="one spike run instead of levels: BASE users, jump to PEAK in 10s, hold 60s, recover")
    args = ap.parse_args()

    k6 = find_k6()
    preflight()
    os.makedirs(OUT_DIR, exist_ok=True)
    tag = f"{TAG_PREFIX}-k6-{int(time.time())}"
    base = f"http://127.0.0.1:{args.port}"
    env = {k: v for k, v in os.environ.items() if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    env["DAILY_LEAD_CAP"] = "100000"
    log = open(os.path.join(OUT_DIR, "k6_server.log"), "w", encoding="utf-8")
    api = subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "load_test_api.py"), "--serve",
                            "--port", str(args.port), "--lead-seconds", str(args.lead_seconds)]
                           + (["--no-in-process-worker"] if args.workers else []),
                           env=env, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT)
    workers = [subprocess.Popen([sys.executable, os.path.join(BASE_DIR, "load_test.py"), "--worker-mode",
                                 "--lead-seconds", str(args.lead_seconds)],
                                env=env, cwd=ROOT_DIR, stdout=log, stderr=subprocess.STDOUT)
               for _ in range(args.workers)]
    users, watcher, results = [], None, []
    accounts_path = os.path.join(OUT_DIR, "k6_accounts.json")
    try:
        if not wait_for_health(base):
            sys.exit("The stubbed API did not come up; see results/load_test/k6_server.log.")
        users = create_accounts(tag, args.accounts, args.seed_leads)
        with open(accounts_path, "w", encoding="utf-8") as f:
            json.dump([{"username": u["username"], "password": PASSWORD} for u in users], f)
        print(f"API up on {base}; {len(users)} accounts with {args.seed_leads} leads each; "
              f"stub {args.lead_seconds}s per lead; "
              + (f"{args.workers} separate worker(s)" if args.workers else "worker in-process"))
        watcher = JobWatcher([str(u["id"]) for u in users])
        watcher.start()
        for vus in [int(args.spike.split(",")[1])] if args.spike else [int(x) for x in args.levels.split(",")]:
            r = run_level(k6, base, vus, args, accounts_path, watcher)
            results.append(r)
            print_level(r)
            if r["api"]["error_pct"] > args.stop_error_pct:
                print(f"Stopping: errors above {args.stop_error_pct}%")
                break
    finally:
        if watcher:
            watcher.stop()
        cleanup(users)
        if os.path.exists(accounts_path):
            os.remove(accounts_path)
        for proc in [api, *workers]:
            proc.terminate()
            proc.wait(timeout=30)
        log.close()

    passing = [r["users"] for r in results if r["passed"]]
    report = {"settings": {k: getattr(args, k) for k in ("levels", "ramp_up", "hold", "accounts", "seed_leads",
                                                         "jobs_per_user", "lead_seconds", "workers")}
                           | {"slots_per_worker": int(os.getenv("MAX_CONCURRENT_JOBS", "10"))},
              "limits": {"reads_ms": {"p50": 800, "p95": 1500, "p99": 3000},
                         "process_ms": {"p50": 2500, "p95": 4000, "p99": 6000},
                         "queue_wait_s": JOB_LIMITS_S, "error_pct": 1},
              "levels": results,
              "highest_passing_users": max(passing) if passing else None}
    with open(os.path.join(OUT_DIR, args.report), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nHighest level passing every limit: {report['highest_passing_users']} users. "
          f"Report: results/load_test/{args.report}")


if __name__ == "__main__":
    main()
