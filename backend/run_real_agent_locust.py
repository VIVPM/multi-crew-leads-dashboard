"""Run bounded Locust real-AI stages with owned test accounts and cleanup.

Usage:
    python backend/run_real_agent_locust.py --levels 1,2,5,10,20,40,80,160,250
    python backend/run_real_agent_locust.py --smoke
    python backend/run_real_agent_locust.py --levels 250 --tavily-key-file key.txt

Stops at the first level with any job error or non-done status. Each level's
run output (per-job cost/stage data joined from Supabase) is printed as one
JSON line: {"users": <level>, "jobs": [...]}.
"""

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from supabase import create_client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BACKEND = HERE
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND))
from security import hash_password  # noqa: E402
from pipeline import normalize_company_key  # noqa: E402


def launch(port, smoke, log_path, env):
    """Start only a local server owned by this run."""
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"Port {port} is already in use")
    if smoke:
        cmd = [sys.executable, str(BACKEND / "load_test_api.py"), "--serve", "--port", str(port),
               "--lead-seconds", "0.1"]
        env = {k: v for k, v in env.items() if k not in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")}
    else:
        cmd = [sys.executable, "-m", "uvicorn", "backend.backend:app", "--host", "127.0.0.1",
               "--port", str(port)]
        env["RUN_WORKER_IN_PROCESS"] = "1"
    handle = log_path.open("w", encoding="utf-8")
    server = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    for _ in range(80):
        if server.poll() is not None:
            raise RuntimeError(f"API exited with code {server.returncode}; see {log_path}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/", timeout=2).status_code == 200:
                return server, handle
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError(f"API did not start; see {log_path}")


def clean_test_data(db, accounts, marker, fixture):
    """Remove only rows created for the temporary benchmark accounts."""
    ids = [str(u["id"]) for u in accounts]
    if not ids:
        return
    leads = db.table("leads").select("id").in_("user_id", ids).execute().data or []
    for lead in leads:
        db.table("analysis_runs").delete().eq("lead_id", lead["id"]).execute()
    for table, field in (("jobs", "user_id"), ("leads", "user_id"),
                         ("refresh_tokens", "user_id"), ("audit_events", "actor_user_id")):
        db.table(table).delete().in_(field, ids).execute()
    keys = sorted({normalize_company_key(item["lead"]["company"], marker) for item in fixture})
    db.table("company_research_cache").delete().in_("company_key", keys).execute()
    db.table("users").delete().in_("id", ids).execute()
    print(f"Cleaned {len(ids)} benchmark accounts and {len(leads)} leads", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--levels", default="1,2,5,10")
    parser.add_argument("--tavily-key-file", type=Path)
    args = parser.parse_args()
    db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    fixture = json.loads((BACKEND / "golden_leads.json").read_text(encoding="utf-8"))["leads"]
    levels = [1] if args.smoke else [int(x) for x in args.levels.split(",")]
    base = fixture
    fixture = []
    for i in range(sum(levels)):
        record = base[i % len(base)]
        lead = dict(record["lead"])
        if i >= len(base):
            lead["name"] = f"{lead['name']} {i // len(base) + 1}"
            lead["email"] = f"copy{i}-{lead['email']}"
        fixture.append({"id": f"{record['id']}-{i}", "lead": lead})
    tag = "real-agent-" + uuid.uuid4().hex[:10]
    icp = (ROOT / "company_icp.txt").read_text(encoding="utf-8") + f"\nBenchmark reference: {tag}."
    password = secrets.token_urlsafe(24)
    accounts = []
    port = 8013 if args.smoke else 8012
    server = handle = None
    with tempfile.TemporaryDirectory(prefix="real-agent-") as folder:
        folder = Path(folder)
        try:
            server_env = dict(os.environ)
            if args.tavily_key_file:
                server_env["TAVILY_API_KEY"] = args.tavily_key_file.read_text(encoding="utf-8").strip()
                args.tavily_key_file.unlink()
                if not server_env["TAVILY_API_KEY"]:
                    raise ValueError("Tavily key file is empty")
            server, handle = launch(port, args.smoke, folder / "api.log", server_env)
            hashed_password = hash_password(password)
            rows = [{"username": f"{tag}-{i}", "password": hashed_password,
                     "company_context": icp} for i in range(len(fixture))]
            accounts = db.table("users").insert(rows).execute().data or []
            if len(accounts) != len(fixture):
                raise RuntimeError("Could not create all benchmark accounts")
            uvx = shutil.which("uvx")
            if not uvx:
                raise RuntimeError("Locust launcher uvx is unavailable")
            offset = 0
            for level in levels:
                items = [{"id": record["id"], "lead": record["lead"],
                          "username": account["username"], "password": password}
                         for record, account in zip(fixture[offset:offset + level],
                                                    accounts[offset:offset + level])]
                offset += level
                users_file = folder / "users.json"
                users_file.write_text(json.dumps(items), encoding="utf-8")
                results_file = folder / "results.jsonl"
                results_file.write_text("", encoding="utf-8")
                env = dict(os.environ, REAL_AGENT_USERS_FILE=str(users_file),
                           REAL_AGENT_OUTPUT_FILE=str(results_file), REAL_AGENT_TIMEOUT_S="1200")
                print(f"Starting {level} concurrent real-agent users", flush=True)
                cmd = [uvx, "--from", "locust>=2.40,<3", "locust", "-f",
                       str(BACKEND / "real_agent_locust.py"), "--headless", "-u", str(level),
                       "-r", str(level), "--run-time", "20m", "--stop-timeout", "120s",
                       "--host", f"http://127.0.0.1:{port}", "--only-summary", "--loglevel", "ERROR"]
                with (folder / f"locust-{level}.log").open("w", encoding="utf-8") as locust_log:
                    result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=locust_log,
                                            stderr=subprocess.STDOUT, timeout=1350)
                entries = [json.loads(line) for line in results_file.read_text(encoding="utf-8").splitlines()]
                if len(entries) != level:
                    raise RuntimeError(f"{level} users produced {len(entries)} results, exit {result.returncode}; "
                                       f"see {folder / f'locust-{level}.log'}")
                details = []
                for entry in entries:
                    if entry["job_id"]:
                        job = db.table("jobs").select("status,error,started_at,created_at").eq(
                            "id", entry["job_id"]).execute().data[0]
                        entry["status"] = job["status"]
                        entry["queue_wait_s"] = (datetime.fromisoformat(job["started_at"]) -
                                                   datetime.fromisoformat(job["created_at"])).total_seconds() if job["started_at"] else None
                        if entry["lead_id"] and job["status"] == "done" and not args.smoke:
                            analysis = db.table("analysis_runs").select("total_cost,duration_seconds,agents_data").eq(
                                "lead_id", entry["lead_id"]).execute().data
                            if analysis:
                                entry["model_cost_usd"] = analysis[0]["total_cost"]
                                entry["stages"] = analysis[0]["agents_data"]
                    details.append(entry)
                print(json.dumps({"users": level, "jobs": details}, default=str), flush=True)
                if result.returncode or any(entry["error"] or entry.get("status") != "done" for entry in entries):
                    print("Stopped at first failed level", flush=True)
                    break
        finally:
            if server is not None:
                server.terminate()
                server.wait(timeout=35)
            if handle is not None:
                handle.close()
            clean_test_data(db, accounts, icp, fixture)


if __name__ == "__main__":
    main()
