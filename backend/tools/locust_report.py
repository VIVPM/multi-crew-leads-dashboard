"""Sanitize and retain real-agent Locust measurements without account or lead data."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _percentiles(values):
    values = sorted(values)
    if not values:
        return None
    return {f"p{p}": values[round((p / 100) * (len(values) - 1))]
            for p in (50, 95, 99)}


def _failure_category(entry):
    error = (entry.get("worker_error") or entry.get("error") or "").lower()
    if entry.get("status") == "done":
        return "client_http_error" if entry.get("http_errors") else (
            "client_error" if entry.get("error") else None)
    if "429" in error or "quota" in error or "rate limit" in error:
        return "judge_quota" if "judge" in error or "hugging face" in error or "scaleway" in error else "provider_quota"
    if "timeout" in error or "timed out" in error or "did not finish" in error:
        return "timeout"
    if entry.get("status") == "failed":
        return "worker_failure"
    if entry.get("http_errors"):
        return "client_http_error"
    return "client_error" if entry.get("error") else "unfinished"


def _safe_job(entry, index):
    errors = [{"step": e["step"], "status": int(e["status"])}
              for e in entry.get("http_errors", [])]
    return {
        "index": index,
        "job_status": entry.get("status", "unknown"),
        "client_error_free": not entry.get("error") and not errors,
        "http_errors": errors,
        "queue_wait_s": entry.get("queue_wait_s"),
        "submit_to_result_s": entry.get("elapsed_s") if entry.get("job_id") else None,
        "model_cost_usd_estimate": entry.get("model_cost_usd"),
        "failure_category": _failure_category(entry),
    }


def save_report(path, workload, levels, *, run_status):
    """Atomically save only allowlisted fields outside the runner's temporary folder."""
    path = Path(path)
    safe_levels = []
    for level in levels:
        jobs = [_safe_job(entry, i + 1) for i, entry in enumerate(level["jobs"])]
        completed = [j for j in jobs if j["job_status"] == "done"]
        clean = [j for j in jobs if j["client_error_free"] and j["job_status"] == "done"]
        costs = [j["model_cost_usd_estimate"] for j in completed
                 if j["model_cost_usd_estimate"] is not None]
        safe_levels.append({
            "virtual_users": level["virtual_users"],
            "locust_exit_code": level.get("locust_exit_code"),
            "summary": {
                "records_captured": len(jobs),
                "completed_jobs": len(completed),
                "failed_jobs": sum(j["job_status"] == "failed" for j in jobs),
                "unknown_or_unfinished_jobs": sum(j["job_status"] not in ("done", "failed") for j in jobs),
                "error_free_client_workflows": len(clean),
                "client_http_error_workflows": sum(bool(j["http_errors"]) for j in jobs),
                "submit_to_result_s_error_free": _percentiles([
                    j["submit_to_result_s"] for j in clean if j["submit_to_result_s"] is not None]),
                "queue_wait_s_completed": _percentiles([
                    j["queue_wait_s"] for j in completed if j["queue_wait_s"] is not None]),
                "model_cost_usd_estimate_recorded": round(sum(costs), 6),
                "model_cost_records": len(costs),
            },
            "jobs": jobs,
        })
    report = {"schema_version": 1, "written_at_utc": datetime.now(timezone.utc).isoformat(),
              "run_status": run_status, "workload": workload, "levels": safe_levels}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return report
