"""Offline proof that real-agent load evidence survives temporary cleanup."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "tools"))

from locust_report import save_report  # noqa: E402


with tempfile.TemporaryDirectory() as root:
    root = Path(root)
    retained = root / "retained" / "run.json"
    with tempfile.TemporaryDirectory(dir=root) as scratch:
        scratch = Path(scratch)
        (scratch / "users.json").write_text('secret-password', encoding="utf-8")
        entries = [
            {"id": "fit-1", "lead_id": 101, "job_id": "private-job", "status": "done",
             "elapsed_s": 12.5, "queue_wait_s": 3.2, "error": None,
             "http_errors": [], "model_cost_usd": 0.02, "stages": [{"secret": "private"}]},
            {"id": "fit-2", "lead_id": 102, "job_id": "private-job-2", "status": "done",
             "elapsed_s": 13.0, "queue_wait_s": 4.0, "error": "job status HTTP 500",
             "http_errors": [{"step": "job_status", "status": 500}]},
            {"id": "fit-3", "lead_id": 103, "job_id": "private-job-3", "status": "failed",
             "elapsed_s": 14.0, "queue_wait_s": 5.0,
             "error": "worker failed: judge 429 INSUFFICIENT_QUOTA private details",
             "http_errors": [], "worker_error": "judge 429 INSUFFICIENT_QUOTA private details"},
        ]
        report = save_report(retained, {"worker_slots_max": 10, "worker_slots_min": 2},
                             [{"virtual_users": 3, "jobs": entries}], run_status="stopped_on_failure")
        assert report["levels"][0]["summary"]["completed_jobs"] == 2
        assert report["levels"][0]["summary"]["error_free_client_workflows"] == 1
        assert report["levels"][0]["summary"]["failed_jobs"] == 1
        assert report["levels"][0]["jobs"][2]["failure_category"] == "judge_quota"
    assert retained.exists(), "report disappeared with temporary run files"
    saved = json.loads(retained.read_text(encoding="utf-8"))
    assert saved["workload"]["worker_slots_max"] == 10
    assert saved["levels"][0]["jobs"][1]["http_errors"] == [{"step": "job_status", "status": 500}]
    assert "private" not in retained.read_text(encoding="utf-8")
    assert "secret-password" not in retained.read_text(encoding="utf-8")

print("real-agent report survives cleanup and excludes sensitive fields")
