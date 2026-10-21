"""One paid lead per Locust user; fixtures and output paths come from the runner."""

import itertools
import json
import os
import time
from pathlib import Path

from gevent import sleep
from locust import HttpUser, task

items = json.loads(Path(os.environ["REAL_AGENT_USERS_FILE"]).read_text(encoding="utf-8"))
max_wait_s = int(os.getenv("REAL_AGENT_TIMEOUT_S", "480"))
output = Path(os.environ["REAL_AGENT_OUTPUT_FILE"])
next_index = itertools.count()
completed = 0


class LeadUser(HttpUser):
    """Authenticate, submit one lead, and wait for its worker result."""

    def on_start(self):
        self.index = next(next_index)
        self.item = items[self.index] if self.index < len(items) else None
        self.token = None
        if self.item is None:
            return
        response = self.client.post("/auth/login", json={
            "username": self.item["username"], "password": self.item["password"],
        }, name="POST /auth/login")
        if response.status_code == 200:
            self.token = response.json()["token"]

    @task
    def process_one(self):
        global completed
        if self.item is None:
            sleep(1)
            return
        item, self.item = self.item, None
        job_id = None
        lead_id = None
        error = None
        started = time.monotonic()
        try:
            if not self.token:
                raise RuntimeError("login failed")
            headers = {"Authorization": f"Bearer {self.token}"}
            lead = self.client.post("/leads", json=item["lead"], headers=headers, name="POST /leads")
            if lead.status_code != 200:
                raise RuntimeError(f"create lead HTTP {lead.status_code}")
            lead_body = lead.json()
            lead_id = lead_body["id"]
            started = time.monotonic()
            queued = self.client.post("/leads/process", json={
                "leads": [lead_body], "force_refresh": True,
            }, headers={**headers, "Idempotency-Key": f"real-ai-{item['id']}"}, name="POST /leads/process")
            if queued.status_code != 202:
                raise RuntimeError(f"enqueue HTTP {queued.status_code}")
            job_id = queued.json()["job_id"]
            while time.monotonic() - started < max_wait_s:
                status = self.client.get(f"/jobs/{job_id}", headers=headers, name="GET /jobs/{id}")
                if status.status_code != 200:
                    raise RuntimeError(f"job status HTTP {status.status_code}")
                state = status.json()
                if state["status"] == "done":
                    break
                if state["status"] == "failed":
                    raise RuntimeError(f"worker failed: {state.get('error', '')[:160]}")
                sleep(3)
            else:
                raise RuntimeError(f"job did not finish in {max_wait_s} s")
        except Exception as exc:
            error = str(exc)
        elapsed_ms = round((time.monotonic() - started) * 1000)
        self.environment.events.request.fire(
            request_type="JOB", name="submit-to-result", response_time=elapsed_ms,
            response_length=0, exception=RuntimeError(error) if error else None,
        )
        with output.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"id": item["id"], "lead_id": lead_id, "job_id": job_id,
                                "elapsed_s": round(elapsed_ms / 1000, 1), "error": error}) + "\n")
        completed += 1
        if completed >= len(items):
            self.environment.runner.quit()
