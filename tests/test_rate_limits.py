"""Rate limits reject excess requests before auth or paid model work, without network."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
for name, value in {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_KEY": "test-key",
    "SECRET_KEY": "test-secret",
    "DAILY_LEAD_CAP": "5",
    "LLM_MODEL": "GEMINI",
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "ALLOWED_ORIGINS": "http://localhost:5173",
}.items():
    os.environ.setdefault(name, value)

from fastapi import HTTPException, Request, Response  # noqa: E402
import backend  # noqa: E402


class Counter:
    """Match the database function's result for one 60-second window."""

    def __init__(self):
        self.hits = {}
        self.calls = []

    def rpc(self, name, params):
        assert name == "consume_rate_limit"
        self.calls.append(params)
        return SimpleNamespace(execute=lambda: self.take(params))

    def take(self, params):
        key = params["p_scope"], params["p_subject"]
        spent = self.hits.get(key, 0)
        cost, maximum = params["p_hits"], params["p_max_hits"]
        if spent + cost <= maximum:
            self.hits[key] = spent + cost
            retry = 0
        else:
            retry = params["p_window_seconds"]
        return SimpleNamespace(data=[{"retry_after_seconds": retry}])


class FakeQuery:
    def __init__(self, name):
        self.name = name

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def execute(self):
        return SimpleNamespace(data=([{"id": 1}] if self.name == "leads" else
                                     [{"company_context": "Test ICP"}]))


class FakeDatabase(Counter):
    def table(self, name):
        assert name != "jobs", "rejected requests must not enqueue work"
        return FakeQuery(name)


request = Request({"type": "http", "headers": []})
counter = Counter()
with patch.object(backend, "supabase", counter):
    for _ in range(5):
        backend._enforce_rate_limit("signup", "tester")
    try:
        backend._enforce_rate_limit("signup", "tester")
        raise AssertionError("signup limit did not reject excess calls")
    except HTTPException as exc:
        assert exc.status_code == 429 and exc.headers["Retry-After"] == "60"
    backend._enforce_rate_limit("signup", "someone-else")
    backend._enforce_rate_limit("process", "user-1", 20)
    try:
        backend._enforce_rate_limit("process", "user-1", 11)
        raise AssertionError("weighted lead limit did not reject excess work")
    except HTTPException as exc:
        assert exc.status_code == 429
    backend._enforce_rate_limit("draft_email", "user-1")

    try:
        backend.signup(backend.SignupRequest(username="Tester", password="password123"), request)
        raise AssertionError("signup skipped rate limit")
    except HTTPException as exc:
        assert exc.status_code == 429
    for _ in range(30):
        backend._enforce_rate_limit("login", "tester")
    with patch.object(backend, "_recent_failure_count", side_effect=AssertionError("checked DB before limit")):
        try:
            backend.login(backend.LoginRequest(username="Tester", password="password123"), request)
            raise AssertionError("login skipped rate limit")
        except HTTPException as exc:
            assert exc.status_code == 429
    token_key = backend.hash_refresh_token("test")[:64]
    for _ in range(120):
        backend._enforce_rate_limit("refresh", token_key)
    try:
        backend.refresh_access_token(backend.RefreshRequest(refresh_token="test"))
        raise AssertionError("refresh skipped rate limit")
    except HTTPException as exc:
        assert exc.status_code == 429
    assert not any("ip" in str(c["p_subject"]) for c in counter.calls)

db = FakeDatabase()
with patch.object(backend, "supabase", db), patch.object(
        backend, "_resolve_api_keys", return_value=("key", "search", True, None)):
    backend._enforce_rate_limit("process", "user-7", 30)
    try:
        backend.process_leads_endpoint(
            backend.ProcessLeadsRequest(leads=[{"id": 1}]), request, Response(), "user-7", None,
        )
        raise AssertionError("processing ignored the per-user allowance")
    except HTTPException as exc:
        assert exc.status_code == 429

    for _ in range(5):
        backend._enforce_rate_limit("draft_email", "user-7")
    with patch.object(backend, "_get_owned_lead", return_value={
        "id": 1, "email_draft": None, "scoring_result": {"lead_score": {"score": 68}}, "score": 68,
    }):
        try:
            backend.draft_email_for_lead("1", "user-7")
            raise AssertionError("drafting ignored the per-user allowance")
        except HTTPException as exc:
            assert exc.status_code == 429

with patch.object(backend.supabase, "rpc", side_effect=RuntimeError("missing migration")), \
     patch.object(backend.logger, "exception"):
    try:
        backend._enforce_rate_limit("process", "user-2")
        raise AssertionError("missing limiter silently allowed work")
    except HTTPException as exc:
        assert exc.status_code == 503

print("test_rate_limits.py: all checks passed")
