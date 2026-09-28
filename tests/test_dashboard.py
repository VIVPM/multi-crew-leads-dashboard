"""No-network checks for dashboard usage fields on the owned lead list."""

import asyncio
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
for name, value in {
    "SUPABASE_URL": "https://example.supabase.co",
    "SUPABASE_KEY": "test-key",
    "SECRET_KEY": "test-secret",
    "DAILY_LEAD_CAP": "5",
    "LLM_MODEL": "GEMINI",
    "GEMINI_API_KEY": "test-key",
    "TAVILY_API_KEY": "test-key",
    "ALLOWED_ORIGINS": "http://localhost:5173",
}.items():
    os.environ.setdefault(name, value)

import backend  # noqa: E402


class FakeQuery:
    """Record filters and return the rows used by the dashboard test."""

    def __init__(self, table, calls, leads):
        self.table = table
        self.calls = calls
        self.leads = leads

    def select(self, columns):
        self.calls.append((self.table, "select", columns))
        return self

    def eq(self, column, value):
        self.calls.append((self.table, "eq", column, value))
        return self

    def in_(self, column, values):
        self.calls.append((self.table, "in", column, values))
        return self

    def order(self, column, desc=False):
        self.calls.append((self.table, "order", column, desc))
        return self

    def range(self, start, end):
        self.calls.append((self.table, "range", start, end))
        return self

    async def execute(self):
        if self.table == "leads":
            return SimpleNamespace(data=[row.copy() for row in self.leads])
        return SimpleNamespace(data=[
            {"lead_id": 10, "total_tokens": 200, "total_cost": 0.003,
             "created_at": "2026-09-04T00:00:00Z"},
            {"lead_id": 10, "total_tokens": 100, "total_cost": 0.001,
             "created_at": "2026-08-04T00:00:00Z"},
        ])


class FakeSupabase:
    """Stand in for the async Supabase client without making requests."""

    def __init__(self, leads):
        self.leads = leads
        self.calls = []

    def table(self, name):
        return FakeQuery(name, self.calls, self.leads)


async def check_dashboard_data():
    """Check ownership, latest analysis, and absent usage without network calls."""
    original = backend.supabase_async
    fake = FakeSupabase([{"id": 10, "score": 85}, {"id": 11, "score": None}])
    backend.supabase_async = fake
    try:
        leads = await backend.get_leads("7", auth_user="7", limit=500, offset=0)
        assert leads[0]["total_tokens"] == 200
        assert leads[0]["total_cost"] == 0.003
        assert leads[0]["processed_at"] == "2026-09-04T00:00:00Z"
        assert leads[1]["total_cost"] is None
        assert leads[1]["total_tokens"] is None
        assert leads[1]["processed_at"] is None
        assert ("analysis_runs", "in", "lead_id", [10, 11]) in fake.calls
        assert ("analysis_runs", "order", "created_at", True) in fake.calls
        try:
            await backend.get_leads("8", auth_user="7", limit=500, offset=0)
        except backend.HTTPException as exc:
            assert exc.status_code == 403
        else:
            raise AssertionError("lead list did not check ownership")
        assert fake.calls.count(("leads", "select", backend.LEAD_LIST_COLUMNS)) == 1
    finally:
        backend.supabase_async = original


asyncio.run(check_dashboard_data())
print("test_dashboard.py: all checks passed")
