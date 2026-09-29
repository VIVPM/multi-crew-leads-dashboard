"""No-network checks that user actions reach audit_events and reads don't."""

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
    "GEMINI_API_KEY": "operator-gemini",
    "TAVILY_API_KEY": "operator-tavily",
    "ALLOWED_ORIGINS": "http://localhost:5173",
}.items():
    os.environ.setdefault(name, value)

from fastapi.testclient import TestClient  # noqa: E402

import backend  # noqa: E402
from security import make_token  # noqa: E402


class FakeQuery:
    """Filter, insert, update and delete rows of one in-memory table."""

    def __init__(self, db, name):
        self.db, self.name = db, name
        self.filters, self.op, self.payload = [], "select", None

    def select(self, *_a, **_k):
        return self

    def insert(self, payload):
        self.op, self.payload = "insert", payload
        return self

    def update(self, payload):
        self.op, self.payload = "update", payload
        return self

    def delete(self):
        self.op = "delete"
        return self

    def eq(self, column, value):
        self.filters.append((column, str(value)))
        return self

    def limit(self, *_a, **_k):
        return self

    order = range = in_ = limit

    def _matches(self, row):
        return all(str(row.get(c)) == v for c, v in self.filters)

    def execute(self):
        rows = self.db.setdefault(self.name, [])
        if self.op == "insert":
            row = {"id": len(rows) + 100, **self.payload}
            rows.append(row)
            return SimpleNamespace(data=[row])
        hits = [r for r in rows if self._matches(r)]
        if self.op == "update":
            for r in hits:
                r.update(self.payload)
        if self.op == "delete":
            self.db[self.name] = [r for r in rows if r not in hits]
        return SimpleNamespace(data=hits, count=len(hits))


class FakeSupabase:
    """Stand in for the sync Supabase client."""

    def __init__(self, db):
        self.db = db

    def table(self, name):
        return FakeQuery(self.db, name)


class FakeAsyncQuery(FakeQuery):
    """Awaitable variant for supabase_async."""

    async def execute(self):
        return FakeQuery.execute(self)


class FakeAsyncSupabase(FakeSupabase):
    def table(self, name):
        return FakeAsyncQuery(self.db, name)


db = {"leads": [{"id": 1, "user_id": "8", "name": "Not yours"}]}
original = backend.supabase, backend.supabase_async
backend.supabase, backend.supabase_async = FakeSupabase(db), FakeAsyncSupabase(db)
try:
    client = TestClient(backend.app)
    me = {"Authorization": f"Bearer {make_token('7', os.environ['SECRET_KEY'])}"}

    def events():
        return db.setdefault("audit_events", [])

    created = client.post("/leads", json={"name": "Asha", "company": "Acme", "email": "asha@acme.test"}, headers=me)
    assert created.status_code == 200, created.text
    lead_id = created.json()["id"]
    last = events()[-1]
    assert (last["action"], last["target_type"], last["target_id"], last["outcome"]) == \
        ("lead.created", "lead", str(lead_id), "success")
    assert last["actor_user_id"] == 7 and last["status_code"] == 200 and last["request_id"]
    assert set(last) == {"id", "actor_user_id", "action", "target_type", "target_id",
                         "outcome", "status_code", "request_id"}

    assert client.delete(f"/leads/{lead_id}", headers=me).status_code == 200
    assert (events()[-1]["action"], events()[-1]["outcome"]) == ("lead.deleted", "success")

    assert client.delete("/leads/1", headers=me).status_code == 404
    last = events()[-1]
    assert (last["action"], last["target_id"], last["outcome"], last["status_code"]) == \
        ("lead.deleted", "1", "rejected", 404), "another user's lead must log a rejected delete"

    before = len(events())
    bad = {"Authorization": "Bearer not-a-token"}
    assert client.delete("/leads/1", headers=bad).status_code == 401
    assert len(events()) == before, "an unverified caller must not produce an actor row"

    secret_text = "Top secret ICP text"
    assert client.put("/account/company-context", json={"company_context": secret_text},
                      headers=me).status_code == 200
    last = events()[-1]
    assert (last["action"], last["target_id"]) == ("account.company_context.updated", "7")
    assert secret_text not in str(last), "settings text must never reach the audit row"

    before = len(events())
    client.get("/", headers=me)
    client.get("/leads/7", headers=me)
    assert len(events()) == before, "reads must not be audited"
finally:
    backend.supabase, backend.supabase_async = original

print("test_audit.py: all checks passed")
