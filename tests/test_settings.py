"""No-network checks for saved API keys, SMTP passwords and daily-credit rules."""

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

import backend  # noqa: E402
from security import encrypt_secret  # noqa: E402
from postgrest.exceptions import APIError  # noqa: E402


class FakeTable:
    """Serve and record one users row without network access."""

    def __init__(self, db, missing_columns=False):
        self.db = db
        self.missing_columns = missing_columns
        self.pending = None

    def select(self, _columns):
        return self

    def update(self, payload):
        self.pending = payload
        return self

    def eq(self, _column, _value):
        return self

    gte = neq = eq

    def execute(self):
        if self.pending is not None:
            self.db["row"].update(self.pending)
            self.db["updates"].append(self.pending)
            return SimpleNamespace(data=[self.db["row"]])
        if self.missing_columns:
            raise APIError({"message": "column missing", "code": "42703", "hint": None, "details": None})
        return SimpleNamespace(data=[self.db["row"]])


class FakeSupabase:
    """Stand in for the sync Supabase client."""

    def __init__(self, row, missing_columns=False):
        self.db = {"row": row, "updates": []}
        self.missing_columns = missing_columns

    def table(self, _name):
        return FakeTable(self.db, self.missing_columns)


secret = os.environ["SECRET_KEY"]
original_supabase, original_model = backend.supabase, backend.LLM_MODEL
try:
    backend.supabase = FakeSupabase({})
    llm, tavily, own = backend._resolve_api_keys("7")
    assert (llm, tavily, own) == ("operator-gemini", "operator-tavily", False)

    backend.supabase = FakeSupabase({
        "gemini_api_key_enc": encrypt_secret("user-gemini-1234", secret),
        "tavily_api_key_enc": encrypt_secret("user-tavily-5678", secret),
    })
    assert backend._resolve_api_keys("7") == ("user-gemini-1234", "user-tavily-5678", True)
    queued_llm, queued_tavily, _ = backend._resolve_api_keys("7", for_job=True)
    assert queued_llm.startswith("enc:") and "user-gemini" not in queued_llm
    assert queued_tavily.startswith("enc:") and "user-tavily" not in queued_tavily
    assert backend.get_credits("7")["unlimited"] is True
    keys = backend.get_api_keys("7")
    assert keys["gemini"] == {"saved": True, "last4": "1234"} and keys["unlimited"] is True

    import worker
    assert worker._job_key(queued_llm) == "user-gemini-1234"
    assert worker._job_key("operator-gemini") == "operator-gemini"

    backend.LLM_MODEL = "CLOUDFLARE"
    llm, tavily, own = backend._resolve_api_keys("7")
    assert (llm, tavily, own) == (backend.LLM_API_KEY, "user-tavily-5678", False), \
        "a saved Gemini key must not replace the Cloudflare model key"
    assert backend.get_credits("7")["unlimited"] is False
    backend.LLM_MODEL = original_model

    backend.supabase = FakeSupabase({}, missing_columns=True)
    assert backend._resolve_api_keys("7") == ("operator-gemini", "operator-tavily", False), \
        "a missing key migration must not break processing"

    fake = FakeSupabase({"email_smtp_password": "legacy-plain-9999"})
    backend.supabase = fake
    assert backend._smtp_password("7", "legacy-plain-9999") == "legacy-plain-9999"
    upgraded = fake.db["updates"][-1]["email_smtp_password"]
    assert upgraded.startswith("gAAAAA") and "legacy-plain" not in upgraded
    assert backend._smtp_password("7", upgraded) == "legacy-plain-9999"
finally:
    backend.supabase, backend.LLM_MODEL = original_supabase, original_model

print("test_settings.py: all checks passed")
