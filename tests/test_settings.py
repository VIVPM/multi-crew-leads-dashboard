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
    assert backend._resolve_api_keys("7") == ("operator-gemini", "operator-tavily", False, None)

    import worker
    enc = lambda value: encrypt_secret(value, secret)  # noqa: E731
    legacy_gemini = {"gemini_api_key_enc": enc("user-gemini-1234"), "tavily_api_key_enc": enc("user-tavily-5678")}

    backend.supabase = FakeSupabase(dict(legacy_gemini))
    llm, tavily, unlimited, own = backend._resolve_api_keys("7")
    assert (llm, tavily, unlimited, own["provider"]) == ("user-gemini-1234", "user-tavily-5678", True, "GEMINI"),         "a Gemini key saved before the model choice still counts as Gemini"
    queued_llm, queued_tavily, _, _ = backend._resolve_api_keys("7", for_job=True)
    assert queued_llm.startswith("enc:") and "user-gemini" not in queued_llm
    assert queued_tavily.startswith("enc:") and "user-tavily" not in queued_tavily
    assert worker._job_llm(queued_llm) == ("user-gemini-1234", "GEMINI", None)
    assert worker._job_key(queued_tavily) == "user-tavily-5678"
    assert worker._job_llm("enc:" + enc("old-queued-gemini")) == ("old-queued-gemini", "GEMINI", None)
    assert worker._job_llm("operator-gemini") == ("operator-gemini", None, None)
    assert backend.get_credits("7")["unlimited"] is True
    keys = backend.get_api_keys("7")
    assert keys["gemini"] == {"saved": True, "last4": "1234"} and keys["unlimited"] is True
    assert keys["active_provider"] == "GEMINI"

    backend.LLM_MODEL = "CLOUDFLARE"
    backend.supabase = FakeSupabase({
        "llm_provider": "CLOUDFLARE", "cloudflare_account_id_enc": enc("acct-42"),
        "cloudflare_api_token_enc": enc("cf-token-9876"), "tavily_api_key_enc": enc("user-tavily-5678"),
    })
    queued_llm, _, unlimited, own = backend._resolve_api_keys("7", for_job=True)
    assert unlimited and own == {"provider": "CLOUDFLARE", "key": "cf-token-9876", "account_id": "acct-42"}
    assert worker._job_llm(queued_llm) == ("cf-token-9876", "CLOUDFLARE", "acct-42")
    assert backend.get_api_keys("7")["active_provider"] == "CLOUDFLARE"

    backend.supabase = FakeSupabase({
        "llm_provider": "CLOUDFLARE", "cloudflare_api_token_enc": enc("cf-token-9876"),
        "gemini_api_key_enc": enc("user-gemini-1234"), "tavily_api_key_enc": enc("user-tavily-5678"),
    })
    assert backend._resolve_api_keys("7") == (backend.LLM_API_KEY, "user-tavily-5678", False, None),         "Cloudflare chosen without an account ID uses the operator's model, not the saved Gemini key"
    assert backend.get_credits("7")["unlimited"] is False

    backend.supabase = FakeSupabase({"tavily_api_key_enc": enc("user-tavily-5678")})
    assert backend._resolve_api_keys("7") == (backend.LLM_API_KEY, "user-tavily-5678", False, None),         "a Tavily key alone keeps the operator's model and daily credits"
    assert backend.get_api_keys("7")["active_provider"] == "CLOUDFLARE"
    backend.LLM_MODEL = original_model

    backend.supabase = FakeSupabase({})
    backend.set_api_keys(backend.ApiKeysRequest(llm_provider="CLOUDFLARE", cloudflare_account_id=" acct-1 "), "7")
    saved_row = backend.supabase.db["row"]
    assert saved_row["llm_provider"] == "CLOUDFLARE" and "acct-1" not in saved_row["cloudflare_account_id_enc"]

    backend.supabase = FakeSupabase({**legacy_gemini, "llm_provider": "APP"})
    assert backend._resolve_api_keys("7") == ("operator-gemini", "user-tavily-5678", False, None),         "choosing the app's model ignores a saved Gemini key"

    backend.supabase = FakeSupabase({}, missing_columns=True)
    assert backend._resolve_api_keys("7") == ("operator-gemini", "operator-tavily", False, None),         "a missing key migration must not break processing"

    saved = backend.encrypt_secret("app-pass-9999", backend.SECRET_KEY)
    assert backend._smtp_password("7", saved) == "app-pass-9999"
finally:
    backend.supabase, backend.LLM_MODEL = original_supabase, original_model

print("test_settings.py: all checks passed")
