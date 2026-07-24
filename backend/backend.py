"""
backend.py — FastAPI server wrapping CrewAI pipeline + Supabase operations.

Run the API:     uvicorn backend.backend:app --host 0.0.0.0 --port 8000
Run the worker:  python backend/worker.py   (processes queued lead-scoring jobs)

Lead processing is asynchronous: POST /leads/process enqueues a job row in
Supabase and returns 202 immediately; worker.py picks it up and writes results.
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import logging
import secrets
import smtplib
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup (logging_setup/security are siblings of this file)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

from logging_setup import configure_logging, request_context, lead_context
from security import hash_password, verify_password, is_legacy_hash, make_token, verify_token

# ---------------------------------------------------------------------------
# Logging — structured JSON, correlated by request_id (see logging_setup.py)
# ---------------------------------------------------------------------------
configure_logging()
logger = logging.getLogger("backend")

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in backend/.env")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_hex(32)
if not os.getenv("SECRET_KEY"):
    logger.warning(
        "SECRET_KEY not set — generated a random one; sessions will not survive "
        "restarts or be shared across instances. Set SECRET_KEY in backend/.env."
    )

MAX_LEADS_PER_REQUEST = 10
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_S = 900
# Well under Gmail's real ~100/day SMTP-relay ceiling (not the often-quoted
# 500/day, which is the web-interface limit) — a safety net so a bug can't
# silently blow through the account's real limit and get it flagged.
EMAIL_SEND_DAILY_CAP = int(os.getenv("EMAIL_SEND_DAILY_CAP", "80"))

# Operator-held keys — users no longer bring their own Gemini/Tavily keys.
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
TAVILY_API_KEY = os.environ["TAVILY_API_KEY"]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(title="Sales Pipeline Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://multi-crew-leads-dashboard-frontend.onrender.com",
        "http://localhost:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: HTTP-layer tracing (every endpoint, not just LLM calls) to
# Grafana Cloud only — Langfuse (see worker.py) stays focused on CrewAI/
# LiteLLM spans, this covers the rest of the app (auth, lead CRUD, admin
# overview) that Langfuse was never meant to show. Non-fatal if unset.
if os.getenv("GRAFANA_OTLP_ENDPOINT") and os.getenv("GRAFANA_OTLP_AUTH"):
    try:
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        _http_tracer_provider = TracerProvider(resource=Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "sales-pipeline-backend"),
            "service.namespace": "lead-coordinator",
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
        }))
        _http_tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
            endpoint=f"{os.environ['GRAFANA_OTLP_ENDPOINT'].rstrip('/')}/v1/traces",
            headers={"Authorization": os.environ["GRAFANA_OTLP_AUTH"]},
        )))
        FastAPIInstrumentor.instrument_app(app, tracer_provider=_http_tracer_provider)
        logger.info("HTTP-layer tracing enabled via OTLP (Grafana Cloud)")
    except Exception:
        logger.exception("Failed to initialize HTTP tracing (non-fatal)")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Every log line emitted while handling this request carries the same
    request_id, so `grep request_id=<x>` shows the full story for one call."""
    request_id = uuid.uuid4().hex[:12]
    with request_context(request_id):
        response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def health():
    return {"status": "ok", "service": "Sales Pipeline Backend"}


# =============================================================================
# Auth plumbing
# =============================================================================

def current_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: validate the Bearer token, return the user_id."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authentication token.")
    try:
        return verify_token(authorization[len("Bearer "):], SECRET_KEY)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid or expired token. Please log in again.")


# Supabase-backed (not in-memory) so the lockout holds across multiple API
# instances, not just per-process.
def _too_many_failures(username: str) -> bool:
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=LOGIN_WINDOW_S)).isoformat()
    resp = (
        supabase.table("login_failures")
        .select("id", count="exact")
        .eq("username", username)
        .gte("failed_at", cutoff)
        .execute()
    )
    return (resp.count or 0) >= LOGIN_MAX_FAILURES


def _record_login_failure(username: str) -> None:
    supabase.table("login_failures").insert({"username": username}).execute()


def _clear_login_failures(username: str) -> None:
    supabase.table("login_failures").delete().eq("username", username).execute()


# =============================================================================
# Request / Response models
# =============================================================================

class SignupRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=8, max_length=128)

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    user_id: str
    username: str
    token: str

class LeadCreate(BaseModel):
    name: str
    job_title: Optional[str] = None
    company: str
    email: str
    use_case: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = "Website"

class LeadUpdate(BaseModel):
    name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    use_case: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    email_draft: Optional[str] = None

class ProcessLeadsRequest(BaseModel):
    leads: List[dict]
    force_refresh: bool = False  # bypass the company-research cache for this batch

class CompanyProfileRequest(BaseModel):
    company_context: str = Field(max_length=4000)

class EmailSettingsRequest(BaseModel):
    smtp_host: str = Field(min_length=1, max_length=255)
    smtp_port: int = Field(default=587, ge=1, le=65535)
    from_address: str = Field(min_length=3, max_length=255)
    # Optional/blank on update: leaving it out keeps the previously saved
    # password rather than overwriting it with an empty value.
    smtp_password: Optional[str] = Field(default=None, max_length=255)


# =============================================================================
# Auth endpoints
# =============================================================================

@app.post("/auth/signup")
def signup(req: SignupRequest):
    existing = supabase.table("users").select("id").eq("username", req.username).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Username already exists.")
    try:
        supabase.table("users").insert(
            {"username": req.username, "password": hash_password(req.password)}
        ).execute()
    except Exception:
        # unique constraint (migrations.sql) closes the check-then-insert race
        raise HTTPException(status_code=400, detail="Username already exists.")
    logger.info("New user signed up: %s", req.username)
    return {"message": "Signup successful."}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    if _too_many_failures(req.username):
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again in a few minutes.")
    user = supabase.table("users").select("*").eq("username", req.username).execute()
    if not user.data or not verify_password(req.password, user.data[0]["password"]):
        _record_login_failure(req.username)
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    row = user.data[0]
    if is_legacy_hash(row["password"]):
        # lazy migration: we just verified the plaintext, re-hash with bcrypt
        supabase.table("users").update(
            {"password": hash_password(req.password)}
        ).eq("id", row["id"]).execute()
        logger.info("Migrated password hash to bcrypt for user: %s", req.username)
    _clear_login_failures(req.username)
    uid = str(row["id"])
    logger.info("User logged in: %s", req.username)
    return LoginResponse(user_id=uid, username=req.username, token=make_token(uid, SECRET_KEY))


# =============================================================================
# Company profile (ICP) — free-text context injected into research/email prompts
# =============================================================================

@app.get("/account/company-context")
def get_company_context(user_id: str = Depends(current_user)):
    resp = supabase.table("users").select("company_context").eq("id", user_id).execute()
    context = resp.data[0].get("company_context") if resp.data else None
    return {"company_context": context or ""}


@app.put("/account/company-context")
def set_company_context(req: CompanyProfileRequest, user_id: str = Depends(current_user)):
    supabase.table("users").update({"company_context": req.company_context}).eq("id", user_id).execute()
    return {"message": "Company profile saved."}


@app.get("/account/email-settings")
def get_email_settings(user_id: str = Depends(current_user)):
    resp = (
        supabase.table("users")
        .select("email_smtp_host,email_smtp_port,email_from_address")
        .eq("id", user_id).execute()
    )
    row = resp.data[0] if resp.data else {}
    return {
        "smtp_host": row.get("email_smtp_host") or "",
        "smtp_port": row.get("email_smtp_port") or 587,
        "from_address": row.get("email_from_address") or "",
        # Never return the password — same principle as never returning a
        # user's login password hash. The frontend shows "configured" or
        # not; re-entering it is how you change it, not editing in place.
        "configured": bool(row.get("email_from_address")),
    }


@app.put("/account/email-settings")
def set_email_settings(req: EmailSettingsRequest, user_id: str = Depends(current_user)):
    payload = {
        "email_smtp_host": req.smtp_host,
        "email_smtp_port": req.smtp_port,
        "email_from_address": req.from_address,
    }
    if req.smtp_password:
        payload["email_smtp_password"] = req.smtp_password
    else:
        existing = supabase.table("users").select("email_smtp_password").eq("id", user_id).execute()
        if not (existing.data and existing.data[0].get("email_smtp_password")):
            raise HTTPException(status_code=400, detail="App password is required for first-time setup.")
    supabase.table("users").update(payload).eq("id", user_id).execute()
    return {"message": "Email sending settings saved."}


# =============================================================================
# Lead CRUD endpoints (all ownership-checked)
# =============================================================================

def _get_owned_lead(lead_id: str, user_id: str) -> dict:
    resp = supabase.table("leads").select("*").eq("id", lead_id).execute()
    if not resp.data or str(resp.data[0].get("user_id")) != user_id:
        raise HTTPException(status_code=404, detail="Lead not found.")
    return resp.data[0]


@app.get("/leads/{user_id}")
def get_leads(
    user_id: str,
    auth_user: str = Depends(current_user),
    limit: int = Query(500, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    # 500 is generous enough that today's users never notice a
    # cap, but it's a real cap now — this used to be select() with no
    # limit at all, unbounded no matter how many leads a user accumulated.
    # limit/offset are accepted (not yet used by the frontend) so real
    # paged UI can be added later without another backend change.
    if user_id != auth_user:
        raise HTTPException(status_code=403, detail="Forbidden.")
    resp = (
        supabase.table("leads")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    return resp.data or []


@app.post("/leads")
def create_lead(lead: LeadCreate, user_id: str = Depends(current_user)):
    payload = lead.dict()
    payload["user_id"] = user_id
    resp = supabase.table("leads").insert(payload).execute()
    return resp.data[0] if resp.data else {}


@app.put("/leads/{lead_id}")
def update_lead(lead_id: str, lead: LeadUpdate, user_id: str = Depends(current_user)):
    _get_owned_lead(lead_id, user_id)
    payload = lead.dict(exclude_unset=True)  # explicit nulls clear a field
    resp = supabase.table("leads").update(payload).eq("id", lead_id).execute()
    return resp.data[0] if resp.data else {}


@app.delete("/leads/{lead_id}")
def delete_lead(lead_id: str, user_id: str = Depends(current_user)):
    _get_owned_lead(lead_id, user_id)
    supabase.table("leads").delete().eq("id", lead_id).execute()
    return {"message": "Lead deleted."}


def _split_subject(draft: str) -> tuple[str, str]:
    """The email crew drafts 'Subject: ...' as the first line — pull that out
    for the actual SMTP Subject header instead of leaving it duplicated at
    the top of the body."""
    first_line, _, rest = draft.partition("\n")
    if first_line.strip().lower().startswith("subject:"):
        return first_line.split(":", 1)[1].strip(), rest.lstrip("\n")
    return "", draft


@app.post("/leads/{lead_id}/send-email")
def send_lead_email(lead_id: str, user_id: str = Depends(current_user)):
    lead = _get_owned_lead(lead_id, user_id)
    if not lead.get("email_draft"):
        raise HTTPException(status_code=400, detail="No email draft to send — process this lead first.")
    if not lead.get("email"):
        raise HTTPException(status_code=400, detail="This lead has no email address.")

    settings = (
        supabase.table("users")
        .select("email_smtp_host,email_smtp_port,email_from_address,email_smtp_password")
        .eq("id", user_id).execute()
    )
    row = settings.data[0] if settings.data else {}
    host, port, from_addr, password = (
        row.get("email_smtp_host"), row.get("email_smtp_port"),
        row.get("email_from_address"), row.get("email_smtp_password"),
    )
    if not (host and from_addr and password):
        raise HTTPException(status_code=400, detail="Set up your email sending settings before sending.")

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    sent_today = (
        supabase.table("leads").select("id", count="exact")
        .eq("user_id", user_id).gte("email_sent_at", cutoff).execute()
    )
    if (sent_today.count or 0) >= EMAIL_SEND_DAILY_CAP:
        raise HTTPException(
            status_code=429,
            detail=f"Daily email send limit ({EMAIL_SEND_DAILY_CAP}) reached. Try again tomorrow.",
        )

    subject, body = _split_subject(lead["email_draft"])
    msg = EmailMessage()
    msg["From"] = from_addr
    msg["To"] = lead["email"]
    msg["Subject"] = subject or f"Following up — {lead.get('company') or lead['name']}"
    msg.set_content(body)

    try:
        with smtplib.SMTP(host, port, timeout=20) as server:
            server.starttls()
            server.login(from_addr, password)
            server.send_message(msg)
    except Exception as exc:
        logger.exception("Failed to send email for lead %s", lead_id)
        raise HTTPException(status_code=502, detail=f"Failed to send email: {exc}")

    sent_at = datetime.now(timezone.utc).isoformat()
    supabase.table("leads").update({"email_sent_at": sent_at}).eq("id", lead_id).execute()
    logger.info("Email sent for lead %s", lead_id)
    return {"message": "Email sent.", "sent_at": sent_at}


# =============================================================================
# Lead processing — enqueue a job; worker.py executes it
# =============================================================================

@app.post("/leads/process", status_code=202)
def process_leads_endpoint(req: ProcessLeadsRequest, user_id: str = Depends(current_user)):
    if not req.leads:
        raise HTTPException(status_code=400, detail="No leads provided.")
    if len(req.leads) > MAX_LEADS_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"At most {MAX_LEADS_PER_REQUEST} leads can be processed per request.",
        )
    lead_ids = [l.get("id") for l in req.leads]
    if any(i is None for i in lead_ids):
        raise HTTPException(status_code=400, detail="Every lead must include its id.")
    owned = (
        supabase.table("leads").select("id").eq("user_id", user_id).in_("id", lead_ids).execute()
    )
    if len(owned.data or []) != len(set(lead_ids)):
        raise HTTPException(status_code=404, detail="One or more leads not found.")

    profile = supabase.table("users").select("company_context").eq("id", user_id).execute()
    company_context = (profile.data[0].get("company_context") if profile.data else None) or ""
    if not company_context.strip():
        raise HTTPException(
            status_code=400,
            detail="Set your company profile & ICP before processing leads.",
        )

    job = {
        "user_id": user_id,
        "status": "pending",
        "leads": req.leads,
        # frozen at enqueue time so a later profile edit can't change an already-queued job
        "our_company_context": company_context,
        "force_refresh": req.force_refresh,
        # operator-held keys, not per-user — worker nulls them on completion anyway
        "gemini_api_key": GEMINI_API_KEY,
        "tavily_api_key": TAVILY_API_KEY,
    }
    resp = supabase.table("jobs").insert(job).execute()
    if not resp.data:
        raise HTTPException(status_code=500, detail="Failed to enqueue processing job.")
    job_id = resp.data[0]["id"]
    logger.info("Enqueued job %s (%d lead(s)) for user %s", job_id, len(req.leads), user_id)
    return {"job_id": job_id, "status": "pending"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, user_id: str = Depends(current_user)):
    resp = (
        supabase.table("jobs")
        .select("id,user_id,status,results,error,created_at")
        .eq("id", job_id)
        .execute()
    )
    if not resp.data or str(resp.data[0].get("user_id")) != user_id:
        raise HTTPException(status_code=404, detail="Job not found.")
    job = resp.data[0]
    job.pop("user_id", None)
    return job


@app.get("/analysis/{lead_id}")
def get_analysis(lead_id: str, user_id: str = Depends(current_user)):
    _get_owned_lead(lead_id, user_id)
    try:
        resp = supabase.table("analysis_runs").select("*").eq("lead_id", lead_id).execute()
    except Exception as exc:
        logger.error("Failed to fetch analysis for lead %s: %s", lead_id, exc)
        raise HTTPException(status_code=500, detail="Failed to fetch analysis data.")
    if not resp.data:
        raise HTTPException(status_code=404, detail="No analysis data found.")
    return resp.data[0]


# =============================================================================
# Result persistence — called by worker.py after the crews finish
# =============================================================================

def persist_results(
    leads: list, scores: list, emails: list, agent_times: dict, cache_hits: list, elapsed: float,
) -> list:
    """
    Write scores/emails/analysis to Supabase for each processed lead.
    `emails` is aligned with `leads` (None where the lead scored below the
    email threshold). `cache_hits` is aligned with `leads` (True where
    company research was served from cache). Returns a summary list for the
    job record.
    """
    results = []
    for lead, score_obj, email_draft, cache_hit in zip(leads, scores, emails, cache_hits):
        with lead_context(lead["id"]):
            results.append(_persist_one_lead(lead, score_obj, email_draft, agent_times, cache_hit, elapsed))
    return results


def _persist_one_lead(
    lead: dict, score_obj, email_draft, agent_times: dict, cache_hit: bool, elapsed: float,
) -> dict:
    pyd = score_obj.pydantic
    update_payload = {
        "score":          pyd.lead_score.score,
        "scoring_result": pyd.dict(),
    }
    if email_draft is not None:
        update_payload["email_draft"] = email_draft.raw
    supabase.table("leads").update(update_payload).eq("id", lead["id"]).execute()

    # Token usage is reported per crew, not per agent; the per-agent rows
    # below split the crew total evenly — estimates, not measurements.
    score_usage = score_obj.token_usage
    email_usage = email_draft.token_usage if email_draft else None
    score_tokens  = getattr(score_usage, "total_tokens",      0) or 0
    email_tokens  = getattr(email_usage, "total_tokens",      0) or 0
    score_prompt  = getattr(score_usage, "prompt_tokens",     0) or 0
    score_compl   = getattr(score_usage, "completion_tokens", 0) or 0
    email_prompt  = getattr(email_usage, "prompt_tokens",     0) or 0
    email_compl   = getattr(email_usage, "completion_tokens", 0) or 0

    total_tokens = score_tokens + email_tokens
    # gemini-2.5-flash pricing: $0.15/M input, $0.60/M output
    total_cost = round(
        (score_prompt + email_prompt) * 0.15 / 1_000_000
        + (score_compl + email_compl) * 0.60 / 1_000_000,
        6,
    )
    cost_per_token = (total_cost / total_tokens) if total_tokens else 0.0

    score_tasks = score_obj.tasks_output or []
    email_tasks = (email_draft.tasks_output if email_draft else None) or []
    per_score = score_tokens // len(score_tasks) if score_tasks else 0
    per_email = email_tokens // len(email_tasks) if email_tasks else 0

    agents_data = []
    for tasks, per_tokens in ((score_tasks, per_score), (email_tasks, per_email)):
        for t in tasks:
            name = t.agent if isinstance(t.agent, str) else getattr(t.agent, "role", str(t.agent))
            agents_data.append({
                "agent": name, "status": "Success",
                "tokens": per_tokens,
                "cost": round(per_tokens * cost_per_token, 6),
                "time_seconds": agent_times.get(name),
            })
    if cache_hit:
        # company research was served from cache — no LLM/search call made for it
        agents_data.append({
            "agent": "Company Research & Cultural Fit Analyst",
            "status": "Cached", "tokens": 0, "cost": 0.0, "time_seconds": 0,
        })
    if email_draft is None:
        # score <= 70 — the email crew never ran for this lead. Shown at 0
        # rather than omitted, so the breakdown always lists all 4 agents.
        agents_data.append({
            "agent": "Email Specialist",
            "status": "Skipped", "tokens": 0, "cost": 0.0, "time_seconds": 0,
        })

    analysis_record = {
        "lead_id":          lead["id"],
        "duration_seconds": elapsed,
        "total_tokens":     total_tokens,
        "total_cost":       total_cost,
        "success_rate":     100.0,
        "agents_executed":  len(agents_data),
        "agents_data":      agents_data,
    }
    try:
        existing = supabase.table("analysis_runs").select("id").eq("lead_id", lead["id"]).execute()
        if existing.data:
            supabase.table("analysis_runs").update(analysis_record).eq("lead_id", lead["id"]).execute()
        else:
            supabase.table("analysis_runs").insert(analysis_record).execute()
    except Exception as exc:
        logger.warning("Failed to save analysis for lead %s: %s", lead.get("name"), exc)

    logger.info("Processed lead %s — score %s", lead.get("name"), pyd.lead_score.score)
    return {
        "lead_id": lead["id"],
        "score": pyd.lead_score.score,
        "email_drafted": email_draft is not None,
    }


# ---------------------------------------------------------------------------
# Optional in-process worker — for deploys without a separate worker service
# (e.g. Render free tier). Set RUN_WORKER_IN_PROCESS=1 to enable; run
# worker.py as its own process instead when you can.
# ---------------------------------------------------------------------------
if os.getenv("RUN_WORKER_IN_PROCESS") == "1":
    import threading

    @app.on_event("startup")
    def _start_worker_thread():
        import asyncio
        from worker import main as worker_main  # lazy import: worker imports this module
        # worker.main() is a coroutine (concurrent job processing) — needs its
        # own event loop, since this thread isn't the one FastAPI/uvicorn runs.
        threading.Thread(
            target=lambda: asyncio.run(worker_main()), daemon=True, name="job-worker",
        ).start()
        logger.info("In-process worker thread started (RUN_WORKER_IN_PROCESS=1)")
