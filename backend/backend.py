"""
backend.py — FastAPI server wrapping CrewAI pipeline + Supabase operations.
Run with: uvicorn backend.backend:app --host 0.0.0.0 --port 8000 --reload
"""

import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"

import asyncio
import hashlib
import logging
import sys
from typing import Optional, List, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("backend")

# ---------------------------------------------------------------------------
# Path + env setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
os.chdir(BASE_DIR)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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


# =============================================================================
# Helpers
# =============================================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(password: str, hashed: str) -> bool:
    return hash_password(password) == hashed


# =============================================================================
# Request / Response models
# =============================================================================

class SignupRequest(BaseModel):
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    user_id: str
    username: str

class LeadCreate(BaseModel):
    name: str
    job_title: Optional[str] = None
    company: str
    email: str
    use_case: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = "Website"
    user_id: str

class LeadUpdate(BaseModel):
    name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    email: Optional[str] = None
    use_case: Optional[str] = None
    industry: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None

class ProcessLeadsRequest(BaseModel):
    leads: List[dict]
    sambanova_api_key: str
    tavily_api_key: str


# =============================================================================
# Auth endpoints
# =============================================================================

@app.post("/auth/signup")
def signup(req: SignupRequest):
    existing = supabase.table("users").select("*").eq("username", req.username).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Username already exists.")
    supabase.table("users").insert(
        {"username": req.username, "password": hash_password(req.password)}
    ).execute()
    logger.info("New user signed up: %s", req.username)
    return {"message": "Signup successful."}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    user = supabase.table("users").select("*").eq("username", req.username).execute()
    if not user.data or not verify_password(req.password, user.data[0]["password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    logger.info("User logged in: %s", req.username)
    return LoginResponse(user_id=str(user.data[0]["id"]), username=req.username)


# =============================================================================
# Lead CRUD endpoints
# =============================================================================

@app.get("/leads/{user_id}")
def get_leads(user_id: str):
    resp = (
        supabase.table("leads")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )
    return resp.data or []


@app.post("/leads")
def create_lead(lead: LeadCreate):
    resp = supabase.table("leads").insert(lead.dict()).execute()
    return resp.data[0] if resp.data else {}


@app.put("/leads/{lead_id}")
def update_lead(lead_id: str, lead: LeadUpdate):
    payload = {k: v for k, v in lead.dict().items() if v is not None}
    resp = supabase.table("leads").update(payload).eq("id", lead_id).execute()
    return resp.data[0] if resp.data else {}


@app.delete("/leads/{lead_id}")
def delete_lead(lead_id: str):
    supabase.table("leads").delete().eq("id", lead_id).execute()
    return {"message": "Lead deleted."}


# =============================================================================
# CrewAI process endpoint
# =============================================================================

@app.post("/leads/process")
async def process_leads_endpoint(req: ProcessLeadsRequest):
    from pipeline import process_leads

    os.environ["TAVILY_API_KEY"] = req.tavily_api_key

    raw_inputs = [{"lead_data": lead} for lead in req.leads]
    try:
        scores, emails = await asyncio.wait_for(
            process_leads(raw_inputs, req.sambanova_api_key),
            timeout=300,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Processing timed out.")

    results = []
    for lead, score_obj, email_draft in zip(req.leads, scores, emails):
        pyd = score_obj.pydantic
        update_payload = {
            "score":          pyd.lead_score.score,
            "scoring_result": pyd.dict(),
            "email_draft":    email_draft.raw,
        }
        supabase.table("leads").update(update_payload).eq("id", lead["id"]).execute()
        results.append({"lead_id": lead["id"], **update_payload})
        logger.info("Processed lead %s — score %s", lead.get("name"), pyd.lead_score.score)

    return {"processed": len(results), "results": results}
