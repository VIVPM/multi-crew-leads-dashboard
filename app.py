import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"   # prevent signal handler errors in Streamlit threads

import asyncio
import hashlib
import warnings
import sys
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — ensure pipeline.py is always importable
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# Change working directory so relative YAML paths in pipeline.py resolve correctly
os.chdir(BASE_DIR)

from pipeline import process_leads   # single import from the CrewAI side


# =============================================================================
# API key loading  (st.secrets on cloud, .env locally)
# =============================================================================

def load_api_keys():
    # Only Supabase keys come from secrets; Sambanova + Serper are entered via sidebar
    required_keys = ["SUPABASE_URL", "SUPABASE_KEY"]
    try:
        for key in required_keys:
            if key in st.secrets:
                os.environ[key] = st.secrets[key]
        return
    except Exception:
        pass
    try:
        load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    except ImportError:
        pass


load_api_keys()

# ---------------------------------------------------------------------------
# Supabase client (loaded after env is ready)
# ---------------------------------------------------------------------------
from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Sales Pipeline — Lead Scoring & Email",
    page_icon="🎯",
    layout="wide",
)

# =============================================================================
# Session-state defaults
# =============================================================================

for _key, _default in [
    ("logged_in", False),
    ("user_id", None),
    ("show_signup", False),
    ("show_login", True),
    ("adding_lead", False),
    ("editing_lead", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# =============================================================================
# Helpers
# =============================================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def refresh_leads():
    if not st.session_state.logged_in or not st.session_state.user_id:
        st.session_state.leads = []
        return
    resp = (
        supabase.table("leads")
        .select("*")
        .eq("user_id", st.session_state.user_id)
        .order("created_at", desc=True)
        .execute()
    )
    st.session_state.leads = resp.data or []


def reset_lead_form_cache(lead: dict | None = None):
    if lead:
        st.session_state["Name"]        = lead.get("name", "")
        st.session_state["Job Title"]   = lead.get("job_title", "")
        st.session_state["Company"]     = lead.get("company", "")
        st.session_state["Email"]       = lead.get("email", "")
        st.session_state["Use Case"]    = lead.get("use_case", "")
        st.session_state["Industry"]    = lead.get("industry", "")
        st.session_state["Location"]    = lead.get("location", "")
        st.session_state["Lead Source"] = lead.get("source", "")
    else:
        for f in ["Name", "Job Title", "Company", "Email", "Use Case", "Industry", "Location", "Lead Source"]:
            st.session_state.setdefault(f, "")


# =============================================================================
# Auth UI  (shown only when not logged in)
# =============================================================================

if not st.session_state.logged_in:
    st.markdown(
        "<h1 style='text-align:center'>🎯 Sales Pipeline — Lead Scoring & Email Generation</h1>",
        unsafe_allow_html=True,
    )

    # Narrow centered column for auth forms
    _, auth_col, _ = st.columns([1, 2, 1])

    with auth_col:
        if st.session_state.show_signup:
            st.markdown("<h3 style='text-align:center'>Signup</h3>", unsafe_allow_html=True)
            with st.form("signup_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Signup", use_container_width=True):
                    if username and password:
                        existing = supabase.table("users").select("*").eq("username", username).execute()
                        if existing.data:
                            st.error("Username already exists.")
                        else:
                            supabase.table("users").insert(
                                {"username": username, "password": hash_password(password)}
                            ).execute()
                            st.success("Signup successful. Please login.")
                            st.session_state.show_signup = False
                            st.session_state.show_login = True
                            st.rerun()
                    else:
                        st.error("Please fill in all fields.")
            if st.button("Back to Login", use_container_width=True):
                st.session_state.show_signup = False
                st.session_state.show_login = True
                st.rerun()

        if st.session_state.show_login:
            st.markdown("<h3 style='text-align:center'>Login</h3>", unsafe_allow_html=True)
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Login", use_container_width=True):
                    if username and password:
                        user = supabase.table("users").select("*").eq("username", username).execute()
                        if user.data and hash_password(password) == user.data[0]["password"]:
                            st.session_state.logged_in = True
                            st.session_state.user_id = user.data[0]["id"]
                            st.session_state.show_login = False
                            st.session_state.pop("leads", None)
                            st.success("Login successful.")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                    else:
                        st.error("Please fill in all fields.")
            if st.button("Create new account", use_container_width=True):
                st.session_state.show_login = False
                st.session_state.show_signup = True
                st.rerun()

    st.stop()


# =============================================================================
# Logged-in area
# =============================================================================

st.title("🎯 Sales Pipeline — Lead Scoring & Email Generation")

# --- Sidebar ---
st.sidebar.header("🔑 API Keys")

sambana_key = st.sidebar.text_input("Sambanova API Key", type="password")
st.sidebar.markdown("[Get a Sambanova API key →](https://cloud.sambanova.ai/)")

serper_key = st.sidebar.text_input("Serper API Key", type="password")
st.sidebar.markdown("[Get a Serper API key →](https://serper.dev/)")

# Inject Serper key into env so SerperDevTool picks it up automatically
if serper_key:
    os.environ["SERPER_API_KEY"] = serper_key

st.sidebar.divider()

if st.sidebar.button("🚪 Log Out"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.pop("leads", None)
    st.session_state.show_login = True
    st.sidebar.success("Logout successful.")
    st.rerun()

if not sambana_key or not serper_key:
    missing = [k for k, v in [("Sambanova", sambana_key), ("Serper", serper_key)] if not v]
    st.sidebar.warning(f"Enter your {' & '.join(missing)} API key(s) above to use the crew.")

# Always refresh leads on every rerun while logged in
refresh_leads()


# =============================================================================
# Lead controls
# =============================================================================

col_add, col_clear, _ = st.columns([1, 1, 6])

with col_add:
    if st.button("➕ Add New Lead"):
        st.session_state.adding_lead = True
        st.session_state.editing_lead = None
        reset_lead_form_cache()
        st.rerun()

with col_clear:
    if st.button("🗑️ Clear Form"):
        st.session_state.adding_lead = False
        st.rerun()


# --- Lead add / edit form ---
if st.session_state.adding_lead:
    reset_lead_form_cache()
    defaults = {
        "name":      st.session_state.get("Name", ""),
        "job_title": st.session_state.get("Job Title", ""),
        "company":   st.session_state.get("Company", ""),
        "email":     st.session_state.get("Email", ""),
        "use_case":  st.session_state.get("Use Case", ""),
        "industry":  st.session_state.get("Industry", ""),
        "location":  st.session_state.get("Location", ""),
        "source":    st.session_state.get("Lead Source", "") or "Website",
    }

    with st.form("lead_form"):
        r1c1, r1c2 = st.columns(2)
        name      = r1c1.text_input("Name",      value=defaults["name"])
        job_title = r1c2.text_input("Job Title", value=defaults["job_title"])
        r2c1, r2c2 = st.columns(2)
        company   = r2c1.text_input("Company",   value=defaults["company"])
        email     = r2c2.text_input("Email",     value=defaults["email"])
        r3c1, r3c2 = st.columns(2)
        use_case  = r3c1.text_input("Use Case",  value=defaults["use_case"])
        industry  = r3c2.text_input("Industry",  value=defaults["industry"])
        r4c1, r4c2 = st.columns(2)
        location  = r4c1.text_input("Location",  value=defaults["location"])
        _sources  = ["Website", "Referral", "Event", "Social Media", "Other"]
        source    = r4c2.selectbox(
            "Lead Source", _sources,
            index=_sources.index(defaults["source"]) if defaults["source"] in _sources else 0,
        )

        if st.form_submit_button("💾 Save Lead"):
            if st.session_state.editing_lead:
                supabase.table("leads").update({
                    "name": name, "job_title": job_title, "company": company,
                    "email": email, "use_case": use_case, "industry": industry,
                    "location": location, "source": source,
                }).eq("id", st.session_state.editing_lead).execute()
                st.success("Lead updated.")
                st.session_state.editing_lead = None
            else:
                supabase.table("leads").insert({
                    "name": name, "job_title": job_title, "company": company,
                    "email": email, "use_case": use_case, "industry": industry,
                    "location": location, "source": source,
                    "user_id": st.session_state.user_id,
                }).execute()
                st.success("Lead added.")
            st.session_state.adding_lead = False
            refresh_leads()
            st.rerun()


# =============================================================================
# Process leads button
# =============================================================================

if st.button("⚡ Process Leads (Score + Email)"):
    if not sambana_key or not serper_key:
        st.error("Please enter both your Sambanova and Serper API Keys in the sidebar first.")
    else:
        unprocessed = [l for l in st.session_state.leads if l.get("score") is None]
        if not unprocessed:
            st.info("No new leads to process — all leads already have a score.")
        else:
            with st.spinner(f"Processing {len(unprocessed)} lead(s) with AI crew…"):
                try:
                    raw_inputs = [{"lead_data": lead} for lead in unprocessed]
                    scores, emails = asyncio.run(process_leads(raw_inputs, sambana_key))

                    for lead, score_obj, email_draft in zip(unprocessed, scores, emails):
                        pyd = score_obj.pydantic
                        supabase.table("leads").update({
                            "score":          pyd.lead_score.score,
                            "scoring_result": pyd.dict(),
                            "email_draft":    email_draft.raw,
                        }).eq("id", lead["id"]).execute()

                    refresh_leads()
                    st.success("✅ Leads processed and scores saved!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing error: {e}")


# =============================================================================
# Dashboard — charts
# =============================================================================

st.subheader("📊 Leads Dashboard")

if st.session_state.leads:
    df = pd.DataFrame(st.session_state.leads)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    industry_counts = (
        df["industry"].fillna("Unknown").value_counts()
        if "industry" in df.columns and not df["industry"].isnull().all() else None
    )
    source_counts = (
        df["source"].fillna("Unknown").value_counts()
        if "source" in df.columns and not df["source"].isnull().all() else None
    )
    score_series = (
        df["score"].dropna()
        if "score" in df.columns and not df["score"].isnull().all() else None
    )
    leads_per_day = (
        df.resample("D", on="created_at").size()
        if "created_at" in df.columns and not df["created_at"].isnull().all() else None
    )
    avg_score_industry = (
        df.groupby(df["industry"].fillna("Unknown"))["score"].mean().sort_values()
        if {"industry", "score"}.issubset(df.columns)
        and not df["industry"].isnull().all()
        and not df["score"].isnull().all()
        else None
    )
    country_counts = None
    if "location" in df.columns and not df["location"].isnull().all():
        df["country"] = df["location"].fillna("Unknown").apply(
            lambda x: x.split(",")[-1].strip() if "," in str(x) else str(x).strip()
        )
        country_counts = df["country"].value_counts()

    # Row 1
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        st.markdown("#### Leads by Industry")
        if industry_counts is not None:
            st.bar_chart(industry_counts)
        else:
            st.info("No industry data")

    with c2:
        st.markdown("#### Leads by Source")
        if source_counts is not None:
            fig, ax = plt.subplots()
            source_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.info("No source data")

    with c3:
        st.markdown("#### Score Distribution")
        if score_series is not None and not score_series.empty:
            fig, ax = plt.subplots()
            ax.hist(score_series, bins=10, edgecolor="black")
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No score data yet")

    # Row 2
    d1, d2, d3 = st.columns(3, gap="small")
    with d1:
        st.markdown("#### Leads Over Time")
        if leads_per_day is not None and not leads_per_day.empty:
            st.line_chart(leads_per_day)
        else:
            st.info("No timestamp data")

    with d2:
        st.markdown("#### Avg Score by Industry")
        if avg_score_industry is not None and not avg_score_industry.empty:
            st.bar_chart(avg_score_industry)
        else:
            st.info("Insufficient score/industry data")

    with d3:
        st.markdown("#### Leads by Country")
        if country_counts is not None and not country_counts.empty:
            fig, ax = plt.subplots()
            country_counts.plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)
        else:
            st.info("No location data")
else:
    st.info("No leads yet — add some leads to see analytics.")


# =============================================================================
# Leads data table with search
# =============================================================================

st.subheader("📋 Leads Data")


def _flatten_to_text(obj) -> str:
    """Recursively flatten a dict/list/primitive to a single lowercase string."""
    try:
        if isinstance(obj, dict):
            return " ".join(_flatten_to_text(v) for v in obj.values()).lower()
        if isinstance(obj, list):
            return " ".join(_flatten_to_text(v) for v in obj).lower()
        return ("" if obj is None else str(obj)).lower()
    except Exception:
        return str(obj).lower()


search_q = st.text_input("🔍 Search leads (name, company, email, location, score, …)")

leads_src = st.session_state.leads or []

if search_q:
    tokens = [t.strip().lower() for t in search_q.split() if t.strip()]
    filtered_leads = [
        lead for lead in leads_src
        if all(t in _flatten_to_text(lead) for t in tokens)
    ]
else:
    filtered_leads = leads_src

st.caption(f"Showing {len(filtered_leads)} of {len(leads_src)} lead(s)")

if filtered_leads:
    for lead in filtered_leads:
        title = f"{lead.get('name', '')} – {lead.get('company', '')}"
        if lead.get("score") is not None:
            title += f"  •  Score: {lead['score']}"

        with st.expander(title):
            st.json({
                "Name":      lead.get("name"),
                "Job Title": lead.get("job_title"),
                "Company":   lead.get("company"),
                "Email":     lead.get("email"),
                "Use Case":  lead.get("use_case"),
                "Industry":  lead.get("industry", "N/A"),
                "Location":  lead.get("location", "N/A"),
                "Source":    lead.get("source", "N/A"),
            })

            if lead.get("scoring_result"):
                st.markdown("**Scoring Result:**")
                st.json(lead["scoring_result"])
            if lead.get("email_draft"):
                st.markdown("**Generated Email Draft:**")
                st.text(lead["email_draft"])

            btn1, btn2, btn3 = st.columns(3, gap="small")
            with btn1:
                if lead.get("score") is None:
                    if st.button("✏️ Edit", key=f"edit_{lead['id']}"):
                        st.session_state.editing_lead = lead["id"]
                        reset_lead_form_cache(lead)
                        st.session_state.adding_lead = True
                        st.rerun()
                else:
                    st.caption("✅ Processed")
            with btn2:
                if st.button("🗑️ Delete", key=f"del_{lead['id']}"):
                    supabase.table("leads").delete().eq("id", lead["id"]).execute()
                    refresh_leads()
                    st.success("Lead deleted.")
                    st.rerun()
            with btn3:
                if st.button("🔄 Refresh", key=f"refresh_{lead['id']}"):
                    refresh_leads()
                    st.rerun()
else:
    st.info("No leads match your search." if search_q else "No leads yet.")
