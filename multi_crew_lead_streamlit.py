import os
import asyncio
import hashlib
import warnings
warnings.filterwarnings('ignore')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from supabase import create_client

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

from crewai import Agent, Task, Crew, LLM, Flow
from crewai.flow.flow import listen, start
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

# =========================
# Env & Supabase
# =========================
load_dotenv(dotenv_path='.env')
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# Pydantic schemas
# =========================
class LeadPersonalInfo(BaseModel):
    name: str
    job_title: str
    role_relevance: int
    professional_background: Optional[str] = None
    years_experience: Optional[int] = None
    linkedin_url: Optional[str] = None
    location: Optional[str] = None

class CompanyInfo(BaseModel):
    company_name: str
    industry: str
    company_size: int
    revenue: Optional[float] = None
    market_presence: int
    company_location: Optional[str] = None
    founding_year: Optional[int] = None
    website: Optional[str] = None

class LeadScore(BaseModel):
    score: int
    scoring_criteria: List[str]
    validation_notes: Optional[str] = None
    demographic_score: int
    firmographic_score: int
    behavioral_score: int

class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo
    company_info: CompanyInfo
    lead_score: LeadScore

# =========================
# YAML configs load
# =========================
files = {
    'lead_agents': 'config/lead_qualification_agents.yaml',
    'lead_tasks': 'config/lead_qualification_tasks.yaml',
    'email_agents': 'config/email_engagement_agents.yaml',
    'email_tasks': 'config/email_engagement_tasks.yaml'
}
configs: Dict[str, dict] = {}
for k, path in files.items():
    with open(path, 'r', encoding='utf-8') as fh:
        configs[k] = yaml.safe_load(fh)

lead_agents_config = configs['lead_agents']
lead_tasks_config = configs['lead_tasks']
email_agents_config = configs['email_agents']
email_tasks_config = configs['email_tasks']

# =========================
# Session defaults
# =========================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'adding_lead' not in st.session_state:
    st.session_state.adding_lead = False
if 'editing_lead' not in st.session_state:
    st.session_state.editing_lead = None

# =========================
# Helpers
# =========================
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

def reset_lead_form_cache(lead: Optional[dict] = None):
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
        st.session_state["Name"] = st.session_state.get("Name", "")
        st.session_state["Job Title"] = st.session_state.get("Job Title", "")
        st.session_state["Company"] = st.session_state.get("Company", "")
        st.session_state["Email"] = st.session_state.get("Email", "")
        st.session_state["Use Case"] = st.session_state.get("Use Case", "")
        st.session_state["Industry"] = st.session_state.get("Industry", "")
        st.session_state["Location"] = st.session_state.get("Location", "")
        st.session_state["Lead Source"] = st.session_state.get("Lead Source", "")

# =========================
# UI: Auth
# =========================
if not st.session_state.logged_in:
    st.title("Welcome to the Sales Pipeline Lead Scoring and Email Generation")

    if st.session_state.show_signup:
        st.subheader("Signup")
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")
            if submit:
                if username and password:
                    existing = supabase.table("users").select("*").eq("username", username).execute()
                    if existing.data:
                        st.error("Username already exists.")
                    else:
                        supabase.table("users").insert({
                            "username": username,
                            "password": hash_password(password)
                        }).execute()
                        st.success("Signup successful. Please login.")
                        st.session_state.show_signup = False
                        st.session_state.show_login = True
                        st.rerun()
                else:
                    st.error("Please fill in all fields.")

    if st.session_state.show_login:
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Login")
            if submit:
                if username and password:
                    user = supabase.table("users").select("*").eq("username", username).execute()
                    if user.data and hash_password(password) == user.data[0]["password"]:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user.data[0]["id"]
                        st.session_state.show_login = False
                        st.session_state.pop("leads", None)   # kill stale cache
                        st.success("Login successful.")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Please fill in all fields.")
        if st.button("Create new account"):
            st.session_state.show_login = False
            st.session_state.show_signup = True
            st.rerun()
    st.stop()

# =========================
# Logged-in area
# =========================
st.title("Sales Pipeline Lead Scoring and Email Generation")

st.sidebar.header("🔑 Enter your API keys")
sambana_key = st.sidebar.text_input("Sambanova API Key", type="password")

if st.sidebar.button("🚪 Log Out"):
    st.sidebar.success("Logout successful.")
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.pop("leads", None)  # delete cache key
    st.session_state.show_login = True
    st.rerun()

if not sambana_key:
    st.sidebar.warning("Please enter Sambanova API Key above to continue")
    st.stop()

# =========================
# LLM & Agents
# =========================
llm3 = LLM(model="sambanova/Meta-Llama-3.3-70B-Instruct", api_key=sambana_key)

lead_data_agent = Agent(
    config=lead_agents_config['lead_data_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm3
)
cultural_fit_agent = Agent(
    config=lead_agents_config['cultural_fit_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm3
)
scoring_validation_agent = Agent(
    config=lead_agents_config['scoring_validation_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm3
)

lead_data_task = Task(config=lead_tasks_config['lead_data_collection'], agent=lead_data_agent)
cultural_fit_task = Task(config=lead_tasks_config['cultural_fit_analysis'], agent=cultural_fit_agent)
scoring_validation_task = Task(
    config=lead_tasks_config['lead_scoring_and_validation'],
    agent=scoring_validation_agent,
    context=[lead_data_task, cultural_fit_task],
    output_pydantic=LeadScoringResult
)

lead_scoring_crew = Crew(
    agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
    tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
    verbose=True
)

email_content_specialist = Agent(config=email_agents_config['email_content_specialist'], llm=llm3)
engagement_strategist = Agent(config=email_agents_config['engagement_strategist'], llm=llm3)

email_drafting = Task(config=email_tasks_config['email_drafting'], agent=email_content_specialist)
engagement_optimization = Task(config=email_tasks_config['engagement_optimization'],
                               context=[email_drafting],
                               agent=engagement_strategist)

email_writing_crew = Crew(
    agents=[email_content_specialist, engagement_strategist],
    tasks=[email_drafting, engagement_optimization],
    verbose=True
)

# =========================
# Data bootstrap
# =========================
refresh_leads()  # always refresh on rerun while logged in

class SalesPipeline(Flow):
    def __init__(self, leads):
        super().__init__()
        self.leads = leads

    @start()
    def score_leads(self):
        scores = lead_scoring_crew.kickoff_for_each(self.leads)
        self.state["scores"] = scores
        return scores

    @listen(score_leads)
    def store_leads_score(self, scores):
        return scores

    @listen(score_leads)
    def filter_leads(self, scores):
        return [score for score in scores if score['lead_score'].score > 70]

    @listen(filter_leads)
    def write_email(self, leads):
        scored_leads = [lead.to_dict() for lead in leads]
        emails = email_writing_crew.kickoff_for_each(scored_leads)
        return emails

    @listen(write_email)
    def send_email(self, emails):
        self.state["emails"] = emails
        return emails

async def process_leads(leads):
    flow = SalesPipeline(leads)
    await flow.kickoff_async()
    return flow.state["scores"], flow.state["emails"]

# =========================
# Lead form controls
# =========================
if st.button("Add New Lead"):
    st.session_state.adding_lead = True
    st.session_state.editing_lead = None
    reset_lead_form_cache()
    st.rerun()

if st.session_state.adding_lead:
    reset_lead_form_cache()
    defaults = {
        "name": st.session_state.get("Name", ""),
        "job_title": st.session_state.get("Job Title", ""),
        "company": st.session_state.get("Company", ""),
        "email": st.session_state.get("Email", ""),
        "use_case": st.session_state.get("Use Case", ""),
        "industry": st.session_state.get("Industry", ""),
        "location": st.session_state.get("Location", ""),
        "source": st.session_state.get("Lead Source", "") or "Website",
    }

    with st.form("lead_form"):
        name      = st.text_input("Name",      value=defaults["name"])
        job_title = st.text_input("Job Title", value=defaults["job_title"])
        company   = st.text_input("Company",   value=defaults["company"])
        email     = st.text_input("Email",     value=defaults["email"])
        use_case  = st.text_input("Use Case",  value=defaults["use_case"])
        industry  = st.text_input("Industry",  value=defaults["industry"])
        location  = st.text_input("Location",  value=defaults["location"])
        source    = st.selectbox(
            "Lead Source",
            ["Website", "Referral", "Event", "Social Media", "Other"],
            index=["Website", "Referral", "Event", "Social Media", "Other"].index(defaults["source"])
            if defaults["source"] in ["Website", "Referral", "Event", "Social Media", "Other"] else 0
        )
        submit = st.form_submit_button("Save Lead")

        if submit:
            if st.session_state.editing_lead:
                supabase.table("leads").update({
                    "name": name, "job_title": job_title, "company": company, "email": email,
                    "use_case": use_case, "industry": industry, "location": location, "source": source
                }).eq("id", st.session_state.editing_lead).execute()
                st.success("Lead updated.")
                st.session_state.editing_lead = None
            else:
                # let DB default set created_at (TIMESTAMPTZ DEFAULT now())
                new_row = {
                    "name": name, "job_title": job_title, "company": company, "email": email,
                    "use_case": use_case, "industry": industry, "location": location,
                    "source": source, "user_id": st.session_state.user_id
                }
                supabase.table("leads").insert(new_row).execute()
                st.success("Lead added.")
            st.session_state.adding_lead = False
            refresh_leads()
            st.rerun()

# =========================
# Actions
# =========================
if st.button("Process Leads"):
    unprocessed = [l for l in st.session_state.leads if l.get("score") is None]
    if not unprocessed:
        st.info("No new leads to process.")
    else:
        with st.spinner("Processing new leads…"):
            try:
                raw_inputs = [{"lead_data": lead} for lead in unprocessed]
                scores, emails = asyncio.run(process_leads(raw_inputs))

                for lead, score_obj, email_draft in zip(unprocessed, scores, emails):
                    pyd = score_obj.pydantic  # CrewAI pydantic output
                    updates = {
                        "score": pyd.lead_score.score,
                        "scoring_result": pyd.dict(),
                        "email_draft": email_draft.raw
                    }
                    supabase.table("leads").update(updates).eq("id", lead["id"]).execute()

                refresh_leads()
                st.success("Leads processed and updated.")
                st.rerun()
            except Exception as e:
                st.error(f"Processing error: {e}")

if st.button("Clear Leads"):
    st.session_state.adding_lead = False
    st.success("Leads cleared from the form.")
    st.rerun()

# =========================
# Visualization: Dashboard-first layout + searchable Leads Data
# =========================
st.subheader("Leads Dashboard")  # big title above graphs

if st.session_state.leads:
    df = pd.DataFrame(st.session_state.leads)

    # Normalize types
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    # Precompute series safely
    industry_counts = (
        df['industry'].fillna("Unknown").value_counts()
        if 'industry' in df.columns and not df['industry'].isnull().all() else None
    )
    source_counts = (
        df['source'].fillna("Unknown").value_counts()
        if 'source' in df.columns and not df['source'].isnull().all() else None
    )
    score_series = (
        df['score'].dropna()
        if 'score' in df.columns and not df['score'].isnull().all() else None
    )
    leads_per_day = (
        df.resample('D', on='created_at').size()
        if 'created_at' in df.columns and not df['created_at'].isnull().all() else None
    )
    avg_score_industry = (
        df.groupby(df['industry'].fillna("Unknown"))['score'].mean().sort_values()
        if {'industry','score'}.issubset(df.columns)
        and not df['industry'].isnull().all()
        and not df['score'].isnull().all()
        else None
    )
    location_counts = (
        df['location'].fillna("Unknown").value_counts().head(10)
        if 'location' in df.columns and not df['location'].isnull().all() else None
    )

    # --- Row 1: 3 charts ---
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
            source_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")  # cleaner
            st.pyplot(fig)
        else:
            st.info("No source data")

    with c3:
        st.markdown("#### Score Distribution")
        if score_series is not None and not score_series.empty:
            fig, ax = plt.subplots()
            ax.hist(score_series, bins=10, edgecolor='black')
            ax.set_xlabel("Score"); ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No score data")

    # --- Row 2: 3 charts ---
    d1, d2, d3 = st.columns(3, gap="small")

    with d1:
        st.markdown("#### Leads Over Time")
        if leads_per_day is not None and not leads_per_day.empty:
            st.line_chart(leads_per_day)
        else:
            st.info("No timestamps")

    with d2:
        st.markdown("#### Average Score by Industry")
        if avg_score_industry is not None and not avg_score_industry.empty:
            st.bar_chart(avg_score_industry)
        else:
            st.info("Insufficient score/industry data")

    with d3:
        st.markdown("#### Leads by Location")
        if location_counts is not None:
            st.bar_chart(location_counts)
        else:
            st.info("No location data")
else:
    st.info("No leads available for analytics.")

# =========================
# Leads Data (below graphs) with search
# =========================
st.subheader("Leads Data")

def _flatten_to_text(obj) -> str:
    """Flatten any nested dict/list/primitive to a single lowercase string."""
    try:
        if isinstance(obj, dict):
            parts = []
            for k, v in obj.items():
                parts.append(str(k))
                parts.append(_flatten_to_text(v))
            return " ".join(parts).lower()
        if isinstance(obj, list):
            return " ".join(_flatten_to_text(v) for v in obj).lower()
        return ("" if obj is None else str(obj)).lower()
    except Exception:
        return str(obj).lower()

search_q = st.text_input("Search leads (matches any field, e.g., name, company, email, location, score)")

leads_src = st.session_state.leads if st.session_state.leads else []

if search_q:
    tokens = [t.strip().lower() for t in search_q.split() if t.strip()]
    filtered_leads = []
    for lead in leads_src:
        haystack = _flatten_to_text(lead)
        if all(t in haystack for t in tokens):
            filtered_leads.append(lead)
else:
    filtered_leads = leads_src

st.caption(f"Showing {len(filtered_leads)} of {len(leads_src)} leads")

if filtered_leads:
    for lead in filtered_leads:
        title = f"{lead.get('name','')} – {lead.get('company','')}"
        if lead.get("score") is not None:
            title += f" (Score: {lead['score']})"

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

            c1, c2, c3 = st.columns(3, gap="small")
            with c1:
                if lead.get("score") is None:
                    if st.button("Edit", key=f"edit_{lead['id']}"):
                        st.session_state.editing_lead = lead["id"]
                        reset_lead_form_cache(lead)
                        st.session_state.adding_lead = True
                        st.rerun()
                else:
                    st.caption('Processed')
                    
            with c2:
                if st.button("Delete", key=f"del_{lead['id']}"):
                    supabase.table("leads").delete().eq("id", lead["id"]).execute()
                    refresh_leads()
                    st.success("Lead deleted.")
                    st.rerun()
            with c3:
                if st.button("Refresh Row", key=f"refresh_{lead['id']}"):
                    refresh_leads()
                    st.rerun()
else:
    st.info("No leads match your search.")
