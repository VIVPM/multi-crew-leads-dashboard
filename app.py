import os
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

import re
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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

def api(method: str, path: str, **kwargs):
    """Call the FastAPI backend and return parsed JSON, or raise on error."""
    try:
        resp = requests.request(method, f"{BACKEND_URL}{path}", timeout=360, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Cannot reach the backend server. Make sure it is running.")
    except requests.exceptions.HTTPError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        raise RuntimeError(detail)


def friendly_error(e: Exception) -> str:
    msg = str(e).lower()
    if "timeout" in msg or "timed out" in msg:
        return "The request timed out. Please try again."
    if "connection" in msg or "network" in msg or "backend" in msg:
        return str(e)
    if "401" in msg or "unauthorized" in msg or "invalid api" in msg:
        return "Invalid API key. Please check your credentials in the sidebar."
    if "429" in msg or "rate limit" in msg:
        return "Rate limit exceeded. Please wait a moment and try again."
    if "500" in msg or "internal server" in msg:
        return "The server encountered an error. Please try again later."
    return str(e)


EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
MAX_FIELD_LENGTH = 255


def validate_lead(name, job_title, company, email, use_case, industry, location):
    errors = []
    if not name or not name.strip():
        errors.append("Name is required.")
    if not company or not company.strip():
        errors.append("Company is required.")
    if not email or not email.strip():
        errors.append("Email is required.")
    elif not EMAIL_REGEX.match(email.strip()):
        errors.append("Invalid email format.")
    for field_name, value in [("Name", name), ("Job Title", job_title),
                               ("Company", company), ("Email", email),
                               ("Use Case", use_case), ("Industry", industry),
                               ("Location", location)]:
        if value and len(value) > MAX_FIELD_LENGTH:
            errors.append(f"{field_name} must be under {MAX_FIELD_LENGTH} characters.")
    return errors


def refresh_leads():
    if not st.session_state.logged_in or not st.session_state.user_id:
        st.session_state.leads = []
        return
    try:
        st.session_state.leads = api("GET", f"/leads/{st.session_state.user_id}")
    except Exception as e:
        st.error(friendly_error(e))
        st.session_state.leads = []


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

    _, auth_col, _ = st.columns([1, 2, 1])

    with auth_col:
        if st.session_state.show_signup:
            st.markdown("<h3 style='text-align:center'>Signup</h3>", unsafe_allow_html=True)
            with st.form("signup_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Signup", use_container_width=True):
                    if username and password:
                        try:
                            api("POST", "/auth/signup", json={"username": username, "password": password})
                            st.success("Signup successful. Please login.")
                            st.session_state.show_signup = False
                            st.session_state.show_login = True
                            st.rerun()
                        except Exception as e:
                            st.error(friendly_error(e))
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
                        try:
                            data = api("POST", "/auth/login", json={"username": username, "password": password})
                            st.session_state.logged_in = True
                            st.session_state.user_id = data["user_id"]
                            st.session_state.show_login = False
                            st.session_state.pop("leads", None)
                            st.success("Login successful.")
                            st.rerun()
                        except Exception as e:
                            st.error(friendly_error(e))
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

gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
st.sidebar.markdown("[Get a Gemini API key →](https://aistudio.google.com/apikey)")

tavily_key = st.sidebar.text_input("Tavily API Key", type="password")
st.sidebar.markdown("[Get a Tavily API key →](https://app.tavily.com)")

st.sidebar.divider()

if st.sidebar.button("🚪 Log Out"):
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.pop("leads", None)
    st.session_state.show_login = True
    st.sidebar.success("Logout successful.")
    st.rerun()

if not gemini_key or not tavily_key:
    missing = [k for k, v in [("Gemini", gemini_key), ("Tavily", tavily_key)] if not v]
    st.sidebar.warning(f"Enter your {' & '.join(missing)} API key(s) above to use the crew.")

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
            validation_errors = validate_lead(name, job_title, company, email, use_case, industry, location)
            if validation_errors:
                for err in validation_errors:
                    st.error(err)
            else:
                try:
                    if st.session_state.editing_lead:
                        api("PUT", f"/leads/{st.session_state.editing_lead}", json={
                            "name": name, "job_title": job_title, "company": company,
                            "email": email, "use_case": use_case, "industry": industry,
                            "location": location, "source": source,
                        })
                        st.success("Lead updated.")
                        st.session_state.editing_lead = None
                    else:
                        api("POST", "/leads", json={
                            "name": name, "job_title": job_title, "company": company,
                            "email": email, "use_case": use_case, "industry": industry,
                            "location": location, "source": source,
                            "user_id": st.session_state.user_id,
                        })
                        st.success("Lead added.")
                    st.session_state.adding_lead = False
                    refresh_leads()
                    st.rerun()
                except Exception as e:
                    st.error(friendly_error(e))


# =============================================================================
# Process leads button
# =============================================================================

if st.button("⚡ Process Leads (Score + Email)"):
    if not gemini_key or not tavily_key:
        st.error("Please enter both your Gemini and Tavily API Keys in the sidebar first.")
    else:
        unprocessed = [l for l in st.session_state.leads if l.get("score") is None]
        if not unprocessed:
            st.info("No new leads to process — all leads already have a score.")
        else:
            with st.spinner(f"Processing {len(unprocessed)} lead(s) with AI crew…"):
                try:
                    result = api("POST", "/leads/process", json={
                        "leads": unprocessed,
                        "gemini_api_key": gemini_key,
                        "tavily_api_key": tavily_key,
                    })
                    refresh_leads()
                    st.success(f"✅ {result['processed']} lead(s) processed and scores saved!")
                    st.rerun()
                except Exception as e:
                    logger.error("Lead processing failed: %s", e, exc_info=True)
                    st.error(friendly_error(e))


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

if filtered_leads:
    export_df = pd.DataFrame([{
        "Name": l.get("name"), "Job Title": l.get("job_title"),
        "Company": l.get("company"), "Email": l.get("email"),
        "Use Case": l.get("use_case"), "Industry": l.get("industry"),
        "Location": l.get("location"), "Source": l.get("source"),
        "Score": l.get("score"),
    } for l in filtered_leads])
    st.download_button(
        "📥 Export CSV", export_df.to_csv(index=False),
        file_name="leads_export.csv", mime="text/csv",
    )

PAGE_SIZES = [10, 25, 50, 100]
page_size = st.selectbox("Leads per page", PAGE_SIZES, index=0)

if "leads_page" not in st.session_state:
    st.session_state.leads_page = 0

total_leads = len(filtered_leads)
total_pages = max(1, (total_leads + page_size - 1) // page_size)

if st.session_state.leads_page >= total_pages:
    st.session_state.leads_page = total_pages - 1

start_idx = st.session_state.leads_page * page_size
end_idx = min(start_idx + page_size, total_leads)
page_leads = filtered_leads[start_idx:end_idx]

st.caption(f"Showing {start_idx + 1}–{end_idx} of {total_leads} lead(s)  |  Page {st.session_state.leads_page + 1} of {total_pages}")

prev_col, next_col, _ = st.columns([1, 1, 6])
with prev_col:
    if st.button("⬅️ Previous", disabled=st.session_state.leads_page == 0):
        st.session_state.leads_page -= 1
        st.rerun()
with next_col:
    if st.button("Next ➡️", disabled=st.session_state.leads_page >= total_pages - 1):
        st.session_state.leads_page += 1
        st.rerun()

if page_leads:
    for lead in page_leads:
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
                    try:
                        api("DELETE", f"/leads/{lead['id']}")
                        refresh_leads()
                        st.success("Lead deleted.")
                        st.rerun()
                    except Exception as e:
                        st.error(friendly_error(e))
            with btn3:
                if st.button("🔄 Refresh", key=f"refresh_{lead['id']}"):
                    refresh_leads()
                    st.rerun()

if not filtered_leads:
    st.info("No leads match your search." if search_q else "No leads yet.")
