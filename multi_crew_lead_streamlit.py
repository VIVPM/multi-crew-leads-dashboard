import streamlit as st
import pandas as pd
import yaml
import os
import asyncio
from dotenv import load_dotenv
from supabase import create_client
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Task, Crew, LLM, Flow
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.flow.flow import listen, start
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from crewai_tools import FileReadTool
import hashlib
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

inputs = {
        "lead_score":{
            "score":95,
            "scoring_criteria":["Role Relevance","Company Size","Market Presence","Cultural Fit"],
            "validation_notes":"""The cultural values of Amazon align with CrewAI's product and pitch.
                                However, potential differences in company culture and values should be considered to ensure a successful partnership.
                                CrewAI should be prepared to adapt to Amazon's fast-paced and competitive environment, while Amazon should be open to CrewAI's innovative approach to AI orchestration.
                            """
        },
        "company_info":{
            "revenue":513,
            "industry":"E-commerce, Technology, Retail",
            "company_name":"Amazon",
            "company_size":1500000,
            "market_presence":10
        },
        "personal_info":{
            "name":"Jeff Bezos",
            "job_title":"Founder & Executive Chairman",
            "role_relevance":10,
            "professional_background":"Jeff Bezos founded Amazon in 1994 and has been instrumental in its growth and innovation, including the development of Amazon Web Services (AWS)."
        }
}

# Define Pydantic models
class LeadPersonalInfo(BaseModel):
    name: str = Field(..., description="The full name of the lead.")
    job_title: str = Field(..., description="The job title of the lead.")
    role_relevance: int = Field(..., description="A score representing how relevant the lead's role is to the decision-making process (0-10).")
    professional_background: Optional[str] = Field(None, description="A brief description of the lead's professional background.")

class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="The name of the company the lead works for.")
    industry: str = Field(..., description="The industry in which the company operates.")
    company_size: int = Field(..., description="The size of the company in terms of employee count.")
    revenue: Optional[float] = Field(None, description="The annual revenue of the company, if available.")
    market_presence: int = Field(..., description="A score representing the company's market presence (0-10).")

class LeadScore(BaseModel):
    score: int = Field(..., description="The final score assigned to the lead (0-100).")
    scoring_criteria: List[str] = Field(..., description="The criteria used to determine the lead's score.")
    validation_notes: Optional[str] = Field(None, description="Any notes regarding the validation of the lead score.")

class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo = Field(..., description="Personal information about the lead.")
    company_info: CompanyInfo = Field(..., description="Information about the lead's company.")
    lead_score: LeadScore = Field(..., description="The calculated score and related information for the lead.")

# Define file paths for YAML configurations
files = {
    'lead_agents': 'config/lead_qualification_agents.yaml',
    'lead_tasks': 'config/lead_qualification_tasks.yaml',
    'email_agents': 'config/email_engagement_agents.yaml',
    'email_tasks': 'config/email_engagement_tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r', encoding='utf-8') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
lead_agents_config = configs['lead_agents']
lead_tasks_config = configs['lead_tasks']
email_agents_config = configs['email_agents']
email_tasks_config = configs['email_tasks']

# Initialize session state for auth
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'show_signup' not in st.session_state:
    st.session_state.show_signup = False
if 'show_login' not in st.session_state:
    st.session_state.show_login = False

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

if not st.session_state.logged_in:
    st.title("Welcome to the Sales Pipeline Lead Scoring and Email Generation")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Signup"):
            st.session_state.show_signup = True
            st.session_state.show_login = False
    with col2:
        if st.button("Login"):
            st.session_state.show_login = True
            st.session_state.show_signup = False

    if st.session_state.show_signup:
        st.subheader("Signup")
        with st.form("signup_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submit = st.form_submit_button("Signup")
            if submit:
                if username and password:
                    existing_user = supabase.table("users").select("*").eq("username", username).execute()
                    if existing_user.data:
                        st.error("Username already exists.")
                    else:
                        password_hash = hash_password(password)
                        supabase.table("users").insert({
                            "username": username,
                            "password": password_hash
                        }).execute()
                        st.success("Signup successful! Please login.")
                        st.session_state.show_signup = False
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
                        st.success("Login successful!")
                        st.session_state.show_login = False
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.error("Please fill in all fields.")
else:
    # Main app content starts here
    st.title("Sales Pipeline Lead Scoring and Email Generation")
    st.sidebar.header("🔑 Enter your API keys")
    sambana_key = st.sidebar.text_input("Sambanova API Key", type="password")
    
    if st.sidebar.button("🚪 Log Out"):
        # clear everything and restart
        st.sidebar.success("Logout successful!")
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.leads = []

    if not sambana_key:
        st.sidebar.warning("Please enter Sambanova API Key above to continue")
        st.stop()

    # Initialize LLMs
    llm3 = LLM(model="sambanova/Meta-Llama-3.3-70B-Instruct", api_key=sambana_key)

    # Creating Lead Scoring Agents
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

    # Creating Lead Scoring Tasks
    lead_data_task = Task(
        config=lead_tasks_config['lead_data_collection'],
        agent=lead_data_agent
    )

    cultural_fit_task = Task(
        config=lead_tasks_config['cultural_fit_analysis'],
        agent=cultural_fit_agent
    )

    scoring_validation_task = Task(
        config=lead_tasks_config['lead_scoring_and_validation'],
        agent=scoring_validation_agent,
        context=[lead_data_task, cultural_fit_task],
        output_pydantic=LeadScoringResult
    )

    # Creating Lead Scoring Crew
    lead_scoring_crew = Crew(
        agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
        tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
        verbose=True
    )

    # Creating Email Writing Agents
    email_content_specialist = Agent(
        config=email_agents_config['email_content_specialist'],
        llm=llm3
    )

    engagement_strategist = Agent(
        config=email_agents_config['engagement_strategist'],
        llm=llm3
    )

    # Creating Email Writing Tasks
    email_drafting = Task(
        config=email_tasks_config['email_drafting'],
        agent=email_content_specialist
    )

    engagement_optimization = Task(
        config=email_tasks_config['engagement_optimization'],
        context=[email_drafting],
        agent=engagement_strategist
    )

    # Creating Email Writing Crew
    email_writing_crew = Crew(
        agents=[email_content_specialist, engagement_strategist],
        tasks=[email_drafting, engagement_optimization],
        verbose=True
    )

    # Initialize session state
    if 'leads' not in st.session_state:
        resp = supabase.table("leads").select("*").eq("user_id", st.session_state.user_id).order("created_at", desc=False).execute()
        st.session_state.leads = resp.data or []

    # Define the SalesPipeline flow with CrewAI Flow
    class SalesPipeline(Flow):
        def __init__(self, leads):
            super().__init__()
            self.leads = leads

        @start()
        def score_leads(self):
            leads = self.leads
            scores = lead_scoring_crew.kickoff_for_each(leads)
            self.state["scores"] = scores
            return scores

        @listen(score_leads)
        def store_leads_score(self, scores):
            # Here we would store the scores in the database
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
            # Here we would send the emails to the leads
            self.state["emails"] = emails
            return emails

    # Streamlit Application
    async def process_leads(leads):
        flow = SalesPipeline(leads)
        await flow.kickoff_async()
        return flow.state["scores"], flow.state["emails"]
        
    if 'adding_lead' not in st.session_state:
        st.session_state.adding_lead = False
    if 'editing_lead' not in st.session_state:
        st.session_state.editing_lead = None

    # Button to add new lead
    if st.button("Add New Lead"):
        st.session_state.adding_lead = True

    # Form to input lead data
    if st.session_state.adding_lead:
        # pre‑fill defaults from session_state if editing
        default_name      = st.session_state.get("Name", "")
        default_title     = st.session_state.get("Job Title", "")
        default_company   = st.session_state.get("Company", "")
        default_email     = st.session_state.get("Email", "")
        default_use_case  = st.session_state.get("Use Case", "")

        with st.form("lead_form"):
            name      = st.text_input("Name",      value=default_name)
            job_title = st.text_input("Job Title", value=default_title)
            company   = st.text_input("Company",   value=default_company)
            email     = st.text_input("Email",     value=default_email)
            use_case  = st.text_input("Use Case",  value=default_use_case)
            submit = st.form_submit_button("Save Lead")

            if submit:
                if st.session_state.editing_lead:
                    #  ————— UPDATE existing row in Supabase —————
                    supabase.table("leads")\
                        .update({
                            "name":      name,
                            "job_title": job_title,
                            "company":   company,
                            "email":     email,
                            "use_case":  use_case,
                        })\
                        .eq("id", st.session_state.editing_lead)\
                        .execute()

                    # mirror locally
                    for l in st.session_state.leads:
                        if l["id"] == st.session_state.editing_lead:
                            l.update({
                                "name":      name,
                                "job_title": job_title,
                                "company":   company,
                                "email":     email,
                                "use_case":  use_case,
                            })
                            break
                    
                    st.success("Lead Updated Successfully!")
                    st.session_state.editing_lead = None

                else:
                    #  ————— INSERT new row (as before) —————
                    new_row = {
                        "name":      name,
                        "job_title": job_title,
                        "company":   company,
                        "email":     email,
                        "use_case":  use_case,
                        "created_at": "now()",
                        "user_id": st.session_state.user_id
                    }
                    resp = supabase.table("leads").insert(new_row).execute()
                    st.session_state.leads.append(resp.data[0])
                    st.success("Leads added Successfully!")
                st.session_state.adding_lead = False
                   
    # Button to process leads
    if st.button("Process Leads"):
        # pick out those without a score yet
        unprocessed = [
            lead for lead in st.session_state.leads
            if lead.get("score") is None
        ]
        if unprocessed:
            with st.spinner("Processing new leads…"):
                try:
                    # run CrewAI on the raw dicts
                    raw_inputs = [{"lead_data": lead} for lead in unprocessed]
                    scores, emails = asyncio.run(process_leads(raw_inputs))
                    # st.text(emails)
                    for lead, score_obj, email_draft in zip(unprocessed, scores, emails):
        # extract the Pydantic model
                        pyd = score_obj.pydantic

                        updates = {
                            "score":           pyd.lead_score.score,
                            "scoring_result":  pyd.dict(),    # now a plain dict
                            "email_draft":     email_draft.raw
                        }
                        supabase.table("leads")\
                                .update(updates)\
                                .eq("id", lead["id"])\
                                .execute()

                        # keep your local copy in sync
                        lead.update(updates)


                    st.success("Leads processed and updated in Supabase!")
                except Exception as e:
                    st.error(f"Processing error: {e}")
        else:
            st.info("No new leads to process.")


    # Button to clear leads
    # Button to clear the in‐form inputs (not the stored leads list)
    if st.button("Clear Leads"):
        # keep the form open
        st.session_state.adding_lead = False
        st.success("Form closed and Leads cleared.")

    # Display leads dashboard
    if st.session_state.leads:
        st.write("## Leads Dashboard")
        for lead in st.session_state.leads:
            title = f"{lead['name']} – {lead['company']}"
            if lead.get("score") is not None:
                title += f" (Score: {lead['score']})"

            with st.expander(title):
                # Show core lead data
                st.json({
                    "Name":      lead["name"],
                    "Job Title": lead["job_title"],
                    "Company":   lead["company"],
                    "Email":     lead["email"],
                    "Use Case":  lead["use_case"],
                })

                # Optional: scoring result & email draft
                if lead.get("scoring_result"):
                    st.write("*Scoring Result:*")
                    st.json(lead["scoring_result"])
                if lead.get("email_draft"):
                    st.write("*Generated Email Draft:*")
                    st.text(lead["email_draft"])

                # For un‑processed leads, show Delete + Edit
                if lead.get("score") is None:
                    c1, c2 = st.columns(2,gap="small")
                    with c1:
                        if st.button("Delete", key=f"del_{lead['id']}"):
                            # 1) remove from Supabase
                            supabase.table("leads") \
                                    .delete() \
                                    .eq("id", lead["id"]) \
                                    .execute()
                            # 2) remove locally
                            st.session_state.leads = [
                                l for l in st.session_state.leads
                                if l["id"] != lead["id"]
                            ]
                            st.success("Leads deleted Successfully!")

                    with c2:
                        if st.button("Edit", key=f"edit_{lead['id']}"):
                            # stash this id & preload form fields
                            st.session_state.editing_lead = lead["id"]
                            st.session_state["Name"]      = lead["name"]
                            st.session_state["Job Title"] = lead["job_title"]
                            st.session_state["Company"]   = lead["company"]
                            st.session_state["Email"]     = lead["email"]
                            st.session_state["Use Case"]  = lead["use_case"]
                            st.session_state.adding_lead  = True
                            
    if st.button("Train Leads"):
        email_writing_crew.train(n_iterations=1, filename="training.pkl",inputs = inputs)
        # # email_expected_outputs = [lead["email_draft"] for lead in processed_leads]
        email_writing_crew.test(n_iterations=2,eval_llm=llm3,inputs = inputs)
                             
    # if st.button("Export Leads to CSV"):
    #         # Prepare data for CSV
    #         leads_data = []
    #         for lead in st.session_state.leads:
    #             lead_dict = {
    #                 "personal_info": str(lead["scoring_result"]["personal_info"]) if lead.get("scoring_result") else None,
    #                 "company_info": str(lead["scoring_result"]["company_info"]) if lead.get("scoring_result") else None,
    #                 "lead_score": str(lead["scoring_result"]["lead_score"]) if lead.get("scoring_result") else None,
    #                 "email_draft": lead.get("email_draft")
    #             }
    #             leads_data.append(lead_dict)
    #         
    #         df = pd.DataFrame(leads_data)
    #         csv = df.to_csv(index=False).encode('utf-8')
    #         
    #         st.download_button(
    #             label="Download CSV",
    #             data=csv,
    #             file_name="leads.csv",
    #             mime="text/csv"
    #         )