import streamlit as st
import pandas as pd
import yaml
import os
import asyncio
from dotenv import load_dotenv
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from crewai import Agent, Task, Crew, LLM, Flow
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from crewai.flow.flow import listen, start
from pydantic import BaseModel, Field
from typing import Dict, Optional, List

# Load environment variables
load_dotenv()

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

st.title("Sales Pipeline Lead Scoring and Email Generation")
st.sidebar.header("🔑 Enter your API keys")
sambana_key  = st.sidebar.text_input("Sambanova API Key", type="password")

if  not sambana_key:
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
    context=[email_content_specialist],
    llm=llm3
)

# Creating Email Writing Tasks
email_drafting = Task(
    config=email_tasks_config['email_drafting'],
    agent=email_content_specialist
)

engagement_optimization = Task(
    config=email_tasks_config['engagement_optimization'],
    context = [email_drafting],
    agent=engagement_strategist
)

# Creating Email Writing Crew
email_writing_crew = Crew(
    agents=[email_content_specialist],
    tasks=[email_drafting],
    verbose=True
)

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
    flow = SalesPipeline(leads)  # Pass leads when creating the flow
    await flow.kickoff_async()   # No arguments needed here
    return flow.state["scores"], flow.state["emails"]

# Initialize session state
if 'leads' not in st.session_state:
    st.session_state.leads = []
if 'adding_lead' not in st.session_state:
    st.session_state.adding_lead = False
    


# Button to add new lead
if st.button("Add New Lead"):
    st.session_state.adding_lead = True

# Form to input lead data
if st.session_state.adding_lead:
    with st.form(key='lead_form'):
        name = st.text_input("Name")
        job_title = st.text_input("Job Title")
        company = st.text_input("Company")
        email = st.text_input("Email")
        use_case = st.text_input("Use Case")
        submit_button = st.form_submit_button("Save Lead")
        
        if submit_button:
            lead = {
                "lead_data": {
                    "name": name,
                    "job_title": job_title,
                    "company": company,
                    "email": email,
                    "use_case": use_case
                }
            }
            st.success("Lead added successfully!")
            st.session_state.leads.append(lead)
            st.session_state.adding_lead = False
            

# Display added leads
if st.session_state.leads:
    st.write("Added Leads:")
    for i, lead in enumerate(st.session_state.leads):
        st.write(f"Lead {i+1}: {lead['lead_data']['name']} - {lead['lead_data']['company']}")

# Button to process leads
costs = 0
if st.button("Process Leads"):
    if st.session_state.leads:
        with st.spinner("Processing leads..."):
            try:
                scores, emails = asyncio.run(process_leads(st.session_state.leads))
                
                # Display processing summary
                st.write(f"Processed {len(scores)} leads. {len(emails)} leads qualified for email generation (score > 70).")
                
                # Display scored leads
                if scores:
                    st.write("Scored Leads:")
                    data = []
                    for i in range(len(scores)):
                        score = scores[i].pydantic
                        data = {
                            'Name': score.personal_info.name,
                            'Job Title': score.personal_info.job_title,
                            'Role Relevance': score.personal_info.role_relevance,
                            'Professional Background': score.personal_info.professional_background,
                            'Company Name': score.company_info.company_name,
                            'Industry': score.company_info.industry,
                            'Company Size': score.company_info.company_size,
                            'Revenue': score.company_info.revenue,
                            'Market Presence': score.company_info.market_presence,
                            'Lead Score': score.lead_score.score,
                            'Scoring Criteria': ', '.join(score.lead_score.scoring_criteria),
                            'Validation Notes': score.lead_score.validation_notes
                        }
                        df_usage_metrics = pd.DataFrame([scores[i].token_usage.model_dump()])
                        costs += (0.150 * df_usage_metrics['total_tokens'].sum() / 1_000_000)
                        df = pd.DataFrame.from_dict(data, orient='index', columns=['Value'])

                        # Reset the index to turn the original column names into a regular column
                        df = df.reset_index()

                        # Rename the index column to 'Attribute'
                        df = df.rename(columns={'index': 'Attribute'})
                        html_table = df.style.set_properties(**{'text-align': 'left'}) \
                        .format({'Attribute': lambda x: f'<b>{x}</b>'}) \
                        .hide(axis='index') \
                        .to_html()
                        st.markdown(html_table, unsafe_allow_html=True)
                    # df = pd.DataFrame(data)
                    st.dataframe(df_usage_metrics, width=1000, height=400, use_container_width=True)
                    st.write(f"Total costs: ${costs:.4f}")
                    st.write("Generated Emails:")
                    if emails:
                        costs = 0
                        for i, email in enumerate(emails):
                            st.subheader(f"Email {i+1}")
                            st.text(email)
                            df_usage_metrics = pd.DataFrame([emails[i].token_usage.model_dump()])

                            # Calculate total costs
                            costs += 0.150 * df_usage_metrics['total_tokens'].sum() / 1_000_000
                        st.dataframe(df_usage_metrics, width=1000, height=400, use_container_width=True)
                        st.write(f"Total costs: ${costs:.4f}")
                    
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("No leads to process.")

# Button to clear leads
if st.button("Clear Leads"):
    st.session_state.leads = []
    st.success("Leads cleared.")