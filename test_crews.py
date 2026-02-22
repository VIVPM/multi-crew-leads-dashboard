# test_crews.py
import os
import yaml
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool, ScrapeWebsiteTool

warnings.filterwarnings('ignore')
load_dotenv(dotenv_path='.env')

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
# LLM & Agents
# =========================
sambana_key = os.getenv("SAMBANOVA_API_KEY") or input("Enter your Sambanova API Key: ")

# 4 LLMs
llm1 = LLM(model="sambanova/Llama-4-Maverick-17B-128E-Instruct", api_key=sambana_key)
llm2 = LLM(model="sambanova/Llama-3.3-Swallow-70B-Instruct-v0.4", api_key=sambana_key)
llm3 = LLM(model="sambanova/Meta-Llama-3.1-8B-Instruct", api_key=sambana_key)
llm4 = LLM(model="sambanova/Meta-Llama-3.3-70B-Instruct", api_key=sambana_key)

# --- Lead Scoring Crew (llm3 + llm4) ---
lead_data_agent = Agent(config=lead_agents_config['lead_data_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()], llm=llm1)
cultural_fit_agent = Agent(config=lead_agents_config['cultural_fit_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()], llm=llm2)
scoring_validation_agent = Agent(config=lead_agents_config['scoring_validation_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()], llm=llm3)

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

# --- Email Writing Crew (llm1 + llm2) ---
email_content_specialist = Agent(config=email_agents_config['email_content_specialist'], llm=llm1)
engagement_strategist = Agent(config=email_agents_config['engagement_strategist'], llm=llm2)

email_drafting = Task(config=email_tasks_config['email_drafting'], agent=email_content_specialist)
engagement_optimization = Task(config=email_tasks_config['engagement_optimization'], context=[email_drafting], agent=engagement_strategist)

email_writing_crew = Crew(
    agents=[email_content_specialist, engagement_strategist],
    tasks=[email_drafting, engagement_optimization],
    verbose=True
)

# =========================
# Test
# =========================

# Inputs for Lead Scoring Crew (uses {lead_data} template var)
lead_scoring_inputs = {"lead_data": {
    "name": "Jane Smith",
    "job_title": "VP of Engineering",
    "company": "TechCorp",
    "email": "jane@techcorp.com",
    "use_case": "AI automation",
    "industry": "Technology",
    "location": "San Francisco, USA",
    "source": "Website"
}}

# Inputs for Email Writing Crew (uses {personal_info}, {company_info}, {lead_score} template vars)
email_crew_inputs = {
    "personal_info": "Jane Smith, VP of Engineering at TechCorp, 10 years experience, based in San Francisco",
    "company_info": "TechCorp, Technology industry, 500 employees, Series B, strong market presence",
    "lead_score": "Score: 85, Demographic: 28, Firmographic: 30, Behavioral: 27"
}

print("=== Testing Lead Scoring Crew ===")
lead_scoring_crew.test(n_iterations=2, eval_llm=llm3, inputs=lead_scoring_inputs)

print("\n=== Testing Email Writing Crew ===")
email_writing_crew.test(n_iterations=2, eval_llm=llm4, inputs=email_crew_inputs)
