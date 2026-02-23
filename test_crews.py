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
# Gemini 2.5 - 4 DIFFERENT Models (all with 1M-token context, low cost)
# =========================
PROVIDER = "gemini"
gemini_key = os.getenv("GEMINI_API_KEY") or input("Enter your Gemini API Key: ")

# 4 distinct Gemini 2.5 models — all stable, all with 1M-token context windows

llm1 = LLM(
    model="gemini/gemini-2.5-flash-lite",           # cheapest & fastest - Lead Data
    api_key=gemini_key,
    temperature=0.7
)

llm2 = LLM(
    model="gemini/gemini-2.5-flash",                # balanced price/perf - Cultural Fit
    api_key=gemini_key,
    temperature=0.7
)

llm3 = LLM(
    model="gemini/gemini-2.5-flash",                # 1M ctx, stable, diff family - Validation
    api_key=gemini_key,
    temperature=0.7
)

llm4 = LLM(
    model="gemini/gemini-2.5-pro",                  # most capable, best reasoning - Strategy
    api_key=gemini_key,
    temperature=0.7
)


# =========================
# --- Lead Scoring Crew ---
# =========================
lead_data_agent = Agent(
    config=lead_agents_config['lead_data_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm1 
)

cultural_fit_agent = Agent(
    config=lead_agents_config['cultural_fit_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm2  
)

scoring_validation_agent = Agent(
    config=lead_agents_config['scoring_validation_agent'],
    tools=[SerperDevTool(), ScrapeWebsiteTool()],
    llm=llm3  
)

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

lead_scoring_crew = Crew(
    agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
    tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
    verbose=True
)

# =========================
# --- Email Writing Crew ---
# =========================
email_content_specialist = Agent(
    config=email_agents_config['email_content_specialist'],
    llm=llm2  
)

engagement_strategist = Agent(
    config=email_agents_config['engagement_strategist'],
    llm=llm4 
)

email_drafting = Task(
    config=email_tasks_config['email_drafting'],
    agent=email_content_specialist
)

engagement_optimization = Task(
    config=email_tasks_config['engagement_optimization'],
    context=[email_drafting],
    agent=engagement_strategist
)

email_writing_crew = Crew(
    agents=[email_content_specialist, engagement_strategist],
    tasks=[email_drafting, engagement_optimization],
    verbose=True
)

# =========================
# Test Inputs
# =========================
lead_scoring_inputs = {
    "lead_data": {
        "name": "Jane Smith",
        "job_title": "VP of Engineering",
        "company": "TechCorp",
        "email": "jane@techcorp.com",
        "use_case": "AI automation",
        "industry": "Technology",
        "location": "San Francisco, USA",
        "source": "Website"
    }
}

email_crew_inputs = {                                                                         
        "personal_info": {                                                                    
        "name": "Jane Smith",                                                                 
        "job_title": "VP of Engineering",                                                     
        "role_relevance": 90,                                                                 
        "professional_background": None,                                                      
        "years_experience": None,                                                             
        "linkedin_url": None,                                                                 
        "location": "San Francisco, USA"                                                      
      },                                                                                      
        "company_info": {                                                                     
        "company_name": "TechCorp",                                                           
        "industry": "Technology",                                                             
        "company_size": 10000,                                                                
        "revenue": None,                                                                      
        "market_presence": 60,                                                                
        "company_location": None,                                                             
        "founding_year": None,                                                                
        "website": None                                                                       
      },                                                                                      
        "lead_score": {                                                                       
        "score": 87,                                                                          
        "scoring_criteria": [                                                                 
        "Role Relevance (Weighted 25%): VP of Engineering is highly relevant for AI automation platform. Score: 9/10 (90 points).",                                         
        "Company Size (Weighted 25%): Inferred as 10,000+ employees (Enterprise) based on 'serving Fortune 500 companies worldwide' and CrewAI's ICP. Score: 100 points.",        
        "Market Presence (Weighted 20%): Provided as 6/10 (60 points).",                      
        "Cultural Fit (Weighted 30%): Assessed as 9/10 (90 points) due to strong alignment in innovation, collaboration, and strategic objectives."                                
        ],                                                                                    
        "validation_notes": "Company size for 'TechCorp' was inferred as 10,000+ employees to align with CrewAI's ICP of 'Enterprise companies looking into Agentic automation' and the context that TechCorp 'serves Fortune 500 companies worldwide'. Direct generic searches for 'TechCorp company size' were inconclusive for an enterprise-level company, showing smaller entities. All other scores were derived directly from provided data.",                                                                        
        "demographic_score": 90,                                                              
        "firmographic_score": 80,                                                             
        "behavioral_score": 90                                                                
      }                                                                                       
  }       

# =========================
# Run Tests
# =========================
if __name__ == "__main__":
    print("=" * 60)
    print(f"Testing with Provider: {PROVIDER}")
    print("=" * 60)
    print(f"\nModels configured (4 DIFFERENT Gemini 2.5 models, 1M-token context):")
    print(f"  LLM1 (Lead Data):      {llm1.model}")
    print(f"  LLM2 (Cultural Fit):   {llm2.model}")
    print(f"  LLM3 (Validation):     {llm3.model}")
    print(f"  LLM4 (Engagement):     {llm4.model}")
    print("=" * 60)
    
    print("\n=== Testing Lead Scoring Crew ===")
    lead_scoring_crew.test(n_iterations=2, eval_llm=llm1, inputs=lead_scoring_inputs)

    print("\n=== Testing Email Writing Crew ===")
    email_writing_crew.test(n_iterations=2, eval_llm=llm1, inputs=email_crew_inputs)