# test_crews.py
import os
import yaml
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from crewai import Agent, Task, Crew, LLM
from crewai.tools import tool
from crewai_tools import ScrapeWebsiteTool
from tavily import TavilyClient

@tool("Tavily Web Search")
def tavily_search_tool(query: str) -> str:
    """Search the web for information using Tavily."""
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    result = client.search(query, max_results=5)
    return str(result)

warnings.filterwarnings('ignore')
load_dotenv(dotenv_path='backend/.env')

# =========================
# Pydantic schemas
# =========================
class LeadPersonalInfo(BaseModel):
    name: str
    job_title: str
    role_relevance: float
    professional_background: Optional[str] = None
    years_experience: Optional[float] = None
    linkedin_url: Optional[str] = None
    location: Optional[str] = None

class CompanyInfo(BaseModel):
    company_name: str
    industry: str
    company_size: float
    revenue: Optional[float] = None
    market_presence: float
    company_location: Optional[str] = None
    founding_year: Optional[int] = None
    website: Optional[str] = None

class LeadScore(BaseModel):
    score: float
    scoring_criteria: List[str]
    validation_notes: Optional[str] = None
    demographic_score: float
    firmographic_score: float
    behavioral_score: float

class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo
    company_info: CompanyInfo
    lead_score: LeadScore

# =========================
# YAML configs load
# =========================
files = {
    'lead_agents': 'backend/config/lead_qualification_agents.yaml',
    'lead_tasks': 'backend/config/lead_qualification_tasks.yaml',
    'email_agents': 'backend/config/email_engagement_agents.yaml',
    'email_tasks': 'backend/config/email_engagement_tasks.yaml'
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
    tools=[tavily_search_tool, ScrapeWebsiteTool()],
    llm=llm1 
)

cultural_fit_agent = Agent(
    config=lead_agents_config['cultural_fit_agent'],
    tools=[tavily_search_tool, ScrapeWebsiteTool()],
    llm=llm2  
)

scoring_validation_agent = Agent(
    config=lead_agents_config['scoring_validation_agent'],
    tools=[tavily_search_tool, ScrapeWebsiteTool()],
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
        "name": "Priya Nair",
        "job_title": "CTO",
        "company": "Freshworks",
        "email": "priya.nair@freshworks.com",
        "use_case": "Customer support automation",
        "industry": "SaaS",
        "location": "Chennai, India",
        "source": "Website"
    }
}

email_crew_inputs = {
  "personal_info": {
    "name": "Priya Nair",
    "job_title": "CTO",
    "role_relevance": 80,
    "professional_background": None,
    "years_experience": None,
    "linkedin_url": None,
    "location": "Chennai, India"
  },
  "company_info": {
    "company_name": "Freshworks",
    "industry": "SaaS (Software Development)",
    "company_size": 4400,
    "revenue": 838.8,
    "market_presence": 80,
    "company_location": None,
    "founding_year": None,
    "website": None
  },
  "lead_score": {
    "score": 85,
    "scoring_criteria": [
      "Role Relevance (Weighted 25%): CTO role is highly relevant for AI automation initiatives. Score: 8/10 (80 points).",
      "Company Size (Weighted 25%): Mid-to-large enterprise (~4400 employees), suitable for scalable AI solutions. Score: 85 points.",
      "Market Presence (Weighted 20%): Strong SaaS presence with score 8/10 (80 points).",
      "Cultural Fit (Weighted 30%): Good alignment with innovation and automation-driven strategies. Score: 85 points."
    ],
    "validation_notes": "The CTO attribution for Priya Nair at Freshworks may not match public records; scoring assumes provided data is correct. Freshworks aligns well with the ICP for AI automation platforms.",
    "demographic_score": 20,
    "firmographic_score": 42.5,
    "behavioral_score": 22.5
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
    
    # --- Step 1: Run lead crew to get actual output for piping ---
    # print("\n=== Running Lead Scoring Crew (kickoff) ===")
    # lead_result = lead_scoring_crew.kickoff(inputs=lead_scoring_inputs)
    # lead_output = lead_result.pydantic
    # print(lead_output)

    # Build email crew inputs from lead crew output
    # email_crew_inputs = {
    #     "personal_info": lead_output.personal_info.model_dump(),
    #     "company_info": lead_output.company_info.model_dump(),
    #     "lead_score": lead_output.lead_score.model_dump()
    # }

    # # --- Step 2: Run email crew with piped inputs ---
    print("\n=== Running Email Writing Crew (kickoff) ===")
    email_result = email_writing_crew.kickoff(inputs=email_crew_inputs)
    email_output = email_result.pydantic
    print(email_output)

    # --- Step 3: Now run .test() for evaluation tables ---
    # print("\n=== Testing Lead Scoring Crew (eval table) ===")
    # lead_scoring_crew.test(n_iterations=3, eval_llm=llm4, inputs=lead_scoring_inputs)

    print("\n=== Testing Email Writing Crew (eval table) ===")
    email_writing_crew.test(n_iterations=3, eval_llm=llm4, inputs=email_crew_inputs)