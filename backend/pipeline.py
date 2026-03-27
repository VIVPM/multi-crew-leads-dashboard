"""
pipeline.py — Pure CrewAI logic for the Sales Pipeline Lead Coordinator.
No Streamlit imports here. All agents, tasks, crews, the Flow, and
the single async entry-point `process_leads` live here.
"""

import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"   # prevent signal handler errors in threads

import asyncio
import logging
import yaml
from typing import Optional, List, Dict

logger = logging.getLogger("pipeline")

from pydantic import BaseModel, Field
from crewai import Agent, Task, Crew, LLM, Flow
from crewai.flow.flow import listen, start
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import tool
from tavily import TavilyClient

@tool("Tavily Web Search")
def tavily_search_tool(query: str) -> str:
    """Search the web for information using Tavily."""
    client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
    result = client.search(query, max_results=5)
    return str(result)


# =============================================================================
# Pydantic schemas
# =============================================================================

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


# =============================================================================
# YAML config loading
# =============================================================================

def _load_configs() -> Dict[str, dict]:
    """Load all four YAML config files relative to this file's directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    files = {
        "lead_agents": "config/lead_qualification_agents.yaml",
        "lead_tasks":  "config/lead_qualification_tasks.yaml",
        "email_agents": "config/email_engagement_agents.yaml",
        "email_tasks":  "config/email_engagement_tasks.yaml",
    }
    configs: Dict[str, dict] = {}
    for key, rel_path in files.items():
        with open(os.path.join(base, rel_path), "r", encoding="utf-8") as fh:
            configs[key] = yaml.safe_load(fh)
    return configs


# Load once at module level (no Streamlit dependency)
_CONFIGS = _load_configs()

# Cache crew instances per API key to avoid rebuilding every request
_crew_cache: Dict[str, tuple] = {}  # keyed by gemini_key


# =============================================================================
# Crew factory — accepts the Sambanova API key from the UI
# =============================================================================

def build_crews(gemini_key: str):
    """
    Build and return (lead_scoring_crew, email_writing_crew) using the
    Sambanova API key supplied by the caller (e.g., from st.sidebar).
    Caches crew instances per API key to avoid rebuilding every request.
    """
    if gemini_key in _crew_cache:
        logger.info("Using cached crews for API key")
        return _crew_cache[gemini_key]
    llm_70b = LLM(model="gemini/gemini-2.5-flash", api_key=gemini_key)
    llm_8b = LLM(model="gemini/gemini-2.5-flash-lite", api_key=gemini_key)
    llm_maverick = LLM(model="gemini/gemini-2.5-flash", api_key=gemini_key)

    search_tools = [tavily_search_tool, ScrapeWebsiteTool()]

    # --- Lead scoring crew ---
    lead_data_agent = Agent(
        config=_CONFIGS["lead_agents"]["lead_data_agent"],
        tools=search_tools,
        llm=llm_70b,
    )
    cultural_fit_agent = Agent(
        config=_CONFIGS["lead_agents"]["cultural_fit_agent"],
        tools=search_tools,
        llm=llm_8b,
    )
    scoring_validation_agent = Agent(
        config=_CONFIGS["lead_agents"]["scoring_validation_agent"],
        tools=search_tools,
        llm=llm_70b,
    )

    lead_data_task = Task(
        config=_CONFIGS["lead_tasks"]["lead_data_collection"],
        agent=lead_data_agent,
    )
    cultural_fit_task = Task(
        config=_CONFIGS["lead_tasks"]["cultural_fit_analysis"],
        agent=cultural_fit_agent,
    )
    scoring_validation_task = Task(
        config=_CONFIGS["lead_tasks"]["lead_scoring_and_validation"],
        agent=scoring_validation_agent,
        context=[lead_data_task, cultural_fit_task],
        output_pydantic=LeadScoringResult,
    )

    lead_scoring_crew = Crew(
        agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
        tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
        verbose=True,
    )

    # --- Email writing crew ---
    email_content_specialist = Agent(
        config=_CONFIGS["email_agents"]["email_content_specialist"],
        llm=llm_maverick,
    )
    engagement_strategist = Agent(
        config=_CONFIGS["email_agents"]["engagement_strategist"],
        llm=llm_maverick,
    )

    email_drafting = Task(
        config=_CONFIGS["email_tasks"]["email_drafting"],
        agent=email_content_specialist,
    )
    engagement_optimization = Task(
        config=_CONFIGS["email_tasks"]["engagement_optimization"],
        context=[email_drafting],
        agent=engagement_strategist,
    )

    email_writing_crew = Crew(
        agents=[email_content_specialist, engagement_strategist],
        tasks=[email_drafting, engagement_optimization],
        verbose=True,
    )

    logger.info("Built new crew instances (caching for future use)")
    _crew_cache[gemini_key] = (lead_scoring_crew, email_writing_crew)
    return lead_scoring_crew, email_writing_crew


# =============================================================================
# Flow
# =============================================================================

class SalesPipeline(Flow):
    def __init__(self, leads, lead_scoring_crew, email_writing_crew):
        super().__init__()
        self.leads = leads
        self._lead_scoring_crew = lead_scoring_crew
        self._email_writing_crew = email_writing_crew

    @start()
    def score_leads(self):
        scores = self._lead_scoring_crew.kickoff_for_each(self.leads)
        self.state["scores"] = scores
        return scores

    @listen(score_leads)
    def store_leads_score(self, scores):
        return scores

    @listen(score_leads)
    def filter_leads(self, scores):
        return [score for score in scores if score["lead_score"].score > 70]

    @listen(filter_leads)
    def write_email(self, leads):
        scored_leads = [lead.to_dict() for lead in leads]
        emails = self._email_writing_crew.kickoff_for_each(scored_leads)
        return emails

    @listen(write_email)
    def send_email(self, emails):
        self.state["emails"] = emails
        return emails


# =============================================================================
# Public async entry-point (called by app.py)
# =============================================================================

async def process_leads(leads: list, gemini_key: str, max_retries: int = 3):
    """
    Score and email-draft all leads in `leads`.

    Args:
        leads: list of lead dicts (rows from Supabase), each wrapped as
               {"lead_data": <lead_dict>}
        gemini_key: Sambanova API key from the Streamlit sidebar.
        max_retries: Number of retry attempts with exponential backoff.

    Returns:
        (scores, emails) — both are lists of CrewAI output objects.
    """
    lead_scoring_crew, email_writing_crew = build_crews(gemini_key)
    flow = SalesPipeline(leads, lead_scoring_crew, email_writing_crew)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Pipeline attempt %d/%d for %d lead(s)", attempt, max_retries, len(leads))
            await flow.kickoff_async()
            logger.info("Pipeline completed successfully on attempt %d", attempt)
            return flow.state["scores"], flow.state["emails"]
        except Exception as e:
            logger.warning("Pipeline attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                logger.error("All %d retry attempts exhausted", max_retries)
                raise
            wait = 2 ** attempt
            logger.info("Retrying in %ds...", wait)
            await asyncio.sleep(wait)
