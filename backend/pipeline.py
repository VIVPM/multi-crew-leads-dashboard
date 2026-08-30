"""
pipeline.py — Pure CrewAI logic for the Sales Pipeline Lead Coordinator.
No Streamlit/FastAPI imports here. All agents, tasks, crews, and the single
async entry-point `process_leads` live here.

Per-lead flow for each item in `leads`:
  1. Personal research + scoring/validation always run fresh (2-task crew,
     `personal_scoring`) — nothing here is safe to cache, since it's specific
     to one person and their submitted data.
  2. Company research (+ cultural fit) is a separate, independently
     kickoff-able single-task crew (`company`) so it can be skipped on a
     cache hit — company facts and cultural fit are shared across every
     lead from the same company under the same ICP. The caller (worker.py)
     supplies `cache_get`/`cache_set` callables; pipeline.py stays storage-
     agnostic (no Supabase import here).
  3. Email drafting (+ optimization, merged into one task) runs per
     qualified lead (score > 70) via the `email` crew.
"""

import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"   # prevent signal handler errors in threads

import asyncio
import hashlib
import logging
import time
import types
import yaml
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("pipeline")

import instructor
import litellm
from litellm import exceptions as litellm_exceptions
from pydantic import BaseModel, ValidationError
from crewai import Agent, Task, Crew, LLM
from crewai.utilities.converter import ConverterError
from crewai_tools import ScrapeWebsiteTool
from crewai.tools import tool
from tavily import TavilyClient

from logging_setup import lead_context

def make_tavily_tool(tavily_key: str):
    """Build a Tavily search tool bound to this caller's key (no global state)."""
    @tool("Tavily Web Search")
    def tavily_search_tool(query: str) -> str:
        """Search the web for information using Tavily."""
        client = TavilyClient(api_key=tavily_key)
        result = client.search(query, max_results=5)
        return str(result)
    return tavily_search_tool


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


class CompanyResearchResult(BaseModel):
    """Output of the standalone company-research task — the cacheable unit."""
    company_info: CompanyInfo
    cultural_fit_score: int
    cultural_fit_notes: Optional[str] = None


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

# Per-attempt pipeline timeout, scaled by lead count in process_leads()
PIPELINE_TIMEOUT_S = int(os.getenv("PIPELINE_TIMEOUT_S", "600"))

# Which provider powers the agents: "GEMINI" or "CLOUDFLARE" (Workers AI,
# through its OpenAI-compatible endpoint). Required, with no default — this
# picks which account gets billed, and a fallback would quietly send real
# traffic to a provider nobody chose.
try:
    LLM_MODEL = os.environ["LLM_MODEL"].strip().upper()
except KeyError:
    raise RuntimeError(
        "LLM_MODEL is required. Set it to GEMINI or CLOUDFLARE in backend/.env."
    ) from None
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/openai/gpt-oss-20b")
CLOUDFLARE_MAX_TOKENS = int(os.getenv("CLOUDFLARE_MAX_TOKENS", "4096"))

_SUPPORTED_LLM_MODELS = ("GEMINI", "CLOUDFLARE")
if LLM_MODEL not in _SUPPORTED_LLM_MODELS:
    raise RuntimeError(
        f"LLM_MODEL={LLM_MODEL!r} is not one of {_SUPPORTED_LLM_MODELS}. "
        "A typo here would otherwise fall through to the default provider and "
        "quietly bill the wrong account."
    )


# =============================================================================
# Company research cache key + summary formatting
# =============================================================================

def normalize_company_key(company_name: str, our_company_context: str) -> str:
    """
    Cache key = normalized company name + a hash of the ICP text, since
    cultural-fit assessment is relative to whichever ICP is asking — two
    users with different ICPs must not share a cached fit score.
    """
    icp_hash = hashlib.sha256((our_company_context or "").encode()).hexdigest()[:16]
    normalized_name = (company_name or "").strip().lower()
    return f"{normalized_name}:{icp_hash}"


def format_company_summary(company_dump: dict) -> str:
    """Render a CompanyResearchResult dict (cached or fresh) as prompt text."""
    info = company_dump.get("company_info", {}) or {}
    lines = [
        f"Company Name: {info.get('company_name', 'Unknown')}",
        f"Industry: {info.get('industry', 'Unknown')}",
        f"Company Size: {info.get('company_size', 'Unknown')}",
        f"Revenue: {info.get('revenue', 'Unknown')}",
        f"Market Presence (0-10): {info.get('market_presence', 'Unknown')}",
        f"Company Location: {info.get('company_location', 'Unknown')}",
        f"Founding Year: {info.get('founding_year', 'Unknown')}",
        f"Website: {info.get('website', 'Unknown')}",
        f"Cultural Fit Score (0-10): {company_dump.get('cultural_fit_score', 'Unknown')}",
        f"Cultural Fit Notes: {company_dump.get('cultural_fit_notes', 'None')}",
    ]
    return "\n".join(lines)


class _CombinedScoreOutput:
    """
    Wraps the scoring crew's CrewOutput plus (optionally) the company
    research crew's CrewOutput behind the same attribute/item access that
    callers (backend.py) already use, so persist_results() sees one combined
    token/task view whether or not company research actually ran this time.
    """

    def __init__(self, scoring_output, company_output=None):
        self._scoring_output = scoring_output
        self._company_output = company_output

    # Exposed separately so the analysis breakdown can attribute tokens per crew
    @property
    def scoring_output(self):
        return self._scoring_output

    @property
    def company_output(self):
        return self._company_output

    @property
    def pydantic(self):
        return self._scoring_output.pydantic

    @property
    def raw(self):
        return self._scoring_output.raw

    @property
    def token_usage(self):
        s = self._scoring_output.token_usage
        if self._company_output is None:
            return s
        c = self._company_output.token_usage
        return types.SimpleNamespace(
            total_tokens=(getattr(s, "total_tokens", 0) or 0) + (getattr(c, "total_tokens", 0) or 0),
            prompt_tokens=(getattr(s, "prompt_tokens", 0) or 0) + (getattr(c, "prompt_tokens", 0) or 0),
            completion_tokens=(getattr(s, "completion_tokens", 0) or 0) + (getattr(c, "completion_tokens", 0) or 0),
        )

    @property
    def tasks_output(self):
        company_tasks = list(self._company_output.tasks_output) if self._company_output else []
        return company_tasks + list(self._scoring_output.tasks_output or [])

    def to_dict(self):
        return self._scoring_output.to_dict()

    def __getitem__(self, key):
        return self._scoring_output[key]


# =============================================================================
# Crew factory — accepts the Gemini and Tavily API keys from the UI
# =============================================================================

def _force_instructor_json_mode() -> None:
    """Have `instructor` read structured output from content, not from a tool call.

    instructor defaults to TOOLS mode, which expects the pydantic object to come
    back as a function call. gpt-oss ignores the forced call and writes the JSON
    into content instead, so instructor sees zero tool calls and gives up with
    "Instructor does not support multiple tool calls" — even though the JSON it
    wanted is sitting right there and is valid. JSON mode parses content, which
    is what this model actually produces.
    """
    if getattr(instructor.from_litellm, "_cf_json_mode_patched", False):
        return

    inner = instructor.from_litellm

    def json_mode(*args, **kwargs):
        kwargs.setdefault("mode", instructor.Mode.JSON)
        return inner(*args, **kwargs)

    json_mode._cf_json_mode_patched = True
    instructor.from_litellm = json_mode


def _force_cloudflare_max_tokens() -> None:
    """Make every LiteLLM call carry a max_tokens, including the ones we don't own.

    CrewAI builds structured output through `instructor`, and its to_pydantic()
    calls litellm.completion() without a max_tokens at all — so the request falls
    back to the Workers AI default and the reasoning burns the budget before any
    JSON appears ("The output is incomplete due to a max_tokens length limit").
    There's no argument to thread through: CrewAI constructs the instructor
    client itself. Wrapping the function is the only seam, so keep it to filling
    in a missing default and nothing else. The flag makes it idempotent —
    build_crews() runs once per job.
    """
    if getattr(litellm.completion, "_cf_max_tokens_patched", False):
        return

    inner = litellm.completion

    def with_default_max_tokens(*args, **kwargs):
        if not kwargs.get("max_tokens"):
            kwargs["max_tokens"] = CLOUDFLARE_MAX_TOKENS
        return inner(*args, **kwargs)

    with_default_max_tokens._cf_max_tokens_patched = True
    litellm.completion = with_default_max_tokens


class _CloudflareLLM(LLM):
    """CrewAI's LLM with the message shape Cloudflare's OpenAI shim insists on.

    Workers AI is OpenAI-compatible on the happy path but stricter about
    messages. OpenAI lets an assistant message carry content=None when it has
    tool_calls; Cloudflare rejects the whole request with a 400 ("required
    properties at '/messages/2' are 'role,content'"). That only bites on the
    second call of a tool round-trip, so it looks like a random mid-run failure
    rather than a format problem. Coercing None to "" is enough — the tool_calls
    field still carries the real payload.
    """

    def _prepare_completion_params(self, *args, **kwargs):
        params = super()._prepare_completion_params(*args, **kwargs)
        for message in params.get("messages") or []:
            content = message.get("content")
            if content is None:
                message["content"] = ""
            elif isinstance(content, list):
                # Same reason: content blocks must arrive as a plain string.
                message["content"] = "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict)
                )
        return params


def _build_llms(llm_key: str):
    """The two model tiers the agents use, for whichever provider LLM_MODEL selects.

    Returns (judgment_llm, retrieval_llm). Gemini splits them — flash for the
    calls where the answer's quality matters, flash-lite for the ones that are
    mostly retrieval — because it costs less and measurably held up. Workers AI
    publishes one model for this job, so both tiers point at it there; the split
    is an optimisation, not something the pipeline depends on.
    """
    if LLM_MODEL == "CLOUDFLARE":
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise RuntimeError(
                "LLM_MODEL=CLOUDFLARE needs CLOUDFLARE_ACCOUNT_ID (the account the "
                "Workers AI endpoint belongs to). Set it in backend/.env."
            )
        # Workers AI speaks the OpenAI wire format, so LiteLLM routes it with the
        # openai/ prefix plus an api_base rather than needing a native provider.
        api_base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        # LiteLLM has no entry for this model name, so it reports the model as not
        # supporting function calling and CrewAI silently falls back to ReAct-style
        # text prompting ("Thought:/Action:"). gpt-oss won't answer that way: it
        # emits its whole turn on the reasoning channel and returns content=None,
        # which surfaces as "Invalid response from LLM call - None or empty" on the
        # very first agent step. Registering the model restores native tool calling,
        # where it answers normally. Cheap and local — register_model only writes to
        # LiteLLM's in-memory model table.
        litellm.register_model({
            f"openai/{CLOUDFLARE_MODEL}": {
                "max_tokens": CLOUDFLARE_MAX_TOKENS,
                "max_input_tokens": 128000,
                "max_output_tokens": CLOUDFLARE_MAX_TOKENS,
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_function_calling": True,
            },
        })
        # max_tokens is not optional here. gpt-oss-20b is a reasoning model: it
        # spends completion tokens on its own reasoning before it emits any
        # content, and Workers AI defaults the cap to 256. Left alone, the
        # reasoning eats the whole budget and the reply comes back with
        # finish_reason="length" and an empty content field, which reaches CrewAI
        # as a blank answer rather than an error. Measured: a lead-scoring reply
        # needs ~550 tokens, so 4096 leaves room for the longer email task too.
        # CrewAI converts task output to pydantic through `instructor`, which
        # builds its own OpenAI client instead of reusing the LLM object's
        # credentials — it reads OPENAI_API_KEY/OPENAI_BASE_URL and otherwise
        # fails with "Missing credentials" partway through the scoring crew.
        # Pointing those at Workers AI is safe here because nothing else in this
        # process talks to OpenAI.
        _force_cloudflare_max_tokens()
        _force_instructor_json_mode()

        os.environ["OPENAI_API_KEY"] = llm_key
        os.environ["OPENAI_BASE_URL"] = api_base

        llm = _CloudflareLLM(
            model=f"openai/{CLOUDFLARE_MODEL}",
            api_key=llm_key,
            api_base=api_base,
            max_tokens=CLOUDFLARE_MAX_TOKENS,
        )
        return llm, llm

    return (
        LLM(model="gemini/gemini-2.5-flash", api_key=llm_key),
        LLM(model="gemini/gemini-2.5-flash-lite", api_key=llm_key),
    )


def build_crews(llm_key: str, tavily_key: str) -> Dict[str, Crew]:
    """
    Build and return a dict of crews:
      - "personal_scoring": personal research -> scoring/validation (always runs)
      - "company": company research + cultural fit (skippable per lead on a cache hit)
      - "email": drafting + optimization, merged into one task

    Built fresh per call (cheap — object construction only), so there is no
    shared state between concurrent requests.
    """
    llm_flash, llm_flash_lite = _build_llms(llm_key)

    search_tools = [make_tavily_tool(tavily_key), ScrapeWebsiteTool()]

    # --- Personal research + scoring/validation ---
    personal_research_agent = Agent(
        config=_CONFIGS["lead_agents"]["personal_research_agent"],
        tools=search_tools,
        llm=llm_flash_lite,
    )
    scoring_validation_agent = Agent(
        config=_CONFIGS["lead_agents"]["scoring_validation_agent"],
        llm=llm_flash,  # no tools: aggregates/validates context already gathered
    )

    personal_research_task = Task(
        config=_CONFIGS["lead_tasks"]["personal_research"],
        agent=personal_research_agent,
    )
    scoring_validation_task = Task(
        config=_CONFIGS["lead_tasks"]["lead_scoring_and_validation"],
        agent=scoring_validation_agent,
        context=[personal_research_task],
        output_pydantic=LeadScoringResult,
    )
    personal_scoring_crew = Crew(
        agents=[personal_research_agent, scoring_validation_agent],
        tasks=[personal_research_task, scoring_validation_task],
        verbose=True,
    )

    # --- Company research (independently kickoff-able for cache skipping) ---
    company_research_agent = Agent(
        config=_CONFIGS["lead_agents"]["company_research_agent"],
        tools=search_tools,
        llm=llm_flash,
    )
    company_research_task = Task(
        config=_CONFIGS["lead_tasks"]["company_research"],
        agent=company_research_agent,
        output_pydantic=CompanyResearchResult,
    )
    company_research_crew = Crew(
        agents=[company_research_agent],
        tasks=[company_research_task],
        verbose=True,
    )

    # --- Email (merged draft + optimize) ---
    email_specialist_agent = Agent(
        config=_CONFIGS["email_agents"]["email_specialist_agent"],
        llm=llm_flash,
    )
    email_task = Task(
        config=_CONFIGS["email_tasks"]["email_drafting_and_optimization"],
        agent=email_specialist_agent,
    )
    email_crew = Crew(
        agents=[email_specialist_agent],
        tasks=[email_task],
        verbose=True,
    )

    return {
        "personal_scoring": personal_scoring_crew,
        "company": company_research_crew,
        "email": email_crew,
    }


# =============================================================================
# Per-lead orchestration (sync — run inside a worker thread, see process_leads)
# =============================================================================

def _score_one_lead(
    lead: dict,
    crews: Dict[str, Crew],
    our_company_context: str,
    cache_get: Optional[Callable[[str], Optional[dict]]],
    cache_set: Optional[Callable[[str, str, dict], None]],
    force_refresh: bool,
    on_stage: Optional[Callable[[str, str], None]] = None,
):
    """Returns (_CombinedScoreOutput, cache_hit: bool) for one lead."""
    company_key = normalize_company_key(lead.get("company", ""), our_company_context)
    cached = None if force_refresh else (cache_get(company_key) if cache_get else None)

    company_output = None
    if cached:
        # The company crew never runs on a hit, so report its stage here
        if on_stage:
            on_stage("company", "cached")
        company_summary = format_company_summary(cached)
        cache_hit = True
    else:
        company_output = crews["company"].kickoff(inputs={
            "lead_data": lead,
            "our_company_context": our_company_context,
        })
        company_dump = company_output.pydantic.dict()
        if cache_set:
            cache_set(company_key, lead.get("company", ""), company_dump)
        company_summary = format_company_summary(company_dump)
        cache_hit = False

    scoring_output = crews["personal_scoring"].kickoff(inputs={
        "lead_data": lead,
        "company_research_summary": company_summary,
    })
    return _CombinedScoreOutput(scoring_output, company_output), cache_hit


def _process_batch(
    leads: list,
    crews: Dict[str, Crew],
    our_company_context: str,
    cache_get: Optional[Callable[[str], Optional[dict]]],
    cache_set: Optional[Callable[[str, str, dict], None]],
    force_refresh: bool,
    on_stage: Optional[Callable[[str, str], None]] = None,
):
    """Runs entirely synchronously — called via asyncio.to_thread so it
    doesn't block the event loop. Returns (scores, emails, cache_hits)."""
    scores = []
    cache_hits = []
    for i, lead in enumerate(leads):
        with lead_context(lead.get("id", i)):
            score_output, cache_hit = _score_one_lead(
                lead, crews, our_company_context, cache_get, cache_set, force_refresh,
                on_stage,
            )
        scores.append(score_output)
        cache_hits.append(cache_hit)

    qualified = [(i, s) for i, s in enumerate(scores) if s["lead_score"].score > 70]
    email_inputs = [{**s.to_dict(), "our_company_context": our_company_context} for _, s in qualified]
    emails_out = crews["email"].kickoff_for_each(email_inputs) if email_inputs else []
    emails_by_index = {i: email for (i, _), email in zip(qualified, emails_out)}
    emails = [emails_by_index.get(i) for i in range(len(leads))]

    return scores, emails, cache_hits


# =============================================================================
# Public async entry-point (called by worker.py)
# =============================================================================

# Failures a retry cannot fix. Re-running these spends the same money to reach
# the same error: a schema the model can't satisfy, a key that isn't valid, a
# request the provider rejected outright. Everything else — timeouts, 429s,
# 5xx, dropped connections, and anything unrecognised — is retried, because one
# wasted attempt is cheaper than failing a job that would have worked.
#
# Looked up by name instead of imported: litellm keeps PermissionDeniedError in
# litellm.exceptions but not on the package root, and instructor has moved
# InstructorRetryException between releases. A name that disappears should cost
# us one classification, not crash the worker at import time.
_PERMANENT_ERROR_NAMES = (
    "AuthenticationError",          # bad or revoked key
    "PermissionDeniedError",        # key lacks access to the model
    "BadRequestError",              # malformed request; identical next time
    "NotFoundError",                # model or endpoint doesn't exist
    "ContextWindowExceededError",   # prompt is too long, and won't shrink
    "UnsupportedParamsError",       # provider rejects a parameter we send
    "ContentPolicyViolationError",  # blocked content
    "JSONSchemaValidationError",    # response can't satisfy the schema
)


def _build_permanent_errors() -> tuple:
    found = [
        exc for exc in (
            getattr(litellm_exceptions, name, None) for name in _PERMANENT_ERROR_NAMES
        )
        if isinstance(exc, type) and issubclass(exc, BaseException)
    ]
    # Structured-output failures: the model produced something the pydantic
    # model rejects. instructor has already retried internally by this point.
    found += [ValidationError, ConverterError]
    try:
        from instructor.core.exceptions import InstructorRetryException
    except ImportError:
        pass
    else:
        found.append(InstructorRetryException)
    return tuple(found)


_PERMANENT_ERRORS = _build_permanent_errors()


async def process_leads(
    leads: list,
    llm_key: str,
    tavily_key: str,
    our_company_context: str,
    cache_get: Optional[Callable[[str], Optional[dict]]] = None,
    cache_set: Optional[Callable[[str, str, dict], None]] = None,
    force_refresh: bool = False,
    max_retries: int = 3,
    on_stage: Optional[Callable[[str, str], None]] = None,
):
    """
    Score and email-draft all leads in `leads`.

    Returns:
        (scores, emails, agent_times, cache_hits) — scores is a list of
        combined output objects, one per lead; emails is aligned with
        `leads` and holds None for leads that scored at or below the email
        threshold; agent_times maps agent role -> seconds taken; cache_hits
        is aligned with `leads`, True where company research was served
        from cache instead of re-run.

    `our_company_context` is required — cultural fit and email drafting are
    meaningless without a real ICP, so there is no generic fallback here.
    Enforce this before enqueuing a job, not just here.

    `cache_get`/`cache_set` are optional storage-backed callables supplied
    by the caller (pipeline.py itself has no Supabase dependency); omit them
    to disable company-research caching entirely.
    """
    if not our_company_context or not our_company_context.strip():
        raise ValueError("our_company_context is required — set a company profile before processing leads.")
    crews = build_crews(llm_key, tavily_key)
    context_text = our_company_context

    task_timing: List[Dict] = []
    start_ref: List[float] = [0.0]

    # Maps each agent to its pipeline stage, read off the crews themselves so it
    # can't drift from the YAML. Roles are stripped — folded scalars (`role: >`)
    # leave a trailing newline.
    stage_by_role = {
        crews["company"].agents[0].role.strip(): "company",
        crews["personal_scoring"].agents[0].role.strip(): "personal_research",
        crews["personal_scoring"].agents[1].role.strip(): "scoring",
        crews["email"].agents[0].role.strip(): "email",
    }

    def _timing_cb(output):
        agent_name = (
            output.agent if isinstance(output.agent, str)
            else getattr(output.agent, "role", str(output.agent))
        )
        task_timing.append({"agent": agent_name, "ts": time.time()})
        # A finished task means its stage is done
        if on_stage:
            stage = stage_by_role.get(agent_name.strip())
            if stage:
                on_stage(stage, "done")
            else:
                logger.warning("No progress stage mapped for agent %r", agent_name)

    for crew in crews.values():
        crew.task_callback = _timing_cb

    timeout_s = PIPELINE_TIMEOUT_S * max(1, len(leads))

    def _run_sync():
        return _process_batch(leads, crews, context_text, cache_get, cache_set, force_refresh, on_stage)

    for attempt in range(1, max_retries + 1):
        try:
            task_timing.clear()
            start_ref[0] = time.time()
            logger.info("Pipeline attempt %d/%d for %d lead(s)", attempt, max_retries, len(leads))
            scores, emails, cache_hits = await asyncio.wait_for(
                asyncio.to_thread(_run_sync), timeout=timeout_s,
            )
            logger.info("Pipeline completed successfully on attempt %d", attempt)

            agent_times: Dict[str, float] = {}
            prev = start_ref[0]
            for entry in task_timing:
                agent_times[entry["agent"]] = round(entry["ts"] - prev, 1)
                prev = entry["ts"]

            return scores, emails, agent_times, cache_hits
        except Exception as e:
            permanent = isinstance(e, _PERMANENT_ERRORS)
            logger.warning(
                "Pipeline attempt %d failed (%s): %s",
                attempt, "permanent" if permanent else "retryable", e,
            )
            if permanent:
                logger.error(
                    "Not retrying: %s can't succeed on a retry, and each attempt "
                    "re-runs every LLM call", type(e).__name__,
                )
                raise
            if attempt == max_retries:
                logger.error("All %d retry attempts exhausted", max_retries)
                raise
            wait = 2 ** attempt
            logger.info("Retrying in %ds...", wait)
            await asyncio.sleep(wait)
