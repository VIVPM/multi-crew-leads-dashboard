"""
pipeline.py — Pure LangGraph logic for the Sales Pipeline Lead Coordinator.
No Streamlit/FastAPI imports here. All agents, nodes, the graph, and the single
async entry-point `process_leads` live here.

One graph runs per lead:

    START -> company -> personal_research -> scoring -> email -> END
                                                    -> END      (score <= 70)

  1. Company research (+ cultural fit) runs first, because scoring consumes
     its summary. The cache check lives inside the node, so a hit skips the
     model call entirely — company facts and cultural fit are shared across
     every lead from the same company under the same ICP. The caller
     (worker.py) supplies `cache_get`/`cache_set`; pipeline.py stays
     storage-agnostic (no Supabase import here).
  2. Personal research + scoring/validation always run fresh — nothing there
     is safe to cache, since it's specific to one person and their submitted
     data.
  3. Email drafting (+ optimization, merged into one prompt) runs only for
     qualified leads, off a conditional edge.

The prompts still come from the same four YAML files the CrewAI version used,
rendered here into system/human messages. That is deliberate: the prompts are
what moves eval scores, so they did not change along with the framework.

Outputs are wrapped in `_StageOutput` / `_CombinedScoreOutput`, which present
the same attribute surface CrewAI's CrewOutput did (`.pydantic`, `.raw`,
`.token_usage`, `.tasks_output`, `.to_dict()`, `[key]`). backend.py's
persist_results reads that surface, so it needed no changes.
"""

import asyncio
import hashlib
import logging
import os
import time
import types
import yaml
from typing import Callable, Dict, List, Optional, TypedDict

logger = logging.getLogger("pipeline")

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from tavily import TavilyClient

from logging_setup import lead_context

# Leads scoring above this get an email drafted; at or below, the email node
# never runs.
EMAIL_SCORE_THRESHOLD = 70

# Super-steps one tool-using agent may take. A react loop spends two per tool
# round-trip (model, then tools), so this is ~12 searches before the run gives
# up with GraphRecursionError. CrewAI's equivalent was Agent(max_iter=25).
TOOL_LOOP_LIMIT = 25

# Longest a scraped page we hand back to an agent may be. Whole pages blow the
# context window for no benefit — the agents want a few facts, not the site.
SCRAPE_CHAR_LIMIT = 8000

_UA = "Mozilla/5.0 (compatible; lead-coordinator/1.0)"


def make_tavily_tool(tavily_key: str):
    """Build a Tavily search tool bound to this caller's key (no global state)."""
    # Name must be a bare identifier: Gemini rejects a function declaration
    # whose name has spaces in it with a 400. CrewAI accepted "Tavily Web
    # Search"; the schema underneath never did.
    @tool("tavily_web_search")
    def tavily_search_tool(query: str) -> str:
        """Search the web for information using Tavily."""
        client = TavilyClient(api_key=tavily_key)
        result = client.search(query, max_results=5)
        return str(result)
    return tavily_search_tool


@tool("read_website_content")
def scrape_website_tool(url: str) -> str:
    """Fetch a web page and return its visible text."""
    resp = requests.get(url, timeout=20, headers={"User-Agent": _UA})
    resp.raise_for_status()
    text = BeautifulSoup(resp.text, "html.parser").get_text(" ", strip=True)
    return text[:SCRAPE_CHAR_LIMIT]


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
    """Output of the standalone company-research node — the cacheable unit."""
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


_CONFIGS = _load_configs()

# Role strings, read off the YAML so they can't drift from it. These are the
# names persist_results writes into analysis_runs.agents_data and keys
# agent_times by, and backend.py hardcodes two of them for its Cached/Skipped
# rows — so they have to stay exactly what the YAML says. The .strip() is
# because the YAML uses folded scalars (`role: >`), which leave a trailing
# newline.
ROLE_COMPANY = _CONFIGS["lead_agents"]["company_research_agent"]["role"].strip()
ROLE_PERSONAL = _CONFIGS["lead_agents"]["personal_research_agent"]["role"].strip()
ROLE_SCORING = _CONFIGS["lead_agents"]["scoring_validation_agent"]["role"].strip()
ROLE_EMAIL = _CONFIGS["email_agents"]["email_specialist_agent"]["role"].strip()

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

# How structured output gets produced. Gemini handles it natively, so
# with_structured_output's default path is used there.
#
# Workers AI cannot, and it is not close: measured against gpt-oss-20b, all
# three of LangChain's methods fail, each differently. `json_schema` returns a
# bare -1.0 where the object should be. `function_calling` is ignored outright —
# the model writes its answer into content instead of emitting the forced call,
# the same behaviour that made CrewAI's instructor integration fall over.
# `json_mode` on its own produces a markdown table, because LangChain never
# tells the model what shape to return. Spelling the schema out in the prompt
# and asking for a JSON object works on both schemas — which is exactly what
# instructor was doing for CrewAI, just without the monkeypatching.
STRUCTURED_VIA_PROMPT = LLM_MODEL == "CLOUDFLARE"


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


# =============================================================================
# CrewOutput-shaped result wrappers
#
# backend.py's persist_results was written against CrewAI's CrewOutput. Rather
# than rewrite it, a stage's result presents the same surface: `.raw`,
# `.pydantic`, `.token_usage`, `.tasks_output`, `.to_dict()`, `obj[key]`.
# =============================================================================

class _AgentRef:
    """Stands in for a CrewAI TaskOutput — persist_results reads only `.agent`."""

    def __init__(self, role: str):
        self.agent = role


class _StageOutput:
    """One pipeline stage's result, shaped like a CrewOutput.

    `roles` lists the agents that ran in this stage, in order, because
    persist_results turns each into one analysis row.
    """

    def __init__(self, roles: List[str], raw: str, pydantic=None,
                 prompt_tokens: int = 0, completion_tokens: int = 0):
        self.raw = raw
        self.pydantic = pydantic
        self.tasks_output = [_AgentRef(r) for r in roles]
        self.token_usage = types.SimpleNamespace(
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def to_dict(self) -> dict:
        return self.pydantic.model_dump() if self.pydantic is not None else {}

    def __getitem__(self, key):
        return getattr(self.pydantic, key)


class _CombinedScoreOutput:
    """
    Wraps the scoring stage's output plus (optionally) the company research
    stage's, behind the same attribute/item access that callers (backend.py)
    already use, so persist_results() sees one combined token/task view
    whether or not company research actually ran this time.
    """

    def __init__(self, scoring_output, company_output=None):
        self._scoring_output = scoring_output
        self._company_output = company_output

    # Exposed separately so the analysis breakdown can attribute tokens per stage
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
# Prompt rendering — the same YAML the CrewAI version used, turned into messages
# =============================================================================

def _system_prompt(agent_cfg: dict) -> str:
    return (
        f"You are {agent_cfg['role'].strip()}. {agent_cfg['backstory'].strip()}\n"
        f"Your personal goal is: {agent_cfg['goal'].strip()}"
    )


def _human_prompt(task_cfg: dict, inputs: dict, context: str = "") -> str:
    parts = [
        task_cfg["description"].format(**inputs).strip(),
        "\nThis is the expected criteria for your final answer:\n"
        + task_cfg["expected_output"].strip(),
    ]
    if context:
        parts.append("\nThis is the context you're working with:\n" + context)
    return "\n".join(parts)


# =============================================================================
# Model calls
# =============================================================================

def _text(content) -> str:
    """Flatten an AIMessage's content to a string.

    It is a plain str for an ordinary reply, but a list of blocks when the
    model returns parts (Gemini does this whenever thinking is in play). The
    raw text ends up in prompts and in leads.email_draft, so a stringified
    list would be visible to the user.
    """
    if isinstance(content, str):
        return content
    return "".join(
        part if isinstance(part, str) else part.get("text", "")
        for part in content or []
    )


def _usage(messages) -> tuple:
    """Sum (prompt, completion) tokens over the AIMessages in `messages`."""
    prompt = completion = 0
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None) or {}
        prompt += meta.get("input_tokens") or 0
        completion += meta.get("output_tokens") or 0
    return prompt, completion


def _chat(llm, messages, schema=None):
    """One model call. Returns (text, parsed_or_None, prompt_tokens, completion_tokens)."""
    if schema is None:
        reply = llm.invoke(messages)
        prompt, completion = _usage([reply])
        return _text(reply.content), None, prompt, completion

    if STRUCTURED_VIA_PROMPT:
        parser = PydanticOutputParser(pydantic_object=schema)
        reply = llm.bind(response_format={"type": "json_object"}).invoke(
            [*messages, HumanMessage(parser.get_format_instructions())],
        )
        text = _text(reply.content)
        prompt, completion = _usage([reply])
        return text, parser.parse(text), prompt, completion

    # include_raw so the underlying AIMessage — and its token counts — stay
    # visible; with_structured_output otherwise hands back only the model.
    out = llm.with_structured_output(schema, include_raw=True).invoke(messages)
    if out.get("parsing_error"):
        raise RuntimeError(f"{schema.__name__} parse failed: {out['parsing_error']}")
    prompt, completion = _usage([out["raw"]])
    return _text(out["raw"].content), out["parsed"], prompt, completion


def _research(llm, tools, system: str, human: str):
    """Run a tool-using agent to a final answer.

    Returns (messages, prompt_tokens, completion_tokens). The whole message list
    comes back so a caller that also wants structured output can re-read the
    same transcript instead of paying for the research twice.
    """
    agent = create_react_agent(llm, tools, prompt=system)
    messages = agent.invoke(
        {"messages": [HumanMessage(human)]},
        {"recursion_limit": TOOL_LOOP_LIMIT},
    )["messages"]
    prompt, completion = _usage(messages)
    return messages, prompt, completion


def _build_llms(llm_key: str):
    """The two model tiers the agents use, for whichever provider LLM_MODEL selects.

    Returns (judgment_llm, retrieval_llm). Gemini splits them — flash for the
    calls where the answer's quality matters, flash-lite for the ones that are
    mostly retrieval — because it costs less and measurably held up. Workers AI
    publishes one model for this job, so both tiers point at it there; the split
    is an optimisation, not something the pipeline depends on.
    """
    if LLM_MODEL == "CLOUDFLARE":
        from langchain_openai import ChatOpenAI

        class _WorkersAIChatOpenAI(ChatOpenAI):
            """ChatOpenAI with the message shape Cloudflare's OpenAI shim insists on.

            Workers AI is OpenAI-compatible on the happy path but stricter about
            messages. OpenAI lets an assistant message carry content=None when it
            has tool_calls, and accepts content as a list of blocks; Cloudflare
            rejects the whole request with a 400 ("Type mismatch of
            '/messages/2/content', 'string' not in 'null'"). It only bites on the
            second call of a tool round-trip, so it looks like a random mid-run
            failure rather than a format problem. Coercing None to "" and
            flattening block lists is enough — the tool_calls field still carries
            the real payload.

            The CrewAI version needed this identical fix one layer down, on
            litellm's params. Defined here rather than at module scope to keep
            langchain_openai a Cloudflare-only import.
            """

            def _get_request_payload(self, *args, **kwargs) -> dict:
                payload = super()._get_request_payload(*args, **kwargs)
                for message in payload.get("messages") or []:
                    content = message.get("content")
                    if content is None:
                        message["content"] = ""
                    elif isinstance(content, list):
                        message["content"] = "".join(
                            part.get("text", "")
                            for part in content
                            if isinstance(part, dict)
                        )
                return payload

        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        if not account_id:
            raise RuntimeError(
                "LLM_MODEL=CLOUDFLARE needs CLOUDFLARE_ACCOUNT_ID (the account the "
                "Workers AI endpoint belongs to). Set it in backend/.env."
            )
        # Workers AI speaks the OpenAI wire format, so ChatOpenAI reaches it with
        # nothing but a base_url swap.
        #
        # max_tokens is not optional here. gpt-oss-20b is a reasoning model: it
        # spends completion tokens on its own reasoning before it emits any
        # content, and Workers AI defaults the cap to 256. Left alone, the
        # reasoning eats the whole budget and the reply comes back with
        # finish_reason="length" and an empty content field. Measured: a
        # lead-scoring reply needs ~550 tokens, so 4096 leaves room for the
        # longer email task too. Unlike the CrewAI version this only has to be
        # set once — nothing builds a second client behind our back, so the
        # litellm.completion and instructor monkeypatches are gone with it.
        llm = _WorkersAIChatOpenAI(
            model=CLOUDFLARE_MODEL,
            api_key=llm_key,
            base_url=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
            max_tokens=CLOUDFLARE_MAX_TOKENS,
        )
        return llm, llm

    from langchain_google_genai import ChatGoogleGenerativeAI

    return (
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=llm_key),
        ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=llm_key),
    )


# =============================================================================
# Graph
# =============================================================================

class LeadState(TypedDict, total=False):
    lead: dict
    icp: str
    company_summary: str
    company_out: Optional[_StageOutput]
    cache_hit: bool
    personal_raw: str
    personal_tokens: tuple
    scoring: _StageOutput
    email: Optional[_StageOutput]


def build_graph(
    llm_key: str,
    tavily_key: str,
    cache_get: Optional[Callable[[str], Optional[dict]]] = None,
    cache_set: Optional[Callable[[str, str, dict], None]] = None,
    force_refresh: bool = False,
    on_stage: Optional[Callable[[str, str], None]] = None,
    agent_times: Optional[Dict[str, float]] = None,
):
    """Compile the per-lead graph.

    Everything except the lead itself is constant across a batch, so it is
    captured here and the compiled graph is invoked once per lead. Built fresh
    per call (cheap — object construction only), so there is no shared state
    between concurrent requests.
    """
    llm_flash, llm_flash_lite = _build_llms(llm_key)
    search_tools = [make_tavily_tool(tavily_key), scrape_website_tool]
    times = agent_times if agent_times is not None else {}

    def _done(stage: str, role: str, started: float, state: str = "done") -> None:
        times[role] = round(time.time() - started, 1)
        if on_stage:
            on_stage(stage, state)

    # --- Company research + cultural fit (skipped on a cache hit) ---
    def company_node(state: LeadState) -> dict:
        started = time.time()
        lead = state["lead"]
        key = normalize_company_key(lead.get("company", ""), state["icp"])
        cached = None if force_refresh else (cache_get(key) if cache_get else None)
        if cached:
            _done("company", ROLE_COMPANY, started, state="cached")
            return {
                "company_summary": format_company_summary(cached),
                "company_out": None,
                "cache_hit": True,
            }

        agent_cfg = _CONFIGS["lead_agents"]["company_research_agent"]
        task_cfg = _CONFIGS["lead_tasks"]["company_research"]
        inputs = {"lead_data": lead, "our_company_context": state["icp"]}
        messages, p, c = _research(
            llm_flash, search_tools,
            _system_prompt(agent_cfg), _human_prompt(task_cfg, inputs),
        )
        # Re-read the finished research as CompanyResearchResult. This is the
        # same second call create_react_agent(response_format=...) would make
        # internally — done here instead so its tokens get counted, which they
        # are not when the prebuilt agent makes it.
        raw, parsed, p2, c2 = _chat(llm_flash, messages, CompanyResearchResult)

        dump = parsed.model_dump()
        if cache_set:
            cache_set(key, lead.get("company", ""), dump)
        _done("company", ROLE_COMPANY, started)
        return {
            "company_summary": format_company_summary(dump),
            "company_out": _StageOutput([ROLE_COMPANY], raw, parsed, p + p2, c + c2),
            "cache_hit": False,
        }

    # --- Personal research ---
    def personal_node(state: LeadState) -> dict:
        started = time.time()
        agent_cfg = _CONFIGS["lead_agents"]["personal_research_agent"]
        task_cfg = _CONFIGS["lead_tasks"]["personal_research"]
        messages, p, c = _research(
            llm_flash_lite, search_tools,
            _system_prompt(agent_cfg),
            _human_prompt(task_cfg, {"lead_data": state["lead"]}),
        )
        _done("personal_research", ROLE_PERSONAL, started)
        return {"personal_raw": _text(messages[-1].content), "personal_tokens": (p, c)}

    # --- Scoring + validation (no tools: aggregates context already gathered) ---
    def scoring_node(state: LeadState) -> dict:
        started = time.time()
        agent_cfg = _CONFIGS["lead_agents"]["scoring_validation_agent"]
        task_cfg = _CONFIGS["lead_tasks"]["lead_scoring_and_validation"]
        inputs = {
            "lead_data": state["lead"],
            "company_research_summary": state["company_summary"],
        }
        raw, parsed, p, c = _chat(
            llm_flash,
            [
                SystemMessage(_system_prompt(agent_cfg)),
                HumanMessage(_human_prompt(task_cfg, inputs, context=state["personal_raw"])),
            ],
            LeadScoringResult,
        )
        _done("scoring", ROLE_SCORING, started)
        pp, pc = state["personal_tokens"]
        # Personal research and scoring share one stage, matching how the
        # CrewAI version reported them as a single crew.
        return {"scoring": _StageOutput(
            [ROLE_PERSONAL, ROLE_SCORING], raw, parsed, pp + p, pc + c,
        )}

    # --- Email (merged draft + optimize) ---
    def email_node(state: LeadState) -> dict:
        started = time.time()
        agent_cfg = _CONFIGS["email_agents"]["email_specialist_agent"]
        task_cfg = _CONFIGS["email_tasks"]["email_drafting_and_optimization"]
        dump = state["scoring"].pydantic.model_dump()
        inputs = {
            "our_company_context": state["icp"],
            "personal_info": dump["personal_info"],
            "company_info": dump["company_info"],
            "lead_score": dump["lead_score"],
        }
        raw, _, p, c = _chat(llm_flash, [
            SystemMessage(_system_prompt(agent_cfg)),
            HumanMessage(_human_prompt(task_cfg, inputs)),
        ])
        _done("email", ROLE_EMAIL, started)
        return {"email": _StageOutput([ROLE_EMAIL], raw, None, p, c)}

    def route_email(state: LeadState) -> str:
        return "email" if state["scoring"].pydantic.lead_score.score > EMAIL_SCORE_THRESHOLD else END

    graph = StateGraph(LeadState)
    graph.add_node("company", company_node)
    graph.add_node("personal_research", personal_node)
    graph.add_node("scoring", scoring_node)
    graph.add_node("email", email_node)

    graph.add_edge(START, "company")
    graph.add_edge("company", "personal_research")
    graph.add_edge("personal_research", "scoring")
    graph.add_conditional_edges("scoring", route_email, ["email", END])
    graph.add_edge("email", END)
    return graph.compile()


# =============================================================================
# Batch orchestration (sync — run inside a worker thread, see process_leads)
# =============================================================================

def _process_batch(graph, leads: list, our_company_context: str):
    """Runs entirely synchronously — called via asyncio.to_thread so it
    doesn't block the event loop. Returns (scores, emails, cache_hits)."""
    scores, emails, cache_hits = [], [], []
    for i, lead in enumerate(leads):
        with lead_context(lead.get("id", i)):
            final = graph.invoke({"lead": lead, "icp": our_company_context})
        scores.append(_CombinedScoreOutput(final["scoring"], final.get("company_out")))
        emails.append(final.get("email"))
        cache_hits.append(final["cache_hit"])
    return scores, emails, cache_hits


# =============================================================================
# Public async entry-point (called by worker.py)
# =============================================================================

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

    agent_times: Dict[str, float] = {}
    graph = build_graph(
        llm_key, tavily_key, cache_get, cache_set, force_refresh, on_stage, agent_times,
    )
    timeout_s = PIPELINE_TIMEOUT_S * max(1, len(leads))

    for attempt in range(1, max_retries + 1):
        try:
            agent_times.clear()
            logger.info("Pipeline attempt %d/%d for %d lead(s)", attempt, max_retries, len(leads))
            scores, emails, cache_hits = await asyncio.wait_for(
                asyncio.to_thread(_process_batch, graph, leads, our_company_context),
                timeout=timeout_s,
            )
            logger.info("Pipeline completed successfully on attempt %d", attempt)
            return scores, emails, agent_times, cache_hits
        except Exception as e:
            logger.warning("Pipeline attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                logger.error("All %d retry attempts exhausted", max_retries)
                raise
            wait = 2 ** attempt
            logger.info("Retrying in %ds...", wait)
            await asyncio.sleep(wait)
