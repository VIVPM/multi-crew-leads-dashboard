"""Graph wiring checks for backend/pipeline.py. Run: python tests/test_pipeline.py

No network and no API keys: the two functions that reach a model (`_chat` and
`_research`) are swapped for stubs, so what this actually exercises is the part
we wrote — graph edges, the email score threshold, the company-cache skip, token
accounting, and whether the result objects still present the CrewOutput surface
persist_results reads.
"""

import asyncio
import re
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

os.environ.setdefault("LLM_MODEL", "GEMINI")  # required at import time

import pipeline  # noqa: E402
from pipeline import (  # noqa: E402
    ROLE_COMPANY, ROLE_EMAIL, ROLE_PERSONAL, ROLE_SCORING,
    CompanyResearchResult, LeadScoringResult, UnsafeURLError, _assert_public_url,
    is_retryable, make_tavily_tool, process_leads, scrape_website_tool,
)

# --- egress allowlist on the scrape tool (v2: tool sandboxing) ---
# The model picks the URL and a lead's own use_case reaches that prompt, so
# these are the addresses an attacker would aim it at.
for _bad in ("http://169.254.169.254/latest/meta-data/",  # cloud credentials
             "http://127.0.0.1:8000/", "http://localhost/", "http://10.0.0.1/",
             "http://192.168.1.1/", "http://[::1]/", "http://0.0.0.0/",
             "file:///etc/passwd", "gopher://evil/"):
    try:
        _assert_public_url(_bad)
        raise AssertionError(f"scrape tool would have fetched {_bad}")
    except UnsafeURLError:
        pass
_assert_public_url("https://example.com")  # a real public host still works

# --- retry classification (v2: retry the transport, never the reasoning) ---
assert is_retryable(TimeoutError("deadline exceeded"))
assert is_retryable(RuntimeError("503 Service Unavailable"))
assert is_retryable(RuntimeError("429 rate limit exceeded"))
assert not is_retryable(RuntimeError("LeadScoringResult parse failed: ..."))
assert not is_retryable(RuntimeError("400 Invalid function name"))
assert not is_retryable(UnsafeURLError("Refusing scheme 'file'"))
assert is_retryable(RuntimeError("something nobody has seen before")), (
    "unknown failures retry — one wasted attempt beats failing a job that would work")

# Tool names reach Gemini as function declarations, which must be bare
# identifiers — a name with spaces in it comes back as a 400 on the first agent
# step. CrewAI tolerated "Tavily Web Search", so this is easy to reintroduce.
for _t in (make_tavily_tool("k"), scrape_website_tool):
    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.:-]{0,127}", _t.name), _t.name

SCORES = [85, 40]  # lead A qualifies for an email, lead B does not


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


def _fake_research(llm, tools, system, human):
    return [_FakeMessage("research findings")], 100, 20


def _fake_chat(llm, messages, schema=None, via_prompt=None):
    if schema is CompanyResearchResult:
        parsed = CompanyResearchResult(
            company_info={
                "company_name": "Acme", "industry": "Tech",
                "company_size": 500, "market_presence": 8,
            },
            cultural_fit_score=7,
            cultural_fit_notes="aligned",
        )
        return "company raw", parsed, 30, 10
    if schema is LeadScoringResult:
        parsed = LeadScoringResult(
            personal_info={"name": "A", "job_title": "CEO", "role_relevance": 9},
            company_info={
                "company_name": "Acme", "industry": "Tech",
                "company_size": 500, "market_presence": 8,
            },
            lead_score={
                "score": SCORES.pop(0), "scoring_criteria": ["role"],
                "demographic_score": 28, "firmographic_score": 27,
                "behavioral_score": 30,
            },
        )
        return "scoring raw", parsed, 50, 25
    return "Subject: hi\n\nBody text.", None, 40, 15


pipeline._research = _fake_research
pipeline._chat = _fake_chat
pipeline._build_llms = lambda key, provider=None: (None, None)

# A cache that behaves like worker.py's: both leads are at the same company, so
# the second must be served from the first's write.
CACHE = {}
stages = []

leads = [
    {"id": "lead-a", "name": "Ada", "company": "Acme"},
    {"id": "lead-b", "name": "Bob", "company": "Acme"},
]

scores, emails, agent_times, cache_hits = asyncio.run(process_leads(
    leads, "fake-llm-key", "fake-tavily-key",
    our_company_context="We sell widgets.",
    cache_get=CACHE.get,
    cache_set=lambda key, name, data: CACHE.__setitem__(key, data),
    on_stage=lambda stage, state: stages.append((stage, state)),
))

# --- alignment: every returned list is per-lead and in order ---
assert len(scores) == len(emails) == len(cache_hits) == 2
assert cache_hits == [False, True], cache_hits

# --- the email node runs only above the threshold ---
assert emails[0] is not None, "score 85 should have been drafted an email"
assert emails[1] is None, "score 40 is below threshold — no email node"
assert emails[0].raw.startswith("Subject:")

# --- CrewOutput surface persist_results reads ---
a = scores[0]
assert a.pydantic.lead_score.score == 85
assert a["lead_score"].score == 85              # item access
assert set(a.to_dict()) == {"personal_info", "company_info", "lead_score"}
assert a.raw == "scoring raw"

# Company research ran for lead A, so its agent row is present; for lead B the
# cache hit means company_output is None and persist_results adds a Cached row.
assert [t.agent for t in a.tasks_output] == [ROLE_COMPANY, ROLE_PERSONAL, ROLE_SCORING]
assert [t.agent for t in scores[1].tasks_output] == [ROLE_PERSONAL, ROLE_SCORING]
assert scores[1].company_output is None

# --- token accounting: research + structuring, summed across both stages ---
# company  = research(100+20) + structure(30+10) = 160
# scoring  = research(100+20) + structure(50+25) = 195
assert a.company_output.token_usage.total_tokens == 160, a.company_output.token_usage
assert a.scoring_output.token_usage.total_tokens == 195, a.scoring_output.token_usage
assert a.token_usage.total_tokens == 355, a.token_usage
assert a.token_usage.prompt_tokens == 280
assert a.token_usage.completion_tokens == 75
# lead B skipped company research entirely, so it pays only the scoring stage
assert scores[1].token_usage.total_tokens == 195

# --- progress stages the frontend polls, in order, per lead ---
assert stages == [
    ("company", "done"), ("personal_research", "done"), ("scoring", "done"), ("email", "done"),
    ("company", "cached"), ("personal_research", "done"), ("scoring", "done"),
], stages

# --- agent_times is keyed by the YAML role strings persist_results looks up ---
assert set(agent_times) == {ROLE_COMPANY, ROLE_PERSONAL, ROLE_SCORING, ROLE_EMAIL}, agent_times

# --- the roles backend.py hardcodes for its Cached/Skipped rows still match ---
assert ROLE_COMPANY == "Company Research & Cultural Fit Analyst"
assert ROLE_EMAIL == "Email Specialist"

print("All pipeline graph checks passed.")
