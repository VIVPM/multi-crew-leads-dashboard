"""Offline checks for probability-weighted scoring in the production pipeline."""

import asyncio
import math
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
os.environ.setdefault("LLM_MODEL", "GEMINI")

import pipeline  # noqa: E402


rubrics = pipeline._CONFIGS["lead_tasks"]["weighted_scoring"]["attributes"]
assert "company" not in pipeline._CONFIGS["lead_tasks"]["weighted_scoring"]["attributes"]["engagement_signal"]["evidence_from"]
assert "company" not in pipeline._CONFIGS["lead_tasks"]["weighted_scoring"]["attributes"]["role_relevance"]["evidence_from"]
assert len(rubrics) == 7
assert sum(rubric["max_points"] for rubric in rubrics.values()) == 100
for rubric in rubrics.values():
    ranges = [band["range"] for band in rubric["bands"]]
    assert len(ranges) == 3
    assert ranges[0][0] == 0 and ranges[-1][1] == rubric["max_points"]
    assert all(left[1] + 1 == right[0] for left, right in zip(ranges, ranges[1:]))

candidates = [
    SimpleNamespace(token=str(score), logprob=math.log(prob))
    for score, prob in [(16, .70), (15, .20), (18, .05), (12, .03), (20, .01)]
]
assert pipeline._weighted_points(candidates, 20) == 16
assert pipeline._weighted_points([SimpleNamespace(token="**", logprob=math.log(.8)), *candidates], 20) == 16
assert pipeline._weighted_points(candidates[:2], 20) == 16, "fewer valid scores still weight"
try:
    pipeline._weighted_points([SimpleNamespace(token="**", logprob=0.0)], 20)
    raise AssertionError("no numeric score token must fail")
except ValueError:
    pass

import openai  # noqa: E402

os.environ["HUGGINGFACE_API_KEY"] = "offline-test-key"
os.environ["CLOUDFLARE_API_TOKEN"] = "offline-test-token"
os.environ["CLOUDFLARE_ACCOUNT_ID"] = "offline-test-account"
original_client = openai.OpenAI
seen = {}


class FakeClient:
    def __init__(self, **kwargs):
        seen["base_url"] = kwargs["base_url"]
        self.chat = SimpleNamespace(completions=self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def create(self, **kwargs):
        assert kwargs["logprobs"] is True and kwargs["top_logprobs"] == 20
        seen["model"] = kwargs["model"]
        prefix = ([SimpleNamespace(token=t) for t in ("<|channel|>", "analysis", "<|message|>", "9",
                                                         "<|channel|>", "final", "<|message|>")]
                  if "cloudflare" in seen["base_url"] else [])
        return SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content="16\nThe role owns lead qualification."),
                logprobs=SimpleNamespace(content=prefix + [
                    SimpleNamespace(token="16", top_logprobs=candidates),
                ]),
            )],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )


openai.OpenAI = FakeClient
try:
    for provider, host, model in (("HUGGING_FACE", "huggingface", pipeline.HF_JUDGE_MODEL),
                                  ("CLOUDFLARE", "cloudflare", pipeline.CLOUDFLARE_MODEL)):
        os.environ["JUDGE_PROVIDER"] = provider
        assert pipeline._judge_attribute(
            rubrics["cultural_fit"], "Use the evidence.", "Bank with large engineering team.",
        )[:3] == (16, "The role owns lead qualification.", (20, 10)), provider
        assert host in seen["base_url"] and seen["model"] == model, seen
finally:
    openai.OpenAI = original_client
for bad in ("", "OPENAI"):
    os.environ["JUDGE_PROVIDER"] = bad
    try:
        pipeline.judge_config()
        raise AssertionError(f"JUDGE_PROVIDER={bad!r} must be rejected")
    except RuntimeError:
        pass
os.environ["JUDGE_PROVIDER"] = "HUGGING_FACE"


class FakeMessage:
    content = "research findings"
    usage_metadata = {"input_tokens": 10, "output_tokens": 5}


def fake_chat(llm, messages, schema=None, via_prompt=None):
    if schema is pipeline.CompanyResearchResult:
        return "company", pipeline.CompanyResearchResult(
            company_info={"company_name": "Acme", "industry": "Tech", "market_presence": 8},
            cultural_fit_score=7, cultural_fit_notes="Builds a competing product.",
        ), 2, 3
    if schema is pipeline.ResearchReport:
        return "report", pipeline.ResearchReport(
            personal_info={"name": "Ada", "job_title": "CEO", "role_relevance": 9},
            company_info={"company_name": "Acme", "industry": "Tech", "market_presence": 8},
            validation_notes="Source says Not a fit.",
        ), 2, 3
    return "Subject: test\n\nHi.", None, 2, 3


calls = []


def fake_judge(rubric, instruction, evidence):
    calls.append(rubric["label"])
    return rubric["max_points"], "Matched the evidence.", (2, 3), []


pipeline._build_llms = lambda *_args: (None, None)
pipeline._research = lambda *args: ([FakeMessage()], 2, 3)
pipeline._chat = fake_chat
pipeline._judge_attribute = fake_judge
scores, emails, _, _ = asyncio.run(pipeline.process_leads(
    [{"name": "Ada", "company": "Acme"}], "fake-key", "fake-search-key", "We sell software.",
))
assert len(calls) == 7
assert scores[0].pydantic.lead_score.score == 100
assert scores[0].pydantic.lead_score.demographic_score == 20
assert scores[0].pydantic.lead_score.firmographic_score == 20
assert scores[0].pydantic.lead_score.behavioral_score == 60
assert emails[0] is not None
assert scores[0].scoring_output.token_usage.total_tokens == 45
print("Weighted pipeline checks passed.")


import requests  # noqa: E402

_original_scrape = pipeline._scrape_url


def _blocked(url):
    raise requests.HTTPError("403 Client Error: Forbidden")


pipeline._scrape_url = _blocked
try:
    blocked_text = pipeline.make_scrape_tool({}).invoke({"url": "https://blocked.example"})
    assert blocked_text.startswith("Could not read https://blocked.example"), blocked_text
    unsafe_text = pipeline.make_scrape_tool({}).invoke({"url": "http://127.0.0.1/"})
finally:
    pipeline._scrape_url = _original_scrape
unsafe_text = pipeline.make_scrape_tool({}).invoke({"url": "http://127.0.0.1/"})
assert unsafe_text.startswith("Could not read http://127.0.0.1/") and "UnsafeURLError" in unsafe_text
print("Scrape error handling checks passed.")

calls_429 = []


def _rate_limited_then_ok(url):
    calls_429.append(url)
    if len(calls_429) < 3:
        raise requests.HTTPError("429 Too Many Requests", response=SimpleNamespace(status_code=429))
    return "page text"


_original_sleep = pipeline.time.sleep
pipeline._scrape_url, pipeline.time.sleep = _rate_limited_then_ok, lambda s: None
try:
    assert pipeline.make_scrape_tool({}).invoke({"url": "https://busy.example"}) == "page text"
    assert len(calls_429) == 3, calls_429
    calls_429.clear()
    pipeline._scrape_url = _blocked
    assert pipeline.make_scrape_tool({}).invoke({"url": "https://blocked.example"}).startswith("Could not read")
finally:
    pipeline._scrape_url, pipeline.time.sleep = _original_scrape, _original_sleep
print("Scrape retry checks passed.")
