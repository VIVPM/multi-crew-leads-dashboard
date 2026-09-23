"""
agent_vs_workflow.py — what does taking the agent out of company research save?

Runs one real, already-scored lead through the pipeline twice:

  current   company research = agent, personal research = agent   (2 agents)
  hybrid    company research = 2 fixed searches + 1 call, personal = agent (1 agent)

Only the company step differs. Everything else — prompts from the YAML, the
scoring and email calls, models, the ICP — is the real pipeline, so a gap in
tokens or score comes from removing that agent and not from rewritten prompts
(the CrewAI-era direct_vs_agent.py rewrote every prompt by hand, so its score
gap measured the prompts as much as the agents).

The company cache is bypassed on both sides; otherwise the current run would
skip company research entirely and there would be nothing to compare.

Costs real money: roughly $0.015 per run pair on Gemini.

Usage:
  backend/.venv/Scripts/python.exe backend/agent_vs_workflow.py              # newest real scored lead
  backend/.venv/Scripts/python.exe backend/agent_vs_workflow.py --lead-id 123
  backend/.venv/Scripts/python.exe backend/agent_vs_workflow.py --runs 3     # average out run-to-run noise
"""

import argparse
import asyncio
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, ".env"))
sys.path.insert(0, HERE)

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402
from supabase import create_client  # noqa: E402

import pipeline  # noqa: E402

# Same rates persist_results and worker.py price with, so these numbers line up
# with the per-lead costs quoted in the README.
PROMPT_RATE, COMPLETION_RATE = 0.15 / 1_000_000, 0.60 / 1_000_000

_real_research = pipeline._research
_company_role = pipeline.ROLE_COMPANY
_company_name = ""

# What a lead looks like the first time it's processed. The stored row also has
# score, scoring_result and email_draft from the earlier run; lead_data is
# formatted straight into the prompts, so passing those would show the model its
# own previous answer.
LEAD_INPUT_FIELDS = ("id", "name", "job_title", "company", "email", "use_case",
                     "industry", "location", "source")


def _workflow_company_research(llm, tools, system, human, cb=None):
    """Stand-in for the company agent: code picks the searches, the model only reads.

    The two queries are the ones the agent actually wrote across the Langfuse
    traces — company facts every time, plus a products search so the summary
    call can judge overlap with the ICP. They go through the pipeline's own
    Tavily tool, so the model sees results formatted exactly as the agent would.
    No model call happens here, so this step reports zero tokens; the
    structuring _chat call in company_node is still counted.
    """
    if _company_role not in system:
        return _real_research(llm, tools, system, human, cb)  # personal research stays an agent

    search = next(t for t in tools if t.name == "tavily_web_search")
    queries = [
        f"{_company_name} company information, revenue, employees, founding year, website",
        f"{_company_name} products and services",
    ]
    with ThreadPoolExecutor(len(queries)) as pool:
        results = list(pool.map(lambda q: search.invoke({"query": q}), queries))
    evidence = "\n\n".join(f"Search: {q}\n{r}" for q, r in zip(queries, results))
    return [SystemMessage(system), HumanMessage(f"{human}\n\nSearch results:\n{evidence}")], 0, 0


def pick_lead(sb, lead_id):
    q = sb.table("leads").select("*").not_.is_("scoring_result", "null")
    if lead_id:
        rows = q.eq("id", lead_id).execute().data
    else:
        # Skip load-test and calibration leads: they're synthetic people, and
        # the whole point is a lead that came through the product for real.
        rows = [r for r in q.order("created_at", desc=True).limit(50).execute().data
                if not str(r.get("name", "")).startswith(("Calibration ", "LOADTEST", "Lead "))
                and r.get("company")]
    if not rows:
        sys.exit("No matching scored lead found.")
    lead = rows[0]
    owner = sb.table("users").select("company_context").eq("id", lead["user_id"]).execute().data
    icp = (owner[0].get("company_context") if owner else "") or ""
    if not icp.strip():
        sys.exit(f"Lead {lead['id']}'s owner has no company profile saved.")
    return lead, icp


def run_once(lead, icp, hybrid):
    global _company_name
    pipeline._research = _workflow_company_research if hybrid else _real_research
    _company_name = lead["company"]
    started = time.time()
    scores, emails, _, _ = asyncio.run(pipeline.process_leads(
        [{k: lead.get(k) for k in LEAD_INPUT_FIELDS}], os.environ["GEMINI_API_KEY"], os.environ["TAVILY_API_KEY"],
        our_company_context=icp, force_refresh=True, max_retries=1,
    ))
    elapsed = time.time() - started
    parts = [scores[0].token_usage] + ([emails[0].token_usage] if emails[0] else [])
    prompt = sum(u.prompt_tokens for u in parts)
    completion = sum(u.completion_tokens for u in parts)
    return {
        "tokens": prompt + completion,
        "cost": prompt * PROMPT_RATE + completion * COMPLETION_RATE,
        "time": elapsed,
        "score": scores[0].pydantic.lead_score.score,
        "company_tokens": scores[0].company_output.token_usage.total_tokens,
    }


def mean(runs, key):
    return statistics.mean(r[key] for r in runs)


def table(hybrid, current):
    def saved(a, b, fmt):
        diff = b - a
        pct = f" ({diff / b * 100:.1f}%)" if b else ""
        return fmt(diff) + pct
    rows = [
        ("Tokens", f"{hybrid['tokens']:,.0f}", f"{current['tokens']:,.0f}",
         saved(hybrid["tokens"], current["tokens"], lambda d: f"{d:,.0f}")),
        ("Cost", f"${hybrid['cost']:.4f}", f"${current['cost']:.4f}",
         saved(hybrid["cost"], current["cost"], lambda d: f"${d:.4f}")),
        ("Time", f"{hybrid['time']:.1f}s", f"{current['time']:.1f}s",
         f"{current['time'] - hybrid['time']:.1f}s"),
        ("Score", f"{hybrid['score']:.0f}", f"{current['score']:.0f}", "—"),
    ]
    head = ("", "Workflow + 1 agent", "2 agents (current)", "Savings")
    widths = [max(len(r[i]) for r in rows + [head]) + 2 for i in range(4)]
    line = lambda left, mid, right: left + mid.join("─" * w for w in widths) + right  # noqa: E731
    cell = lambda r: "│" + "│".join(f" {v:<{w - 1}}" for v, w in zip(r, widths)) + "│"  # noqa: E731
    print(line("┌", "┬", "┐"))
    print(cell(head))
    for r in rows:
        print(line("├", "┼", "┤"))
        print(cell(r))
    print(line("└", "┴", "┘"))


if __name__ == "__main__":
    # The Windows console defaults to cp1252, which has no box-drawing
    # characters — the first run crashed on the table after spending the money.
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lead-id", type=int)
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    lead, icp = pick_lead(sb, args.lead_id)
    print(f"Lead {lead['id']}: {lead['name']}, {lead.get('job_title')} at {lead['company']} "
          f"(stored score {lead.get('score')}) — {args.runs} run(s) per side, cache bypassed\n")

    current_runs, hybrid_runs = [], []
    for i in range(args.runs):
        # Alternate the order so neither side always runs against a warmer
        # Tavily or provider cache.
        for hybrid in ((False, True) if i % 2 == 0 else (True, False)):
            r = run_once(lead, icp, hybrid)
            (hybrid_runs if hybrid else current_runs).append(r)
            label = "workflow+1 agent" if hybrid else "2 agents        "
            print(f"  run {i + 1} {label}  score {r['score']:>3}  tokens {r['tokens']:>6,}  "
                  f"(company step {r['company_tokens']:,})  ${r['cost']:.4f}  {r['time']:.1f}s",
                  flush=True)
    pipeline._research = _real_research

    summary = lambda runs: {k: mean(runs, k) for k in ("tokens", "cost", "time", "score")}  # noqa: E731
    print()
    table(summary(hybrid_runs), summary(current_runs))
    print("\nOne lead, and scores vary ~±3.5 points run to run on the same lead — a score gap")
    print("inside that is noise. Use --runs to average; tokens vary too, since the agent")
    print("chooses how many searches to make.")
