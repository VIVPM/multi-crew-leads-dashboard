"""Checks backend.persist_results against a real Supabase. No LLM calls.

Run: backend/.venv/Scripts/python.exe tests/test_persist_results.py

persist_results is pure database code — it reads attributes off finished
pipeline objects and writes two rows. So it can be exercised with synthetic
_StageOutput objects instead of a real lead, which costs nothing and pins the
numbers exactly rather than hoping a live run produces something checkable.

Deliberately NOT in CI: it needs Supabase credentials and writes rows. The two
suites CI runs (test_security, test_pipeline) stay network-free. What this
covers is the seam the no-network tests can't reach — that the CrewOutput-shaped
shim pipeline.py returns still satisfies everything persist_results reads, and
that what lands in `leads` and `analysis_runs` is right.

Creates its own user and leads, and deletes them again in a finally block.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

os.environ.setdefault("LLM_MODEL", "GEMINI")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(dotenv_path=os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "backend", ".env"))

from backend import persist_results, supabase  # noqa: E402
from pipeline import (  # noqa: E402
    ROLE_COMPANY, ROLE_EMAIL, ROLE_PERSONAL, ROLE_SCORING,
    LeadScoringResult, _CombinedScoreOutput, _StageOutput,
)


def _scoring(score: int) -> LeadScoringResult:
    return LeadScoringResult(
        personal_info={"name": "A", "job_title": "CEO", "role_relevance": 9},
        company_info={"company_name": "Acme", "industry": "Tech",
                      "company_size": 500, "market_presence": 8},
        lead_score={"score": score, "scoring_criteria": ["role"],
                    "demographic_score": 28, "firmographic_score": 27,
                    "behavioral_score": score - 55},
    )


user_id = None
try:
    username = f"persist-test-{uuid.uuid4().hex[:8]}@example.com"
    user_id = supabase.table("users").insert(
        {"username": username, "password": "x"}).execute().data[0]["id"]
    rows = supabase.table("leads").insert([
        {"user_id": user_id, "name": "Lead A", "company": "Acme", "email": "a@x.com"},
        {"user_id": user_id, "name": "Lead B", "company": "Acme", "email": "b@x.com"},
    ]).execute().data
    lead_a, lead_b = rows[0], rows[1]

    # Lead A: company research ran, scored 85, email drafted.
    # Lead B: company research served from cache, scored 40, no email.
    # personal research 90+40, scoring 10+10 — deliberately lopsided, so an
    # even split (75/75) and the real figures (130/20) can't be confused.
    score_a = _CombinedScoreOutput(
        _StageOutput([ROLE_PERSONAL, ROLE_SCORING], "scoring raw", _scoring(85), 100, 50,
                     per_role=[(90, 40), (10, 10)]),
        _StageOutput([ROLE_COMPANY], "company raw", None, 30, 10),
    )
    score_b = _CombinedScoreOutput(
        _StageOutput([ROLE_PERSONAL, ROLE_SCORING], "scoring raw", _scoring(40), 100, 50,
                     per_role=[(90, 40), (10, 10)]),
        None,
    )
    email_a = _StageOutput([ROLE_EMAIL], "Subject: hi\n\nBody.", None, 40, 20)

    results = persist_results(
        [lead_a, lead_b], [score_a, score_b], [email_a, None],
        {ROLE_COMPANY: 1.0, ROLE_PERSONAL: 2.0, ROLE_SCORING: 3.0, ROLE_EMAIL: 4.0},
        [False, True], 12.5,
    )

    # --- the summary handed back to the job record ---
    assert [r["score"] for r in results] == [85, 40], results
    assert [r["email_drafted"] for r in results] == [True, False], results

    # --- what landed in `leads` ---
    saved = {r["id"]: r for r in supabase.table("leads")
             .select("id,score,scoring_result,email_draft")
             .eq("user_id", user_id).execute().data}
    a, b = saved[lead_a["id"]], saved[lead_b["id"]]
    assert a["score"] == 85 and b["score"] == 40
    assert a["email_draft"].startswith("Subject:")
    assert not b["email_draft"], "lead B scored below the threshold — no draft"
    assert set(a["scoring_result"]) == {"personal_info", "company_info", "lead_score"}
    assert a["scoring_result"]["lead_score"]["score"] == 85

    # --- what landed in `analysis_runs` ---
    runs = {r["lead_id"]: r for r in supabase.table("analysis_runs")
            .select("*").in_("lead_id", [lead_a["id"], lead_b["id"]]).execute().data}
    ra, rb = runs[lead_a["id"]], runs[lead_b["id"]]

    # A: company 40 + scoring 150 + email 60
    assert ra["total_tokens"] == 250, ra["total_tokens"]
    # B: company was cached, so only the scoring stage's 150
    assert rb["total_tokens"] == 150, rb["total_tokens"]
    assert ra["duration_seconds"] == 12.5 and ra["success_rate"] == 100.0

    # Both leads report all four agents: A because every stage ran, B because
    # the cached and skipped stages are still listed at zero.
    assert ra["agents_executed"] == 4 and rb["agents_executed"] == 4
    by_name_a = {x["agent"]: x for x in ra["agents_data"]}
    assert set(by_name_a) == {ROLE_COMPANY, ROLE_PERSONAL, ROLE_SCORING, ROLE_EMAIL}
    # Every row is the agent's real usage, not a share of a stage total. An even
    # split would have put 75 on both of the last two.
    assert by_name_a[ROLE_COMPANY]["tokens"] == 40
    assert by_name_a[ROLE_EMAIL]["tokens"] == 60
    assert by_name_a[ROLE_PERSONAL]["tokens"] == 130
    assert by_name_a[ROLE_SCORING]["tokens"] == 20
    # and they still add up to the stage total, so nothing is lost or invented
    assert by_name_a[ROLE_PERSONAL]["tokens"] + by_name_a[ROLE_SCORING]["tokens"] == 150

    # A stage that reports no per-agent figures still falls back to the split —
    # which is what a real CrewAI TaskOutput does, having no .tokens at all.
    from pipeline import _StageOutput as _SO
    assert [t.tokens for t in _SO([ROLE_PERSONAL, ROLE_SCORING], "r", None, 100, 50).tasks_output]         == [None, None]
    # agent_times is keyed by the YAML role strings and reaches the rows
    assert by_name_a[ROLE_EMAIL]["time_seconds"] == 4.0

    by_name_b = {x["agent"]: x for x in rb["agents_data"]}
    assert by_name_b[ROLE_COMPANY]["status"] == "Cached"
    assert by_name_b[ROLE_COMPANY]["tokens"] == 0
    assert by_name_b[ROLE_EMAIL]["status"] == "Skipped"
    assert by_name_b[ROLE_EMAIL]["tokens"] == 0

    # Cost is derived from the split, so the parts must add up to the whole.
    assert abs(sum(x["cost"] for x in ra["agents_data"]) - ra["total_cost"]) < 1e-4
    assert ra["total_cost"] > 0

    # Re-running updates the existing row rather than inserting a second one.
    persist_results([lead_a], [score_a], [email_a], {}, [False], 9.0)
    again = supabase.table("analysis_runs").select("id,duration_seconds").eq(
        "lead_id", lead_a["id"]).execute().data
    assert len(again) == 1, f"re-run created {len(again)} analysis rows"
    assert again[0]["duration_seconds"] == 9.0, "re-run should overwrite, not append"

    print("test_persist_results.py: all checks passed")
finally:
    if user_id:
        leads = supabase.table("leads").select("id").eq("user_id", user_id).execute().data or []
        if leads:
            supabase.table("analysis_runs").delete().in_(
                "lead_id", [x["id"] for x in leads]).execute()
        supabase.table("leads").delete().eq("user_id", user_id).execute()
        supabase.table("users").delete().eq("id", user_id).execute()
