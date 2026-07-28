"""
Tier-1 scoring reliability checks for the lead scorer.

This does NOT measure accuracy — accuracy needs a reference (a human-scored
gold set, or real conversion outcomes), and that's Tier 2. What this measures
is whether the scorer is *sound enough to be worth validating*:

  1. Reliability  — same lead scored N times: does the number hold still?
  2. Threshold    — do those repeats ever straddle the 70 email cutoff?
  3. Discriminant — do obvious-fit and obvious-misfit leads separate?
  4. Sensitivity  — drop the seniority; does the score fall like the rubric says?
  5. Invariance   — change something cosmetic; does the score stay put?

Threshold stability is the one that matters most in practice: a lead whose
score wanders 16->30 is harmless (never gets an email either way), but one
wandering 68->72 gets an email only sometimes, off the same input.

Company research is served from the existing cache (force_refresh=False), so
repeats exercise the personal-research + scoring crews rather than re-paying
for company lookups. Variance measured here is the scorer's, not the web's.

Run with the backend virtualenv:
    backend\\.venv\\Scripts\\python.exe backend\\scoring_eval.py
"""

import os
import sys
import json
import asyncio
import warnings
import statistics
from datetime import datetime

# Windows' console defaults to cp1252, which can't print the emoji CrewAI's
# event bus emits — reconfigure before anything else touches stdout/stderr.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    from pipeline import process_leads
    from worker import cache_get_company, cache_set_company, supabase
except ModuleNotFoundError as e:
    venv_python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    sys.exit(
        f"{e}\n\n"
        f"This script needs the backend virtualenv, not the system Python "
        f"({sys.executable}).\nRun it with:\n\n"
        f'    "{venv_python}" "{__file__}"\n'
    )

REPEATS = 10         # runs per lead for the reliability check
# The scorer carries ~3-5 points of run-to-run noise, so a single run per side
# can't resolve a seniority effect — an earlier n=1 version of this check read
# -25 one run and -5 the next. Repeat both sides and compare means.
SENSITIVITY_REPEATS = 5
MAX_RETRIES = 2      # pipeline-level retries; absorbs a transient LLM blip
EMAIL_THRESHOLD = 70  # scores above this get an email drafted (pipeline.py)

# Pass/fail bars. Deliberately loose — these catch "broken", not "excellent".
MAX_STDEV = 5.0
MIN_SEPARATION = 25
MIN_SENIORITY_DROP = 10
MAX_COSMETIC_DRIFT = 8


def load_icp() -> str:
    """Use the ICP that's actually saved in the app, so the company-research
    cache keys line up (the key is company name + a hash of this text — a
    reworded ICP silently misses the cache and re-runs the research)."""
    username = os.getenv("EVAL_USERNAME", "vivek@gmail.com")
    rows = supabase.table("users").select("username,company_context").execute().data or []
    for r in rows:
        if r.get("username") == username and (r.get("company_context") or "").strip():
            return r["company_context"]
    for r in rows:
        if (r.get("company_context") or "").strip():
            return r["company_context"]
    sys.exit("No company_context set on any user — configure an ICP in the app first.")


# Five leads spanning the range. Companies are ones already in the research
# cache, so repeats cost the scoring crew rather than fresh web lookups.
# The two mid-seniority leads use invented names on purpose: the web can't
# resolve them, so the submitted title drives the score instead of a
# researched identity — that's what puts them near the 70 cutoff, which is
# the band where variance actually changes the outcome.
RELIABILITY_LEADS = {
    "strong_ceo_vercel": {
        "expect": "strong",
        "lead": {
            "name": "Guillermo Rauch", "job_title": "CEO", "company": "Vercel",
            "email": "guillermo.rauch@example.com",
            "use_case": "Agent orchestration for build, preview and deploy automation",
            "industry": "Technology & Software", "location": "San Francisco, USA",
            "source": "Website",
        },
    },
    "strong_cto_hashicorp": {
        "expect": "strong",
        "lead": {
            "name": "Armon Dadgar", "job_title": "CTO", "company": "HashiCorp",
            "email": "armon.dadgar@example.com",
            "use_case": "Multi-agent automation for infrastructure provisioning",
            "industry": "Technology & Software", "location": "San Francisco, USA",
            "source": "Referral",
        },
    },
    "mid_eng_manager_vercel": {
        "expect": "borderline",
        "lead": {
            "name": "Jordan Ellis", "job_title": "Engineering Manager", "company": "Vercel",
            "email": "jordan.ellis@example.com",
            "use_case": "Evaluating agent orchestration for internal tooling",
            "industry": "Technology & Software", "location": "San Francisco, USA",
            "source": "Website",
        },
    },
    "mid_analyst_datadog": {
        "expect": "borderline",
        "lead": {
            "name": "Riley Chen", "job_title": "Business Analyst", "company": "Datadog",
            "email": "riley.chen@example.com",
            "use_case": "Curious about AI agents for internal reporting",
            "industry": "Technology & Software", "location": "New York, USA",
            "source": "Social Media",
        },
    },
    "weak_owner_plumbing": {
        "expect": "weak",
        "lead": {
            "name": "Marcus Webb", "job_title": "Owner",
            "company": "Webb & Sons Plumbing, LLC",
            "email": "marcus.webb@example.com",
            "use_case": "Wants cheaper invoicing software",
            "industry": "Other", "location": "Dayton, USA", "source": "Other",
        },
    },
}

# Seniority test uses an unidentifiable person on purpose. Perturbing the title
# of a well-known lead measures nothing: personal research finds who they really
# are and (correctly) scores that over the submitted claim.
SENIORITY_SENIOR = {
    "name": "Jordan Ellis", "job_title": "Chief Technology Officer", "company": "Vercel",
    "email": "jordan.ellis@example.com",
    "use_case": "Agent orchestration for build, preview and deploy automation",
    "industry": "Technology & Software", "location": "San Francisco, USA", "source": "Website",
}
SENIORITY_JUNIOR = {**SENIORITY_SENIOR, "job_title": "Summer Intern"}

# Cosmetic-only variant of the first reliability lead — must not move the score.
COSMETIC = {**RELIABILITY_LEADS["strong_ceo_vercel"]["lead"],
            "name": "  guillermo rauch ", "company": "vercel ",
            "location": "san francisco, usa"}


async def score_once(lead: dict, icp: str, gemini_key: str, tavily_key: str):
    """One pipeline run. Returns (score, cache_hit) or (None, None) on failure."""
    try:
        scores, _emails, _times, cache_hits = await process_leads(
            [lead], gemini_key, tavily_key,
            our_company_context=icp,
            cache_get=cache_get_company,
            cache_set=cache_set_company,
            force_refresh=False,
            max_retries=MAX_RETRIES,
        )
        return scores[0].pydantic.lead_score.score, bool(cache_hits[0])
    except Exception as exc:
        print(f"    run failed: {type(exc).__name__}: {str(exc)[:110]}")
        return None, None


async def score_many(label: str, lead: dict, n: int, icp: str, gk: str, tk: str):
    out, hits = [], []
    for i in range(n):
        score, hit = await score_once(lead, icp, gk, tk)
        print(f"    {label} {i + 1}/{n}: score={score} cache_hit={hit}", flush=True)
        if score is not None:
            out.append(score)
            hits.append(hit)
    return out, hits


async def main():
    gemini_key = os.environ["GEMINI_API_KEY"]
    tavily_key = os.environ["TAVILY_API_KEY"]
    icp = load_icp()

    # --only <reliability|sensitivity|invariance> re-runs one section without
    # re-paying for the other 50 runs.
    only = None
    if "--only" in sys.argv:
        i = sys.argv.index("--only")
        only = sys.argv[i + 1] if len(sys.argv) > i + 1 else None
    do_rel = only in (None, "reliability")
    do_sens = only in (None, "sensitivity")
    do_inv = only in (None, "invariance")

    total = (REPEATS * len(RELIABILITY_LEADS) if do_rel else 0) \
        + (SENSITIVITY_REPEATS * 2 if do_sens else 0) + (1 if do_inv else 0)
    print("=" * 74)
    print("TIER-1 SCORING RELIABILITY CHECKS" + (f"  [--only {only}]" if only else ""))
    print(f"{total} pipeline runs; company research served from cache where present.")
    print("=" * 74, flush=True)

    per_lead, all_hits = {}, []
    if do_rel:
        for i, (name, spec) in enumerate(RELIABILITY_LEADS.items(), 1):
            print(f"\n[{i}/{len(RELIABILITY_LEADS)}] {name} (expect: {spec['expect']})", flush=True)
            scores, hits = await score_many(name, spec["lead"], REPEATS, icp, gemini_key, tavily_key)
            per_lead[name] = {"expect": spec["expect"], "scores": scores}
            all_hits.extend(hits)

    senior = junior = cosmetic = []
    if do_sens:
        print(f"\n[sensitivity] unidentifiable person, senior vs junior title "
              f"({SENSITIVITY_REPEATS} runs each)", flush=True)
        senior, sh = await score_many("senior", SENIORITY_SENIOR, SENSITIVITY_REPEATS, icp, gemini_key, tavily_key)
        junior, jh = await score_many("junior", SENIORITY_JUNIOR, SENSITIVITY_REPEATS, icp, gemini_key, tavily_key)
        all_hits.extend(sh + jh)

    if do_inv:
        print("\n[invariance] cosmetic-only changes", flush=True)
        cosmetic, _ = await score_many("cosmetic", COSMETIC, 1, icp, gemini_key, tavily_key)

    checks = []

    # --- reliability + threshold stability, per lead ---
    flippers = []
    for name, d in per_lead.items():
        s = d["scores"]
        if len(s) < 2:
            checks.append({"check": f"reliability:{name}", "pass": None,
                           "detail": "not enough successful runs", "scores": s})
            continue
        sd = statistics.pstdev(s)
        above = sum(1 for x in s if x > EMAIL_THRESHOLD)
        d.update({"mean": round(statistics.mean(s), 1), "stdev": round(sd, 2),
                  "min": min(s), "max": max(s), "spread": max(s) - min(s),
                  "above_threshold": above, "below_threshold": len(s) - above})
        if 0 < above < len(s):
            flippers.append(f"{name} ({above}/{len(s)} above {EMAIL_THRESHOLD}, range {min(s)}-{max(s)})")
        checks.append({
            "check": f"reliability:{name}", "pass": sd <= MAX_STDEV, "scores": s,
            "mean": d["mean"], "stdev": d["stdev"], "spread": d["spread"],
            "detail": f"mean {d['mean']} stdev {sd:.2f} range {min(s)}-{max(s)} (bar: stdev <= {MAX_STDEV})",
        })

    checks.append({
        "check": "threshold_stability", "pass": not flippers,
        "detail": ("no lead straddles the email cutoff"
                   if not flippers else "straddles cutoff: " + "; ".join(flippers)),
        "flippers": flippers,
    })

    # --- discriminant: strong vs weak (borderline excluded by design) ---
    strong = [x for d in per_lead.values() if d["expect"] == "strong" for x in d["scores"]]
    weak = [x for d in per_lead.values() if d["expect"] == "weak" for x in d["scores"]]
    if strong and weak:
        gap = min(strong) - max(weak)
        checks.append({
            "check": "discriminant", "pass": gap >= MIN_SEPARATION,
            "detail": f"worst strong {min(strong)} - best weak {max(weak)} = {gap} (bar: >= {MIN_SEPARATION})",
        })

    if len(senior) >= 2 and len(junior) >= 2:
        s_mean, j_mean = statistics.mean(senior), statistics.mean(junior)
        drop = s_mean - j_mean
        checks.append({
            "check": "sensitivity:seniority", "pass": drop >= MIN_SENIORITY_DROP,
            "senior_scores": senior, "junior_scores": junior,
            "senior_mean": round(s_mean, 1), "junior_mean": round(j_mean, 1),
            "detail": (f"CTO {s_mean:.1f} (sd {statistics.pstdev(senior):.1f}) -> "
                       f"Intern {j_mean:.1f} (sd {statistics.pstdev(junior):.1f}) "
                       f"= -{drop:.1f} (bar: >= {MIN_SENIORITY_DROP})"),
        })
    elif senior and junior:
        checks.append({"check": "sensitivity:seniority", "pass": None,
                       "detail": "not enough successful runs per side to judge"})

    base_scores = per_lead.get("strong_ceo_vercel", {}).get("scores", [])
    if base_scores and cosmetic:
        base = statistics.mean(base_scores)
        drift = abs(base - cosmetic[0])
        checks.append({
            "check": "invariance:cosmetic", "pass": drift <= MAX_COSMETIC_DRIFT,
            "detail": f"|{base:.1f} - {cosmetic[0]}| = {drift:.1f} (bar: <= {MAX_COSMETIC_DRIFT})",
        })

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    for c in checks:
        status = "PASS" if c["pass"] else ("SKIP" if c["pass"] is None else "FAIL")
        print(f"  [{status}] {c['check']:32} {c['detail']}")

    graded = [c for c in checks if c["pass"] is not None]
    passed = sum(1 for c in graded if c["pass"])
    print(f"\n  {passed}/{len(graded)} checks passed")
    print(f"  cache hit rate on reliability runs: {sum(all_hits)}/{len(all_hits)}")
    print("\n  Reliability, not accuracy: nothing here proves a score is 'right'.")
    print("  That needs the Tier-2 human-scored gold set.")

    results_dir = os.path.join(ROOT_DIR, "scoring_eval_results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "repeats": REPEATS, "email_threshold": EMAIL_THRESHOLD,
            "cache_hits": f"{sum(all_hits)}/{len(all_hits)}",
            "thresholds": {
                "max_stdev": MAX_STDEV, "min_separation": MIN_SEPARATION,
                "min_seniority_drop": MIN_SENIORITY_DROP,
                "max_cosmetic_drift": MAX_COSMETIC_DRIFT,
            },
            "per_lead": per_lead, "checks": checks,
            "passed": passed, "graded": len(graded),
        }, f, indent=2)
    print(f"\n  Report: {path}")
    return 0 if passed == len(graded) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
