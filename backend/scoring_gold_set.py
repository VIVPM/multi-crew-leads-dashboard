"""
Tier-2 scoring accuracy: compare agent scores against a human-scored gold set.

Tier 1 (scoring_eval.py) showed the scorer is *stable*. Stable isn't the same
as *right* — a scorer that always says 92 is perfectly reliable and useless.
This measures agreement with a human, which is the closest reference available
(real conversion outcomes would be better, but there's no CRM data here).

Two steps:

    python backend/scoring_gold_set.py --export
        Writes gold_set_template.csv — lead details with an empty human_score
        column and NO agent score, so filling it in isn't anchored by what the
        agent already decided. Score each 0-100 yourself, then:

    python backend/scoring_gold_set.py --compare gold_set_template.csv
        Joins your scores back to the agent's and reports:
          MAE, mean signed error (bias), % within 10, Spearman rank
          correlation, and the 70-threshold confusion matrix.

    python backend/scoring_gold_set.py --compare gold_set_template.csv --rescore
        Same, but re-runs the pipeline against the ICP saved right now
        instead of reading stored scores — how you measure whether a prompt
        or rubric change actually helped. Stored scores are left alone, so
        this is an experiment rather than a mutation.

Ranking agreement (Spearman) matters more than absolute agreement: a rep acts
on "who do I contact first", so ordering is what has to be right.

Honest limits, worth keeping attached to any number this produces:
  - One rater. It measures agreement with you, not objective truth.
  - Not truly blind if you've already browsed these leads in the app.
  - n is small, so treat the numbers as directional, not precise.
"""

import os
import sys
import csv
import json
import statistics
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
    from supabase import create_client
except ModuleNotFoundError as e:
    venv_python = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")
    sys.exit(f"{e}\n\nRun with the backend virtualenv:\n\n    \"{venv_python}\" \"{__file__}\"\n")

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

EMAIL_THRESHOLD = 70
DEFAULT_N = 20
FIELDS = ["id", "name", "job_title", "company", "industry", "location", "source", "use_case"]


def _spread_sample(rows, n):
    """Evenly spaced picks across the score-sorted list. Random sampling would
    return mostly 90s — the real distribution is heavily top-skewed — and a
    gold set with no low or mid leads can't reveal much."""
    rows = sorted(rows, key=lambda r: r["score"])
    if len(rows) <= n:
        return rows
    step = (len(rows) - 1) / (n - 1)
    return [rows[round(i * step)] for i in range(n)]


def export(n: int):
    rows = supabase.table("leads").select(
        "id,name,job_title,company,industry,location,source,use_case,score"
    ).not_.is_("score", "null").execute().data or []
    if not rows:
        sys.exit("No scored leads to export — process some leads first.")

    picked = _spread_sample(rows, n)
    path = os.path.join(ROOT_DIR, "gold_set_template.csv")
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS + ["human_score"])
        for r in picked:
            w.writerow([r.get(k) or "" for k in FIELDS] + [""])

    agent = [r["score"] for r in picked]
    print(f"Wrote {len(picked)} leads to {path}")
    print(f"(agent scores withheld from the file; their range is {min(agent)}-{max(agent)})")
    print("\nNext: score each lead 0-100 in the human_score column, judging only")
    print("from the lead's details — how good a prospect is this, really? Then:")
    print(f"\n    python backend/scoring_gold_set.py --compare gold_set_template.csv")


def _ranks(xs):
    """Average ranks, ties shared — needed for a correct Spearman on scores
    that repeat a lot (this data has many 85s and 96s)."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def _pearson(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def _rescore_rows(rows):
    """Re-run the pipeline for these leads against whatever ICP is saved now,
    WITHOUT touching the stored scores. Used to measure a prompt/rubric change
    against the same gold set — an experiment, not a mutation."""
    import asyncio
    from pipeline import process_leads
    from worker import cache_get_company, cache_set_company

    icp = supabase.table("users").select("company_context").eq(
        "username", os.getenv("EVAL_USERNAME", "vivek@gmail.com")).execute().data[0]["company_context"]
    gk, tk = os.environ["GEMINI_API_KEY"], os.environ["TAVILY_API_KEY"]

    async def run():
        out = {}
        for i, r in enumerate(rows, 1):
            lead = {k: r.get(k) or "" for k in FIELDS if k != "id"}
            try:
                scores, _e, _t, hits = await process_leads(
                    [lead], gk, tk, our_company_context=icp,
                    cache_get=cache_get_company, cache_set=cache_set_company,
                    force_refresh=False, max_retries=2,
                )
                # int keys: compare() looks these up as int(row["id"]), and the
                # CSV hands back strings
                out[int(r["id"])] = scores[0].pydantic.lead_score.score
                print(f"  [{i}/{len(rows)}] {r['name'][:26]:26} -> {out[int(r['id'])]}"
                      f"  (cache_hit={bool(hits[0])})", flush=True)
            except Exception as exc:
                print(f"  [{i}/{len(rows)}] {r['name'][:26]:26} FAILED: {type(exc).__name__}", flush=True)
        return out

    return asyncio.run(run())


def compare(path: str, rescore: bool = False):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("human_score") or "").strip()]
    if len(rows) < 5:
        sys.exit(f"Only {len(rows)} rows have a human_score — fill in more before comparing.")

    if rescore:
        print(f"Re-scoring {len(rows)} leads against the currently saved ICP "
              f"(stored scores are left untouched)...\n", flush=True)
        live = _rescore_rows(rows)
        print()
    else:
        ids = [int(r["id"]) for r in rows]
        live = {r["id"]: r["score"] for r in
                (supabase.table("leads").select("id,score").in_("id", ids).execute().data or [])}

    pairs, missing = [], []
    for r in rows:
        agent = live.get(int(r["id"]))
        if agent is None:
            missing.append(r["id"])
            continue
        pairs.append((r["name"], agent, float(r["human_score"])))

    human = [p[2] for p in pairs]
    agent = [float(p[1]) for p in pairs]
    errs = [a - h for a, h in zip(agent, human)]

    mae = statistics.mean(abs(e) for e in errs)
    bias = statistics.mean(errs)
    within10 = sum(1 for e in errs if abs(e) <= 10) / len(errs) * 100
    rho = _pearson(_ranks(agent), _ranks(human))

    tp = sum(1 for a, h in zip(agent, human) if a > EMAIL_THRESHOLD and h > EMAIL_THRESHOLD)
    fp = sum(1 for a, h in zip(agent, human) if a > EMAIL_THRESHOLD and h <= EMAIL_THRESHOLD)
    fn = sum(1 for a, h in zip(agent, human) if a <= EMAIL_THRESHOLD and h > EMAIL_THRESHOLD)
    tn = sum(1 for a, h in zip(agent, human) if a <= EMAIL_THRESHOLD and h <= EMAIL_THRESHOLD)

    print("=" * 74)
    print(f"TIER-2 SCORING ACCURACY  (n={len(pairs)})")
    print("=" * 74)
    print(f"  MAE                 {mae:.1f} points")
    print(f"  Mean signed error   {bias:+.1f}  ({'agent scores HIGHER than you' if bias > 0 else 'agent scores LOWER than you'})")
    print(f"  Within 10 points    {within10:.0f}%")
    print(f"  Spearman rank rho   {rho:.2f}   (ordering agreement; >0.7 is decent)")
    print(f"\n  Email decision at >{EMAIL_THRESHOLD}:")
    print(f"    both email      {tp:3}      agent emails, you wouldn't  {fp:3}  <- wasted outreach")
    print(f"    neither         {tn:3}      you would, agent doesn't    {fn:3}  <- missed leads")
    if missing:
        print(f"\n  skipped (no live agent score): {missing}")

    print("\n  Biggest disagreements:")
    for name, a, h in sorted(pairs, key=lambda p: -abs(p[1] - p[2]))[:5]:
        print(f"    {name[:28]:28} agent {a:5.0f}  you {h:5.0f}   diff {a - h:+.0f}")

    print("\n  One rater, small n — directional, not precise.")

    out_dir = os.path.join(ROOT_DIR, "scoring_eval_results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"tier2_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(), "n": len(pairs),
            "mae": round(mae, 2), "mean_signed_error": round(bias, 2),
            "within_10_pct": round(within10, 1), "spearman_rho": round(rho, 3),
            "threshold": EMAIL_THRESHOLD,
            "confusion": {"both_email": tp, "agent_only": fp, "human_only": fn, "neither": tn},
            "pairs": [{"name": n, "agent": a, "human": h} for n, a, h in pairs],
        }, f, indent=2)
    print(f"\n  Report: {out}")


if __name__ == "__main__":
    if "--export" in sys.argv:
        i = sys.argv.index("--export")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else DEFAULT_N
        export(n)
    elif "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        if len(sys.argv) <= i + 1:
            sys.exit("Usage: --compare <filled_csv> [--rescore]")
        compare(sys.argv[i + 1], rescore="--rescore" in sys.argv)
    else:
        sys.exit(__doc__)
