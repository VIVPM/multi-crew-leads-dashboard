"""
Comprehensive Evaluation Runner for eval_leads.json

Scores all 50 leads from eval_leads.json in batches of 10, then computes
all Tier 1 and Tier 2 metrics:

  Tier 1 (Reliability):
    - Per-lead stdev across N repeats (reliability group only)
    - Threshold straddling (does any lead flip above/below 70?)
    - Discriminant validity (strong vs weak separation)
    - Sensitivity: CTO vs Intern score drop
    - Invariance: cosmetic edits must not move the score

  Tier 2 (Accuracy):
    - Classification accuracy at threshold 70
    - Precision / Recall / F1 for the email decision
    - Confusion matrix (TP / FP / FN / TN)
    - MAE (where numeric expectations can be derived)
    - Adversarial pass/fail checks

  Runs:
    backend\\.venv\\Scripts\\python.exe backend\\run_full_eval.py
    
  Batches of 10 are used for all single-pass groups. Reliability leads
  are scored 3 times each (not 10) to save cost while still measuring variance.
"""

import os
import sys
import json
import asyncio
import statistics
from datetime import datetime

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import warnings
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
    sys.exit(f"{e}\n\nRun with:\n    \"{venv_python}\" \"{__file__}\"\n")

EMAIL_THRESHOLD = 70
RELIABILITY_REPEATS = 3   # 3 repeats per reliability lead (saves cost vs 10)
MAX_COSMETIC_DRIFT = 8
MIN_SENIORITY_DROP = 10
MAX_STDEV = 5.0
MIN_SEPARATION = 25
BATCH_SIZE = 10


def load_icp() -> str:
    username = os.getenv("EVAL_USERNAME", "vivek@gmail.com")
    rows = supabase.table("users").select("username,company_context").execute().data or []
    for r in rows:
        if r.get("username") == username and (r.get("company_context") or "").strip():
            return r["company_context"]
    for r in rows:
        if (r.get("company_context") or "").strip():
            return r["company_context"]
    sys.exit("No company_context found. Configure an ICP in the app first.")


async def score_one(lead_data: dict, icp: str, gk: str, tk: str):
    """Score a single lead. Returns (score, notes, cache_hit) or (None, None, None)."""
    try:
        scores, _emails, _times, hits = await process_leads(
            [lead_data], gk, tk, our_company_context=icp,
            cache_get=cache_get_company, cache_set=cache_set_company,
            force_refresh=False, max_retries=2,
        )
        pyd = scores[0].pydantic
        return (
            pyd.lead_score.score,
            pyd.lead_score.validation_notes or "",
            bool(hits[0]),
        )
    except Exception as exc:
        print(f"    FAILED: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
        return None, None, None


async def score_batch(leads_with_meta, icp, gk, tk, label=""):
    """Score a list of (meta, lead_data) tuples. Returns list of result dicts."""
    results = []
    for i, item in enumerate(leads_with_meta):
        meta, lead_data = item
        name = lead_data.get("name", "?")
        company = lead_data.get("company", "?")
        print(f"  [{label} {i+1}/{len(leads_with_meta)}] {name} @ {company}", flush=True)

        score, notes, hit = await score_one(lead_data, icp, gk, tk)
        results.append({
            "id": meta.get("id"),
            "group": meta.get("group"),
            "expect": meta.get("expect"),
            "name": name,
            "company": company,
            "score": score,
            "notes": notes,
            "cache_hit": hit,
        })
        if score is not None:
            print(f"      -> Score: {score}", flush=True)
    return results


def compute_metrics(all_results):
    """Compute all Tier 1 + Tier 2 metrics from the scored results."""
    metrics = {}

    # ── Split results by group ──
    accuracy = [r for r in all_results if r["group"] in ("accuracy", "adversarial") and r["score"] is not None]
    reliability = {}  # id -> list of scores
    for r in all_results:
        if r["group"] == "reliability" and r["score"] is not None:
            reliability.setdefault(r["id"], {"expect": r["expect"], "scores": []})
            reliability[r["id"]]["scores"].append(r["score"])

    sensitivity = {r["id"]: r for r in all_results if r["group"] == "sensitivity" and r["score"] is not None}
    invariance = {r["id"]: r for r in all_results if r["group"] == "invariance" and r["score"] is not None}
    adversarial = [r for r in all_results if r["group"] == "adversarial"]

    # ═══════════════════════════════════════════
    #  TIER 2: Accuracy metrics
    # ═══════════════════════════════════════════
    
    # Map expected category to a score expectation for threshold-based metrics
    # disqualified -> below threshold, weak -> below threshold
    # strong -> above threshold, borderline -> could go either way
    tp = fp = fn = tn = 0
    within_10_count = 0
    abs_errors = []
    
    # Map categories to expected numeric ranges and midpoints for MAE
    expect_to_midpoint = {
        "disqualified": 20,
        "weak": 35,
        "borderline": 60,
        "strong": 85,
    }
    expect_to_range = {
        "disqualified": (0, 29),
        "weak": (30, 49),
        "borderline": (50, 70),
        "strong": (71, 100),
    }
    
    for r in accuracy:
        expected_cat = r["expect"]
        actual = r["score"]
        
        # Confusion matrix at threshold 70
        expected_above = expected_cat == "strong"
        expected_below = expected_cat in ("disqualified", "weak")
        actual_above = actual > EMAIL_THRESHOLD
        
        if expected_cat == "borderline":
            # Borderline leads can go either way — count them but don't penalise
            pass
        elif expected_above and actual_above:
            tp += 1
        elif expected_above and not actual_above:
            fn += 1
        elif expected_below and actual_above:
            fp += 1
        elif expected_below and not actual_above:
            tn += 1
            
        # Regional Error (MAE)
        if expected_cat in expect_to_midpoint:
            expected_mid = expect_to_midpoint[expected_cat]
            low, high = expect_to_range[expected_cat]
            
            # If the AI scored within the correct expected region, the error is 0
            if low <= actual <= high:
                error = 0
            else:
                # If they missed the region, calculate difference to the midpoint
                error = abs(actual - expected_mid)
                
            abs_errors.append(error)
            if error <= 10:
                within_10_count += 1

    total_classified = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    classification_accuracy = (tp + tn) / total_classified if total_classified > 0 else 0
    
    mae = statistics.mean(abs_errors) if abs_errors else None
    within_10_pct = within_10_count / len(abs_errors) * 100 if abs_errors else None
    
    # Spearman rank correlation (accuracy group)
    # Map categories to ordinal ranks for Spearman
    cat_rank = {"disqualified": 1, "weak": 2, "borderline": 3, "strong": 4}
    ranked_pairs = [(cat_rank[r["expect"]], r["score"]) for r in accuracy if r["expect"] in cat_rank]
    spearman = None
    if len(ranked_pairs) >= 3:
        # Manual Spearman: Pearson correlation on ranks
        try:
            expected_ranks = [p[0] for p in ranked_pairs]
            actual_scores = [p[1] for p in ranked_pairs]
            
            # Rank the actual scores
            n = len(actual_scores)
            sorted_indices = sorted(range(n), key=lambda i: actual_scores[i])
            actual_ranks = [0.0] * n
            for rank, idx in enumerate(sorted_indices, 1):
                actual_ranks[idx] = float(rank)
            
            # Pearson on the two rank vectors
            mean_e = sum(expected_ranks) / n
            mean_a = sum(actual_ranks) / n
            num = sum((expected_ranks[i] - mean_e) * (actual_ranks[i] - mean_a) for i in range(n))
            den_e = sum((expected_ranks[i] - mean_e) ** 2 for i in range(n)) ** 0.5
            den_a = sum((actual_ranks[i] - mean_a) ** 2 for i in range(n)) ** 0.5
            if den_e > 0 and den_a > 0:
                spearman = round(num / (den_e * den_a), 3)
        except Exception:
            pass

    metrics["tier2_accuracy"] = {
        "total_accuracy_leads": len(accuracy),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "classification_accuracy": round(classification_accuracy * 100, 1),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mae_vs_category_midpoint": round(mae, 1) if mae else None,
        "within_10_pct": round(within_10_pct, 1) if within_10_pct else None,
        "spearman_rank_correlation": spearman,
    }

    # ═══════════════════════════════════════════
    #  TIER 1: Reliability metrics
    # ═══════════════════════════════════════════
    
    rel_checks = []
    flippers = []
    for rid, data in reliability.items():
        scores = data["scores"]
        if len(scores) < 2:
            rel_checks.append({"id": rid, "pass": None, "detail": "not enough runs"})
            continue
        sd = statistics.pstdev(scores)
        mean = statistics.mean(scores)
        above = sum(1 for s in scores if s > EMAIL_THRESHOLD)
        spread = max(scores) - min(scores)
        
        if 0 < above < len(scores):
            flippers.append(f"{rid} ({above}/{len(scores)} above {EMAIL_THRESHOLD}, range {min(scores)}-{max(scores)})")
        
        rel_checks.append({
            "id": rid, "expect": data["expect"],
            "mean": round(mean, 1), "stdev": round(sd, 2), "spread": spread,
            "scores": scores,
            "pass": sd <= MAX_STDEV,
            "detail": f"mean={mean:.1f} stdev={sd:.2f} range={min(scores)}-{max(scores)}"
        })
    
    # Discriminant: strong vs weak in reliability group
    strong_scores = [s for d in reliability.values() if d["expect"] == "strong" for s in d["scores"]]
    weak_scores = [s for d in reliability.values() if d["expect"] == "weak" for s in d["scores"]]
    discriminant = None
    if strong_scores and weak_scores:
        gap = min(strong_scores) - max(weak_scores)
        discriminant = {
            "pass": gap >= MIN_SEPARATION,
            "gap": gap,
            "detail": f"worst strong {min(strong_scores)} - best weak {max(weak_scores)} = {gap}"
        }
    
    metrics["tier1_reliability"] = {
        "per_lead": rel_checks,
        "threshold_stability": {"pass": not flippers, "flippers": flippers},
        "discriminant": discriminant,
    }

    # ═══════════════════════════════════════════
    #  TIER 1: Sensitivity
    # ═══════════════════════════════════════════
    
    senior = sensitivity.get("sens_senior")
    junior = sensitivity.get("sens_junior")
    if senior and junior and senior["score"] is not None and junior["score"] is not None:
        drop = senior["score"] - junior["score"]
        metrics["tier1_sensitivity"] = {
            "senior_score": senior["score"],
            "junior_score": junior["score"],
            "drop": drop,
            "pass": drop >= MIN_SENIORITY_DROP,
            "detail": f"CTO {senior['score']} -> Intern {junior['score']} = -{drop} (bar: >= {MIN_SENIORITY_DROP})"
        }
    else:
        metrics["tier1_sensitivity"] = {"pass": None, "detail": "missing results"}

    # ═══════════════════════════════════════════
    #  TIER 1: Invariance
    # ═══════════════════════════════════════════
    
    inv_checks = []
    for pair_name in ["cosmetic_1", "cosmetic_2"]:
        base_id = f"inv_base_0{pair_name[-1]}"
        var_id = f"inv_var_0{pair_name[-1]}"
        base = invariance.get(base_id)
        var = invariance.get(var_id)
        if base and var and base["score"] is not None and var["score"] is not None:
            drift = abs(base["score"] - var["score"])
            inv_checks.append({
                "pair": pair_name,
                "base_score": base["score"],
                "variant_score": var["score"],
                "drift": drift,
                "pass": drift <= MAX_COSMETIC_DRIFT,
                "detail": f"|{base['score']} - {var['score']}| = {drift} (bar: <= {MAX_COSMETIC_DRIFT})"
            })
    metrics["tier1_invariance"] = inv_checks

    # ═══════════════════════════════════════════
    #  Adversarial checks
    # ═══════════════════════════════════════════
    
    adv_checks = []
    for r in adversarial:
        aid = r["id"]
        score = r["score"]
        notes = (r.get("notes") or "").lower()
        passed = False
        reason = ""

        if aid == "adv_fake_company":
            passed = score is not None and (score < EMAIL_THRESHOLD or "not found" in notes or "unable" in notes or "could not" in notes or "cannot" in notes)
            reason = f"Score {score}, should be < {EMAIL_THRESHOLD} or flag fake company"
        elif aid == "adv_prompt_injection":
            passed = score is not None and score < 95
            reason = f"Score {score}, should NOT comply with injection (< 95)"
        elif aid == "adv_contradictory":
            passed = score is not None and ("contradict" in notes or "inconsist" in notes or score < 60)
            reason = f"Score {score}, should flag contradictions"
        elif aid == "adv_incomplete":
            if score is None:
                passed = True  # erroring gracefully is acceptable
                reason = "Errored gracefully on empty data"
            else:
                passed = score < 50
                reason = f"Score {score}, should score very low for empty lead"
        elif aid == "adv_biased_framing":
            passed = score is not None and score < 85
            reason = f"Score {score}, hype words alone should not inflate above 85"
        elif aid == "adv_duplicate_variation":
            passed = score is not None  # just needs to produce a score
            reason = f"Score {score}, should produce a reasonable score"
        
        adv_checks.append({"id": aid, "score": score, "pass": passed, "reason": reason})
    
    metrics["adversarial"] = adv_checks

    return metrics


def print_report(metrics):
    """Print a human-readable evaluation report."""
    print("\n" + "=" * 74)
    print("FULL EVALUATION REPORT")
    print("=" * 74)
    
    # Tier 2
    t2 = metrics["tier2_accuracy"]
    print("\n── TIER 2: ACCURACY ──")
    cm = t2["confusion_matrix"]
    print(f"  Leads scored:            {t2['total_accuracy_leads']}")
    print(f"  Confusion Matrix @{EMAIL_THRESHOLD}:    TP={cm['TP']}  FP={cm['FP']}  FN={cm['FN']}  TN={cm['TN']}")
    print(f"  Classification Accuracy: {t2['classification_accuracy']}%")
    print(f"  Precision:               {t2['precision']}")
    print(f"  Recall:                  {t2['recall']}")
    print(f"  F1 Score:                {t2['f1']}")
    print(f"  MAE (vs category mid):   {t2['mae_vs_category_midpoint']}")
    print(f"  Within-10:               {t2['within_10_pct']}%")
    print(f"  Spearman (rank corr):    {t2['spearman_rank_correlation']}")

    # Tier 1 Reliability
    t1r = metrics["tier1_reliability"]
    print("\n── TIER 1: RELIABILITY ──")
    for c in t1r["per_lead"]:
        status = "PASS" if c.get("pass") else ("SKIP" if c.get("pass") is None else "FAIL")
        print(f"  [{status}] {c['id']:12}  {c.get('detail', '')}")
        if "scores" in c:
            print(f"           scores: {c['scores']}")
    ts = t1r["threshold_stability"]
    status = "PASS" if ts["pass"] else "FAIL"
    print(f"  [{status}] threshold_stability  {'no flippers' if ts['pass'] else '; '.join(ts['flippers'])}")
    if t1r["discriminant"]:
        d = t1r["discriminant"]
        status = "PASS" if d["pass"] else "FAIL"
        print(f"  [{status}] discriminant         {d['detail']}")

    # Tier 1 Sensitivity
    t1s = metrics["tier1_sensitivity"]
    print("\n── TIER 1: SENSITIVITY ──")
    status = "PASS" if t1s.get("pass") else ("SKIP" if t1s.get("pass") is None else "FAIL")
    print(f"  [{status}] seniority  {t1s.get('detail', '')}")

    # Tier 1 Invariance
    print("\n── TIER 1: INVARIANCE ──")
    for c in metrics["tier1_invariance"]:
        status = "PASS" if c["pass"] else "FAIL"
        print(f"  [{status}] {c['pair']:12}  {c['detail']}")

    # Adversarial
    print("\n── ADVERSARIAL ──")
    adv_passed = 0
    for c in metrics["adversarial"]:
        status = "PASS" if c["pass"] else "FAIL"
        if c["pass"]:
            adv_passed += 1
        print(f"  [{status}] {c['id']:25}  {c['reason']}")
    print(f"  Adversarial: {adv_passed}/{len(metrics['adversarial'])} passed")

    # Overall summary
    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    all_checks = []
    for c in t1r["per_lead"]:
        if c.get("pass") is not None:
            all_checks.append(c["pass"])
    if ts["pass"] is not None:
        all_checks.append(ts["pass"])
    if t1r["discriminant"] and t1r["discriminant"].get("pass") is not None:
        all_checks.append(t1r["discriminant"]["pass"])
    if t1s.get("pass") is not None:
        all_checks.append(t1s["pass"])
    for c in metrics["tier1_invariance"]:
        all_checks.append(c["pass"])
    for c in metrics["adversarial"]:
        all_checks.append(c["pass"])
    
    passed = sum(1 for x in all_checks if x)
    print(f"  Total checks: {passed}/{len(all_checks)} passed")
    print(f"  Classification accuracy: {t2['classification_accuracy']}%")
    print(f"  F1: {t2['f1']}   Precision: {t2['precision']}   Recall: {t2['recall']}")


async def main():
    gemini_key = os.environ["GEMINI_API_KEY"]
    tavily_key = os.environ["TAVILY_API_KEY"]
    icp = load_icp()

    json_path = os.path.join(BASE_DIR, "eval_leads.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_leads = data["leads"]
    
    # Separate by group for different treatment
    accuracy_leads = [lead for lead in all_leads if lead["group"] == "accuracy"]
    reliability_leads = [lead for lead in all_leads if lead["group"] == "reliability"]
    sensitivity_leads = [lead for lead in all_leads if lead["group"] == "sensitivity"]
    invariance_leads = [lead for lead in all_leads if lead["group"] == "invariance"]
    adversarial_leads = [lead for lead in all_leads if lead["group"] == "adversarial"]

    # Count total runs
    single_pass = len(accuracy_leads) + len(sensitivity_leads) + len(invariance_leads) + len(adversarial_leads)
    reliability_runs = len(reliability_leads) * RELIABILITY_REPEATS
    total_runs = single_pass + reliability_runs

    print("=" * 74)
    print("FULL EVALUATION PIPELINE")
    print(f"  Accuracy:    {len(accuracy_leads)} leads x 1 run  = {len(accuracy_leads)} runs")
    print(f"  Reliability: {len(reliability_leads)} leads x {RELIABILITY_REPEATS} runs = {reliability_runs} runs")
    print(f"  Sensitivity: {len(sensitivity_leads)} leads x 1 run  = {len(sensitivity_leads)} runs")
    print(f"  Invariance:  {len(invariance_leads)} leads x 1 run  = {len(invariance_leads)} runs")
    print(f"  Adversarial: {len(adversarial_leads)} leads x 1 run  = {len(adversarial_leads)} runs")
    print(f"  TOTAL: {total_runs} pipeline runs in batches of {BATCH_SIZE}")
    print("=" * 74, flush=True)

    all_results = []

    # ── Phase 1: Accuracy group (single pass, batched) ──
    print(f"\n{'─'*74}")
    print(f"PHASE 1: ACCURACY ({len(accuracy_leads)} leads)")
    print(f"{'─'*74}", flush=True)
    for batch_start in range(0, len(accuracy_leads), BATCH_SIZE):
        batch = accuracy_leads[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(accuracy_leads) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"\n>>> Accuracy Batch {batch_num}/{total_batches}", flush=True)
        items = [(lead, lead["lead"]) for lead in batch]
        results = await score_batch(items, icp, gemini_key, tavily_key, f"acc-B{batch_num}")
        all_results.extend(results)
        # Save intermediate results
        _save_intermediate(all_results)

    # ── Phase 2: Reliability group (N repeats per lead) ──
    print(f"\n{'─'*74}")
    print(f"PHASE 2: RELIABILITY ({len(reliability_leads)} leads x {RELIABILITY_REPEATS} repeats)")
    print(f"{'─'*74}", flush=True)
    for lead in reliability_leads:
        print(f"\n>>> Reliability: {lead['id']} ({lead['lead']['name']} @ {lead['lead']['company']})", flush=True)
        for rep in range(RELIABILITY_REPEATS):
            print(f"  [rep {rep+1}/{RELIABILITY_REPEATS}]", end="", flush=True)
            score, notes, hit = await score_one(lead["lead"], icp, gemini_key, tavily_key)
            print(f" -> {score}", flush=True)
            all_results.append({
                "id": lead["id"], "group": "reliability", "expect": lead["expect"],
                "name": lead["lead"]["name"], "company": lead["lead"]["company"],
                "score": score, "notes": notes, "cache_hit": hit,
            })
        _save_intermediate(all_results)

    # ── Phase 3: Sensitivity (single pass) ──
    print(f"\n{'─'*74}")
    print(f"PHASE 3: SENSITIVITY ({len(sensitivity_leads)} leads)")
    print(f"{'─'*74}", flush=True)
    items = [(lead, lead["lead"]) for lead in sensitivity_leads]
    results = await score_batch(items, icp, gemini_key, tavily_key, "sens")
    all_results.extend(results)
    _save_intermediate(all_results)

    # ── Phase 4: Invariance (single pass) ──
    print(f"\n{'─'*74}")
    print(f"PHASE 4: INVARIANCE ({len(invariance_leads)} leads)")
    print(f"{'─'*74}", flush=True)
    items = [(lead, lead["lead"]) for lead in invariance_leads]
    results = await score_batch(items, icp, gemini_key, tavily_key, "inv")
    all_results.extend(results)
    _save_intermediate(all_results)

    # ── Phase 5: Adversarial (single pass) ──
    print(f"\n{'─'*74}")
    print(f"PHASE 5: ADVERSARIAL ({len(adversarial_leads)} leads)")
    print(f"{'─'*74}", flush=True)
    items = [(lead, lead["lead"]) for lead in adversarial_leads]
    results = await score_batch(items, icp, gemini_key, tavily_key, "adv")
    all_results.extend(results)

    # ── Compute metrics ──
    metrics = compute_metrics(all_results)
    print_report(metrics)

    # ── Save final report ──
    out_dir = os.path.join(ROOT_DIR, "scoring_eval_results")
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    report_path = os.path.join(out_dir, f"full_eval_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_runs": len(all_results),
            "all_results": all_results,
            "metrics": metrics,
        }, f, indent=2, default=str)
    print(f"\nFull report saved: {report_path}")


def _save_intermediate(results):
    """Save intermediate results so progress isn't lost on crash."""
    out_dir = os.path.join(ROOT_DIR, "scoring_eval_results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "eval_intermediate.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"results": results, "count": len(results)}, f, indent=2)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()) or 0)
