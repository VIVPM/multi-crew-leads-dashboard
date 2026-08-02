"""Compute metrics from intermediate results, score missing adversarial leads, and save final report."""
import os
import sys
import json
import asyncio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from run_full_eval import compute_metrics, print_report, score_batch, load_icp

async def main():
    intermediate = os.path.join(ROOT_DIR, "scoring_eval_results", "eval_intermediate.json")
    with open(intermediate, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_results = data["results"]
    print(f"Loaded {len(all_results)} results from intermediate file.")

    # Load eval leads to get the adversarial leads
    json_path = os.path.join(BASE_DIR, "eval_leads.json")
    with open(json_path, "r", encoding="utf-8") as f:
        leads_data = json.load(f)
    
    adversarial_leads = [lead for lead in leads_data["leads"] if lead["group"] == "adversarial"]
    
    gemini_key = os.environ["GEMINI_API_KEY"]
    tavily_key = os.environ["TAVILY_API_KEY"]
    icp = load_icp()

    print(f"\nScoring remaining {len(adversarial_leads)} adversarial leads...")
    items = [(entry, entry["lead"]) for entry in adversarial_leads]
    adv_results = await score_batch(items, icp, gemini_key, tavily_key, "adv")
    all_results.extend(adv_results)

    print(f"\nComputing metrics for all {len(all_results)} leads...\n")
    metrics = compute_metrics(all_results)
    print_report(metrics)

    # Save final report
    from datetime import datetime
    out_dir = os.path.join(ROOT_DIR, "scoring_eval_results")
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

if __name__ == "__main__":
    asyncio.run(main())
