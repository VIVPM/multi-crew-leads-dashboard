"""
run_golden_eval.py — Scores every lead in golden_leads.json with the production pipeline
and compares the result with the human-reviewed gold points.

Reports accuracy, precision, recall, f1, spearman (percentages), total MAE,
p50/p95/p99 latency, and mean cost per lead — nothing else. Results are written
after every lead, so an interrupted run keeps its progress. Costs money: each
lead runs the full research, judge, and email path.

    python backend/run_golden_eval.py [--limit N] [--ids id1,id2]
"""

import argparse
import asyncio
import json
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
sys.path.insert(0, BASE)

import pipeline  # noqa: E402
from logging_setup import job_context  # noqa: E402
from tracing_setup import setup_langfuse_tracing  # noqa: E402

CRITERIA = re.compile(r"\((\d+)/(\d+)\)\s*$")


class UsageMeter:
    """Wraps the pipeline's model calls to count tokens per provider for cost estimates."""

    def __init__(self):
        self.tokens = {"flash": [0, 0], "lite": [0, 0], "judge": [0, 0]}
        self._research, self._chat, self._judge = pipeline._research, pipeline._chat, pipeline._judge_attribute
        pipeline._research, pipeline._chat, pipeline._judge_attribute = self.research, self.chat, self.judge

    def _tier(self, llm):
        return "lite" if "lite" in str(getattr(llm, "model", "")) else "flash"

    def research(self, llm, *args, **kwargs):
        out = self._research(llm, *args, **kwargs)
        self._add(self._tier(llm), out[1], out[2])
        return out

    def chat(self, llm, *args, **kwargs):
        out = self._chat(llm, *args, **kwargs)
        self._add(self._tier(llm), out[2], out[3])
        return out

    def judge(self, rubric, instruction, evidence):
        out = self._judge(rubric, instruction, evidence)
        self._add("judge", *out[2])
        return out

    def _add(self, key, prompt, completion):
        self.tokens[key][0] += prompt or 0
        self.tokens[key][1] += completion or 0

    def take(self) -> dict:
        """Return this lead's tokens and estimated USD, then reset the counters."""
        f, lt, j = self.tokens["flash"], self.tokens["lite"], self.tokens["judge"]
        usd = (pipeline._estimated_cost(f[0], f[1], "main")
               + pipeline._estimated_cost(lt[0], lt[1], "research")
               + pipeline._estimated_cost(j[0], j[1], "judge"))
        snapshot = {"tokens": {k: list(v) for k, v in self.tokens.items()}, "estimated_usd": round(usd, 5)}
        self.tokens = {"flash": [0, 0], "lite": [0, 0], "judge": [0, 0]}
        return snapshot


def attribute_points(criteria: list, names: list) -> dict:
    """Map the scorer's criteria lines, written in rubric order, back to attribute names."""
    points = {}
    for name, line in zip(names, criteria):
        match = CRITERIA.search(line)
        points[name] = int(match.group(1)) if match else None
    return points


def score_lead(record: dict, icp: str, meter: UsageMeter) -> dict:
    """Run one golden lead through the weighted pipeline and capture its outputs."""
    started = time.time()
    names = list(pipeline._CONFIGS["lead_tasks"]["weighted_scoring"]["attributes"])
    try:
        llm_key = (os.environ["GEMINI_API_KEY"] if pipeline.LLM_MODEL == "GEMINI"
                   else os.environ["CLOUDFLARE_API_TOKEN"])
        scores, emails, _, _ = asyncio.run(pipeline.process_leads(
            [record["lead"]], llm_key, os.environ["TAVILY_API_KEY"], icp,
            max_retries=2, force_refresh=True,
        ))
        result = scores[0].pydantic
        company = scores[0].company_output.pydantic if scores[0].company_output else None
        email = emails[0]
        return {
            "id": record["id"], "ok": True, "seconds": round(time.time() - started, 1),
            "attributes": attribute_points(result.lead_score.scoring_criteria, names),
            "total": result.lead_score.score,
            "email": email is not None,
            "email_text": email.raw if email else None,
            "cultural_fit_notes": company.cultural_fit_notes if company else None,
            "criteria": result.lead_score.scoring_criteria,
            **meter.take(),
        }
    except Exception as exc:
        return {"id": record["id"], "ok": False, "seconds": round(time.time() - started, 1),
                "error": f"{type(exc).__name__}: {str(exc)[:300]}", **meter.take()}


CATEGORY_RANGE = {"disqualified": (0, 70), "weak": (0, 64), "borderline": (65, 75), "strong": (76, 100)}
CATEGORY_MID = {"disqualified": 35, "weak": 32, "borderline": 70, "strong": 88}
CATEGORY_RANK = {"disqualified": 1, "weak": 2, "borderline": 3, "strong": 4}


def expected_category(record: dict) -> str:
    """Bucket a golden lead by competitor or gold total, matching the app's own 65-75 Borderline band."""
    if record["category"] == "competitor":
        return "disqualified"
    total = record["gold"]["total"]
    return "borderline" if 65 <= total <= 75 else "strong" if total > 75 else "weak"


def _ranks(values: list) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    for rank, i in enumerate(order, 1):
        ranks[i] = float(rank)
    return ranks


def headline_metrics(done: list, results: dict) -> dict:
    """The original eval's headline numbers: clear-label classification, category error, and rank order.

    Borderline leads are left out of classification, as before. A competitor counts as inside
    its category at any score up to 70, since its other attributes can still earn points.
    """
    tp = fp = fn = tn = 0
    errors = []
    for r in done:
        cat, score = expected_category(r), results[r["id"]]["total"]
        above = score > 70
        if cat == "strong":
            tp += above
            fn += not above
        elif cat in ("weak", "disqualified"):
            fp += above
            tn += not above
        low, high = CATEGORY_RANGE[cat]
        errors.append(0 if low <= score <= high else abs(score - CATEGORY_MID[cat]))
    classified = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    expected = [CATEGORY_RANK[expected_category(r)] for r in done]
    actual = _ranks([results[r["id"]]["total"] for r in done])
    spearman = None
    if len(done) >= 3:
        me, ma = statistics.mean(expected), statistics.mean(actual)
        num = sum((e - me) * (a - ma) for e, a in zip(expected, actual))
        den = (sum((e - me) ** 2 for e in expected) * sum((a - ma) ** 2 for a in actual)) ** 0.5
        spearman = round(num / den, 3) if den else None
    return {
        "clear_label_leads": classified,
        "classification_accuracy": round((tp + tn) / classified * 100, 1) if classified else None,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "error_vs_category": round(statistics.mean(errors), 1) if errors else None,
        "within_10_of_category_pct": round(sum(e <= 10 for e in errors) / len(errors) * 100, 1) if errors else None,
        "spearman": spearman,
    }


def _pctl(values: list, p: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[idx]


def _pct(ratio: Optional[float]) -> Optional[float]:
    """0-1 ratio as a 0-100 percentage, 1 decimal."""
    return round(ratio * 100, 1) if ratio is not None else None


def summarize(records: list, results: dict) -> dict:
    """Accuracy, precision, recall, f1, spearman, MAE, latency and cost per lead — nothing else."""
    done = [r for r in records if results.get(r["id"], {}).get("ok")]
    total_err = [abs(results[r["id"]]["total"] - r["gold"]["total"]) for r in done]
    seconds = [results[r["id"]]["seconds"] for r in done]
    costs = [results[r["id"]].get("estimated_usd", 0) for r in done]
    headline = headline_metrics(done, results)
    return {
        "accuracy": headline["classification_accuracy"],
        "precision": _pct(headline["precision"]),
        "recall": _pct(headline["recall"]),
        "f1": _pct(headline["f1"]),
        "spearman": _pct(headline["spearman"]),
        "mae": round(statistics.mean(total_err), 2) if total_err else None,
        "latency_seconds": {
            "p50": round(_pctl(seconds, 50), 1),
            "p95": round(_pctl(seconds, 95), 1),
            "p99": round(_pctl(seconds, 99), 1),
        } if seconds else None,
        "cost_usd_per_lead": round(statistics.mean(costs), 4) if costs else None,
    }


def main():
    """Score the golden set (optionally a subset) and write results plus a summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int)
    parser.add_argument("--ids")
    args = parser.parse_args()

    with open(os.path.join(BASE, "golden_leads.json"), encoding="utf-8") as fh:
        records = json.load(fh)["leads"]
    if args.ids:
        wanted = set(args.ids.split(","))
        records = [r for r in records if r["id"] in wanted]
    if args.limit:
        records = records[:args.limit]
    with open(os.path.join(BASE, "..", "company_icp.txt"), encoding="utf-8") as fh:
        icp = fh.read()

    out_dir = os.path.join(BASE, "..", "results", "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"golden_eval_{datetime.now():%Y-%m-%d_%H-%M-%S}.json")
    meter = UsageMeter()
    trace_provider = setup_langfuse_tracing(logging.getLogger("golden_eval"))
    results = {}
    try:
        for i, record in enumerate(records, 1):
            runs = []
            for repeat in range(record.get("repeats", 1)):
                with job_context(f"golden-eval-{record['id']}-{repeat + 1}"):
                    runs.append(score_lead(record, icp, meter))
            r = runs[0]
            if len(runs) > 1:
                r["repeat_totals"] = [x["total"] for x in runs if x["ok"]]
                r["estimated_usd"] = round(sum(x["estimated_usd"] for x in runs), 5)
            results[record["id"]] = r
            status = (f"total {r['total']} (gold {record['gold']['total']}) email {r['email']} (gold {record['gold']['email']})"
                      if r["ok"] else f"FAILED {r['error']}")
            print(f"[{i}/{len(records)}] {record['id']:<10} {status} | {r['seconds']}s ~${r['estimated_usd']:.4f}", flush=True)
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"results": results, "summary": summarize(records, results)}, fh, indent=2, ensure_ascii=False)

        summary = summarize(records, results)
        print("\nSUMMARY")
        print(json.dumps(summary, indent=2))
        print("Saved:", out_path)
    finally:
        if trace_provider and not trace_provider.force_flush():
            print("Warning: Langfuse trace flush did not finish.", file=sys.stderr)


if __name__ == "__main__":
    main()
