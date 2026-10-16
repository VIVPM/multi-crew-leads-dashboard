"""Exercise evaluation trace setup, per-run correlation, and flush without model calls."""

import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
os.environ.setdefault("LLM_MODEL", "GEMINI")
import run_golden_eval as eval_runner  # noqa: E402
from logging_setup import current_correlation_ids  # noqa: E402


def test_evaluation_tracing():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        backend = root / "backend"
        backend.mkdir()
        (backend / "golden_leads.json").write_text(json.dumps({"leads": [{
            "id": "sample", "lead": {}, "gold": {"total": 65, "email": False}, "repeats": 2,
        }]}), encoding="utf-8")
        (root / "company_icp.txt").write_text("Test ICP", encoding="utf-8")
        seen = []

        def fake_score(record, icp, meter):
            seen.append(current_correlation_ids()["job_id"])
            return {"id": record["id"], "ok": True, "total": 65, "email": False,
                    "seconds": 0, "estimated_usd": 0}

        provider = Mock()
        provider.force_flush.return_value = True
        with (patch.object(eval_runner, "BASE", str(backend)),
              patch.object(eval_runner, "score_lead", side_effect=fake_score),
              patch.object(eval_runner, "summarize", return_value={"checks": []}),
              patch.object(eval_runner, "setup_langfuse_tracing", return_value=provider) as setup,
              patch.object(sys, "argv", ["run_golden_eval.py"])):
            eval_runner.main()

        assert seen == ["golden-eval-sample-1", "golden-eval-sample-2"]
        setup.assert_called_once()
        provider.force_flush.assert_called_once()
        assert (root / "results" / "eval").is_dir()


if __name__ == "__main__":
    test_evaluation_tracing()
    print("test_eval_tracing.py: all checks passed")
