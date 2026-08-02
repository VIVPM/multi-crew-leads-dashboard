<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

---

## What this app is

Sales-lead scoring and email drafting. A user adds a lead (or imports a CSV),
CrewAI agents research and score it, and leads scoring above 70 get a drafted
outreach email they can edit and send from their own SMTP account.

Stack: **FastAPI** + **React/Vite** + **CrewAI** (Gemini) + **Supabase**
(PostgREST, not SQLAlchemy), deployed on **Render free tier**.

## Layout

| Path | Holds |
|---|---|
| `backend/backend.py` | All API endpoints, auth, Supabase writes, `persist_results` |
| `backend/pipeline.py` | `build_crews()`, `process_leads()` — CrewAI only, no Supabase |
| `backend/worker.py` | Job queue loop: claims jobs, runs the pipeline, writes results |
| `backend/config/*.yaml` | Agent roles, goals, backstories and task prompts |
| `frontend/src/App.jsx` | Page routing, job polling (`waitForJob`), lead save/process |
| `frontend/src/components/` | `LeadForm`, `LeadsTable` (+ analysis modal), `BulkImport`, `Auth` |
| `frontend/src/App.css` | All styling — one file, no CSS modules |
| `tests/test_security.py` | The only no-network test suite |

## How processing flows

`POST /leads/process` validates, checks the daily credit cap, inserts a `jobs`
row and returns **202 + job_id** — it never waits for the LLM. `worker.py`
claims the job, runs three crews, and writes results; the frontend polls
`GET /jobs/{id}`. `RUN_WORKER_IN_PROCESS=1` runs the worker as a thread inside
the API process (that is how local dev and the single-service deploy run).

Three crews, four agents:
1. `company` — Company Research & Cultural Fit (skipped on a cache hit)
2. `personal_scoring` — Personal Research → Lead Scorer and Validator
3. `email` — Email Specialist (only for scores above 70)

Company research runs **first** because scoring consumes its summary. Each crew
is a separate `kickoff()` with its own measured `token_usage`.

## Commands

Always use the venv Python — these scripts import `pipeline.py`.

```bash
backend/.venv/Scripts/python.exe -m uvicorn backend.backend:app --host 127.0.0.1 --port 8000
backend/.venv/Scripts/python.exe -m ruff check backend/ tests/
backend/.venv/Scripts/python.exe tests/test_security.py     # no network
cd frontend && npm run lint && npm run build && npm test
```

Scripts that spend real money (never run unprompted):
`backend/adversarial_testing.py`, `backend/scoring_eval.py`,
`backend/scoring_gold_set.py --rescore`, `backend/run_full_eval.py` (plus its
`score_eval_leads.py`/`compute_metrics.py` helpers), `backend/load_test.py --calibrate`.

## Things that have burned us

- **Restart the backend after editing `backend/`.** Nothing hot-reloads, and
  with `RUN_WORKER_IN_PROCESS=1` a stale process also runs stale pipeline and
  worker code. A feature can look completely broken for this reason alone.
- **`migrations.sql` is gitignored** and applied by hand in the Supabase SQL
  editor; there is no local Postgres connection string, only the PostgREST key,
  so **DDL cannot be run from here** — ask the user to run it. It is idempotent.
  **Migration before code**: an endpoint selecting a column that does not exist
  yet returns a `42703` error and looks like a total outage.
- **Agent roles carry a trailing newline** — the YAML uses folded scalars
  (`role: >`). Always `.strip()` before comparing a role string.
- **`.alert` is `display: flex`** (a row). Any panel reusing it that stacks
  children must override `flex-direction`.
- **Every route is a sync `def`**, so FastAPI runs it in anyio's threadpool,
  capped at 40 — that is the first ceiling under load. `GET /leads` and
  `GET /jobs/{id}` are genuinely async; a half-converted `async def` that still
  makes blocking calls is worse than leaving it sync.
- Supabase is **PostgREST over HTTP**; `pool_size`/`max_overflow` do not apply.

## Conventions

- **Never** add a `Co-Authored-By` trailer or any AI attribution to commits.
- The user dictates commit timing and dates — do not commit unprompted. Set
  both `GIT_AUTHOR_DATE` and `GIT_COMMITTER_DATE` when they specify a date.
- Prose (comments, commit messages, README) should read as a person wrote it:
  contractions, no "leverage"/"seamless"/"robust", no filler. Say why a choice
  was made, not that it is excellent.
- **No hardcoded fallbacks for real config** — `DAILY_LEAD_CAP` is required and
  fails at startup rather than defaulting.
- Verify before claiming something works: run it, and say plainly when a step
  was skipped or a number is an estimate.
- Local-only planning notes live in `upgrade_roadmap.txt` and
  `evaluation_plan.txt` (both gitignored); keep the roadmap current when a
  feature lands.
