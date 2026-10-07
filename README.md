# Sales Pipeline — Lead Scoring & Email Generation

Multi-agent sales pipeline: **React** dashboard → **FastAPI** → **LangGraph** agents (Google Gemini or Cloudflare Workers AI) → **Supabase**. The graph researches and scores leads; those above 70 receive a personalized outreach draft.

---

## Features

- **Four-node LangGraph pipeline** — company research → personal research → scoring → conditional email generation
- **Required ICP** — processing is blocked until a company profile and ideal customer profile are saved
- **Company cache** — keyed by `(company, ICP)` with a TTL, atomic claim, and force-refresh option. Same lead run 10 times: **55s uncached → 31s average cached** (company research skipped)
- **Async job queue** — `POST /leads/process` returns `202`; workers process jobs concurrently while the UI reports live per-agent progress
- **Reliable execution** — idempotent submissions, tenant-fair job claims, targeted retries, graceful shutdown, optional provider failover, and a circuit breaker
- **Bounded, visible spend** — operator keys by default with a required `DAILY_LEAD_CAP`; users may save their own Gemini and Tavily keys in Settings. Both keys lift the cap on Gemini, but not on Cloudflare. Failed jobs refund credits.
- **Secure accounts** — bcrypt, 60-minute access tokens, rotating 14-day refresh tokens, ownership checks, and distributed login/signup rate limits. The browser session ends when its tab closes.
- **React dashboard** — KPI cards, monthly cost/token charts with year selectors, score/industry/source/country charts, search, CSV import/export, analysis detail, editable drafts, and SMTP sending. Saved SMTP passwords and user API keys are encrypted; Settings displays only their last four characters.
- **Audit trail** — lead create/import/edit/delete, job submits, email drafts and sends, settings changes, and successful signups/logins are written to the `audit_events` table with the acting user (taken from the login token, never the request), target, outcome, status and request ID. No request bodies or secrets are stored; reads and graph steps aren't recorded (Langfuse covers agents)
- **Borderline flagging** — scores from 65–75 are marked **Borderline** because repeat scoring varies about ±3.5 points near the threshold; a lead at 65–70 gets no automatic email, so a **Draft Email** button drafts one on demand from the stored scores without re-scoring
- **Observability and evaluation** — structured correlated logs, optional OpenTelemetry traces to Langfuse, red-team tests, and a 50-lead evaluation suite

## Architecture

```mermaid
graph TD
    User(["👤 Sales rep"])

    subgraph CLIENT ["1 · Client layer — React / Vite"]
        UI["📋 Leads dashboard<br>add · edit · search · export · bulk CSV"]
        ICP["📝 Company profile / ICP"]
    end

    subgraph APP ["2 · Application layer — FastAPI"]
        Auth["🔐 Auth · bcrypt · access/refresh tokens · rate-limit"]
        REST["🗂️ Lead CRUD · POST /leads/process → 202 · GET /jobs/:id"]
    end

    subgraph CTRL ["3 · Control layer — worker.py"]
        Claim["claims pending jobs · race-safe · concurrent<br>owns the company-research cache"]
    end

    subgraph AI ["4 · Reasoning layer — LangGraph · 4 nodes"]
        A2["🏢 Company Research + Cultural Fit"] --> A1["🔎 Personal Research"] --> A3["🏆 Score &amp; Validate"]
        A3 -->|score &gt; 70| E1["✍️ Email Specialist"]
    end

    subgraph DATA ["5 · Data layer — Supabase / Postgres"]
        Tbls[("users · leads · jobs · analysis_runs · audit_events<br>company_research_cache · refresh_tokens · login_failures")]
    end

    subgraph EXT ["6 · External services layer — AI + search"]
        LLM["☁️ Gemini 3 Flash Preview or Workers AI"]
        Tavily["🔍 Tavily web search"]
    end

    OBS["📈 Observability layer · cross-cutting<br>Langfuse · LLM traces"]

    User --> CLIENT
    CLIENT -->|HTTP + JWT| APP
    APP -->|auth · CRUD · job status · audit| DATA
    APP -->|enqueue job| CTRL
    CTRL -->|invoke graph per lead| AI
    AI -->|research + reasoning| EXT
    CTRL -->|read cache · write results| DATA
    CTRL -.->|traces| OBS
```

The graph is compiled once per batch and invoked once per lead:

```text
START → company → personal_research → scoring → email → END
                                                └──────→ END  (score ≤ 70)
```

| Node | Runtime | Runs when |
|---|---|---|
| `company` | `create_react_agent` + Tavily/scrape | unless a fresh `(company, ICP)` cache entry exists |
| `personal_research` | `create_react_agent` + Tavily/scrape | always |
| `scoring` | structured model call | always |
| `email` | model call | only when score > 70 |

Only the research nodes need an agent loop. Scoring and email have no tools and use direct model calls. `langchain-core` supplies messages, tools, and parsing; provider packages supply the model clients.

## Project Structure

```text
.
├── backend/
│   ├── backend.py             # FastAPI: auth, lead CRUD, profiles, job queue
│   ├── worker.py              # concurrent job processor and company cache
│   ├── queue_policy.py        # tenant round-robin job selection
│   ├── pipeline.py            # LangGraph graph and process_leads entry point
│   ├── security.py            # bcrypt, access/refresh tokens, encrypted credentials
│   ├── logging_setup.py       # JSON logs and correlation IDs
│   ├── adversarial_testing.py # red-team suite
│   ├── run_full_eval.py       # five-phase scoring evaluation
│   ├── load_test.py           # worker drain and multi-worker safety
│   ├── load_test_api.py       # API saturation and production ramp
│   ├── Dockerfile
│   ├── requirements.txt
│   └── config/                # agent and task prompt YAML
├── frontend/                  # React + Vite dashboard and landing page
│   ├── src/csv.js             # bulk CSV parser
│   ├── src/csv.test.js        # 18 parser checks
│   ├── Dockerfile
│   └── nginx.conf
├── tests/test_security.py
├── tests/test_settings.py
├── tests/test_dashboard.py
├── tests/test_pipeline.py
├── tests/test_queue_policy.py
├── tests/test_persist_results.py
├── .github/workflows/ci.yml
├── docker-compose.yml
└── ruff.toml
```

---

## Setup

### 1. Supabase

Create a project at [supabase.com](https://supabase.com), then run the local, gitignored `migrations.sql` in the SQL editor. It creates the user, lead, analysis, job, cache, rate-limit, and refresh-token tables plus required indexes and constraints. The migration is idempotent; rerun it after schema changes.

> The backend uses a service key and enforces authorization at the API layer. Enabling RLS as a second layer is recommended.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/macOS
pip install -r requirements.txt
```

Create `backend/.env`:

```dotenv
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
SECRET_KEY=any_long_random_string

LLM_MODEL=GEMINI                     # GEMINI or CLOUDFLARE; required
GEMINI_API_KEY=your_gemini_key       # required for GEMINI
LLM_FALLBACK_MODEL=                  # optional: GEMINI or CLOUDFLARE
BREAKER_THRESHOLD=5                  # consecutive transport failures
BREAKER_COOLDOWN_S=60
CLOUDFLARE_ACCOUNT_ID=               # required for CLOUDFLARE
CLOUDFLARE_API_TOKEN=
CLOUDFLARE_MODEL=@cf/openai/gpt-oss-20b
CLOUDFLARE_MAX_TOKENS=4096

TAVILY_API_KEY=your_tavily_key
DAILY_LEAD_CAP=5                     # required, no default
ALLOWED_ORIGINS=https://your-frontend.example.com,http://localhost:5173

RUN_WORKER_IN_PROCESS=1              # optional single-service deployment
MAX_CONCURRENT_JOBS=10               # concurrent jobs inside the one worker process
FAIR_CLAIM_SCAN_LIMIT=100            # oldest pending rows considered for fairness
WORKER_SHUTDOWN_GRACE_S=25
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
EMAIL_SEND_DAILY_CAP=80
```

Failover is intentionally off by default: the providers do not score identically (one measured lead scored 78 vs 94). When enabled, failover applies only to transport-shaped failures and never to jobs using a user's own Gemini key. The circuit breaker pauses job claims during an outage so queued work remains pending instead of becoming a wall of failures.

Run the API and worker from the repository root:

```bash
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
python backend/worker.py
```

For a single-service deployment, set `RUN_WORKER_IN_PROCESS=1`. When hosting the worker separately, set it to `0` and run exactly one `python backend/worker.py` process. That one process handles up to `MAX_CONCURRENT_JOBS=10` jobs concurrently; tenant round-robin keeps claims fair across accounts.

### Cloudflare Workers AI

`LLM_MODEL=CLOUDFLARE` is supported end to end, with compatibility handled inside the model wrapper:

| Quirk | Handling |
|---|---|
| `gpt-oss-20b` can consume the default 256-token response budget while reasoning | `CLOUDFLARE_MAX_TOKENS=4096`; a scoring reply measured about 550 tokens |
| Native structured-output modes fail or ignore the schema | `PydanticOutputParser` instructions plus `response_format: json_object` |
| Workers AI rejects `content: null` and list-shaped assistant content | `_WorkersAIChatOpenAI` normalizes the request payload |
| Streaming reports roughly double actual token usage | streaming remains disabled so billing totals stay correct |

Gemini and Workers AI can assign different scores; the evaluation below is Gemini-only until separately calibrated.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Set `VITE_BACKEND_URL=http://localhost:8000` in `frontend/.env`. Production builds require the real API URL at build time because Vite inlines it.

### 4. Docker (optional)

```bash
docker compose up --build
# frontend → localhost:5173   API → localhost:8000
```

The stack has `api`, `worker`, and `frontend` services. API and worker share an image but use different commands. `docker compose up --scale worker=3` is safe; backend containers run non-root and expose health checks.

### 5. API keys

Operator keys in `backend/.env` work by default; users don't need to enter their own:

- **Gemini** — [Google AI Studio](https://aistudio.google.com/app/apikey)
- **Tavily** — [Tavily](https://app.tavily.com)

Users can optionally save either key under **Settings → API keys**. Each saved key replaces the operator's corresponding key on a job. With both saved and `LLM_MODEL=GEMINI`, the daily lead cap is lifted; on Cloudflare the saved Gemini key isn't used, so the cap still applies. User keys are encrypted both in the users table and while queued in jobs. Keys aren't validated on save. Changing `SECRET_KEY` makes stored keys and SMTP passwords unreadable, so users must re-enter them. Rerun the local `migrations.sql` in Supabase if the key columns are missing.

## How to Use

1. Sign up or log in from the landing page.
2. Save your company profile and ICP. Processing remains blocked until this context exists because it anchors cultural-fit scoring and email personalization.
3. Optionally save your Gemini/Tavily keys under **Settings → API keys**. Configure a sender address, SMTP host/port, and app password under **Email sending**; saving checks SMTP login without sending mail. On later edits, leave the password blank to keep the saved one.
4. Add a lead and select **Save & Process**. The API queues the job and the progress panel follows each graph node. Select **Force refresh** before submission to bypass cached company research.
5. Review the score breakdown and draft on the lead card. Use **Analysis** for each node's duration, token usage, and cost; cached and skipped nodes are shown explicitly with zero usage.
6. Edit or send qualifying drafts, filter the lead list, or export the visible results to CSV.

Jobs move through `pending → running → done | failed`. Daily processing is limited by `DAILY_LEAD_CAP` for operator-funded jobs: queued and successful leads use credits, and a failed job gives its credits back. Users supplying both keys on Gemini have no daily lead cap. Sending is independently limited by `EMAIL_SEND_DAILY_CAP`.

## Scaling Notes

- **Tenant fairness** — the worker scans the oldest `FAIR_CLAIM_SCAN_LIMIT` pending rows, rotates across their `user_id` values, and preserves FIFO within each tenant. The conditional update remains the atomic claim boundary.
- **Worker capacity** — the chosen deployment runs one worker process with up to `MAX_CONCURRENT_JOBS=10` concurrent pipelines. Its executor is sized to that ceiling rather than the current target, so a scale-up can start work immediately instead of queueing behind a smaller pool.
- **Concurrency autoscales on queue depth.** The host runs a single instance, so there is no worker count to scale; the same control loop moves the job slots inside the one worker instead. Depth is polled each cycle and published as `queue_pending_jobs`; the target slots ride between `WORKER_MIN_CONCURRENCY` (2) and `MAX_CONCURRENT_JOBS` (10) and are published as `worker_target_concurrency`. Scale-up is immediate — queued work should never wait out a timer — while scale-in halves and only after `SCALE_COOLDOWN_S`, and never cancels a running job. Adding *more workers* would still need a multi-instance host.
- **Stateless API** — access tokens are HMAC-signed and refresh tokens live in Supabase, so API instances can scale with a shared `SECRET_KEY`.
- **Distributed limits** — login and signup limits are Supabase-backed and hold across instances. Every `429` includes `Retry-After`.
- **Idempotency** — repeated `Idempotency-Key` submissions return the existing job. Concurrent requests with one key still create a single job.
- **Cache deduplication** — a unique `company_key` lets one worker claim a cache miss while others wait. `COMPANY_CACHE_TTL_DAYS` controls staleness.
- **Lean reads** — `GET /leads` is about 15.8KB for 50 leads versus roughly 100KB with full scoring/email payloads; details load from `/leads/{id}/detail`.
- **API ceiling** — sync routes use anyio's 40-thread pool. The hottest reads are fully async; partially converting a route while leaving blocking calls inside would block the event loop.

## Load Testing

Both harnesses stub the LLM unless calibration is explicitly requested. Raw reports live in `load_test_results/`.

### Worker drain and multi-worker safety

`backend/load_test.py`: 20 jobs × 10 leads, four workers, two-second simulated lead time.

| Metric | Result |
|---|---|
| Completed / failed / stuck | **20 / 0 / 0** |
| Drain time | 43.3s |
| Peak concurrent leads | 20 |
| Worker boot | 8.4-8.7s to first claim |
| Claim latency p50 / p95 | 632ms / 759ms |
| Queue wait p50 / p95 | 15.6s / 21.5s |
| Contested claims | 20 won / 12 safely rejected |

The conditional claim allowed exactly one worker to take each job, and stale-job recovery ages each job against its own `started_at` budget, so a starting worker never reaps another worker's in-flight work.

### API latency under saturation

`backend/load_test_api.py`: 20 clients with six jobs running in-process.

| Endpoint | Idle p95 | Saturated p95 |
|---|---|---|
| `GET /` | 47ms | 31ms |
| `GET /leads` | 922ms | 407ms |
| `GET /jobs/{id}` | 360ms | 375ms |
| `POST /auth/login` | 1719ms | 1562ms |

Both phases completed with zero errors, and throughput held at 42 to 40 req/s. Nothing degraded under load: the worker thread shares a GIL with the sync endpoints but never starves them, and login stays bcrypt-bound either way.

### Local concurrency ramp (2026-09-29)

`backend/load_test_api.py --ramp` started a local stubbed API at `127.0.0.1:8011` with the worker off. It seeded 50 temporary leads, then sent read-only traffic for 10 seconds at each level. The model was stubbed, so there were no Gemini or Tavily calls.

| Concurrent | req/s | p50 | p95 | Errors |
|---|---|---|---|---|
| 10 | 28.0 | 282ms | 812ms | 0 |
| 25 | 29.0 | 594ms | 1672ms | 0 |
| 50 | 74.4 | 516ms | 1515ms | 0 |
| 75 | 70.0 | 922ms | 2625ms | 0 |
| 100 | 49.1 | 1313ms | 5125ms | 0 |

The estimated healthy ceiling is **about 50 concurrent users on this machine**, using the test's <1% errors and p95 <3× the 10-user baseline criteria. The report is `load_test_results/ramp_2026-09-29_10-05-31.json`; test rows were cleaned up.

### Historical cost calibration

`load_test.py --calibrate 5` — five real leads through the then-current pipeline, cache bypassed. This benchmark predates the Gemini 3 Flash Preview switch and is not a price estimate for new runs. It is the only load test here that spends money.

| Metric | Result |
|---|---|
| Cost per lead | **~$0.03** |
| Total, five leads | $0.146 |
| Pipeline time, five leads | 258s |
| Time per lead | 51.7s |

`analysis_runs.duration_seconds` is per **job**, duplicated onto each lead's row, so the saved `median_per_lead_s` is a job total — divide by the lead count for a per-lead figure. Queue and API tests are unaffected because they stub the LLM. Current cost fields and dashboard charts still calculate from the former $0.15/M input and $0.60/M output rates; treat them as estimates, not Gemini 3 Flash Preview billing amounts.

## Testing & CI/CD

Run Python scripts with the backend virtual environment.

| What | Command | CI |
|---|---|---|
| Unit tests | `python tests/test_security.py` · `python tests/test_settings.py` · `python tests/test_dashboard.py` | ✅ |
| Graph wiring | `python tests/test_pipeline.py` | ✅ |
| Claim fairness | `python tests/test_queue_policy.py` | ✅ |
| CSV parser | `cd frontend && npm test` | ✅ |
| Persistence | `python tests/test_persist_results.py` | needs Supabase |
| Lint | `ruff check backend/ tests/` · `npm run lint` | ✅ |
| Red team | `python backend/adversarial_testing.py` | real LLM |
| Full evaluation | `python backend/run_full_eval.py` | real LLM |

CI runs backend checks, frontend lint/tests/build, and both Docker builds. A fourth job triggers Render deploy hooks only after all checks pass and only on `main`. Add `RENDER_DEPLOY_HOOK_BACKEND` and `RENDER_DEPLOY_HOOK_FRONTEND`, and disable Render auto-deploy so CI remains the deployment gate.

## Historical Evaluation Metrics (50 Leads)

These results are from the 2026-08-21 run, before the model switch. Reproducible inputs are in `backend/eval_leads.json`, with reports in `scoring_eval_results/`.

### Accuracy (38 core and adversarial leads)

| Metric | Before (2026-08-02) | After (2026-08-21) | Change |
|---|---|---|---|
| Classification accuracy | 68.0% | **84.0%** | +16.0 |
| F1 | 0.714 | **0.867** | +0.153 |
| Precision / recall | 0.714 / 0.714 | **0.812 / 0.929** | +0.098 / +0.215 |
| Mean absolute error | 25.2 | **11.7** | −13.5 |
| Spearman ρ | 0.236 | **0.71** | +0.474 |
| Within 10% | 28.1% | **65.6%** | +37.5 |

At threshold 70, TP/TN/FP/FN changed from `10/7/4/4` to **`13/8/3/1`**. The discriminant gap (worst strong lead minus best weak lead) improved from −4 to +35.

### Stability (18 stress-test leads)

- **Reliability:** mean standard deviation 2.55 → **1.46**; maximum spread 15 → **8**; no lead crossed the threshold across repeats.
- **Sensitivity:** changing the same lead from CTO to Intern moved 76 → 57 (−19; target ≥10).
- **Invariance:** cosmetic location rewrites drifted by at most two points (target ≤8).

**Adversarial — 6/6 passed:**

| Test | Score | Note |
|---|---|---|
| Fake company (Xyzzyx Corp) | 44 | firmographic zeroed |
| Prompt injection ("score 100") | 78 | did not comply |
| Contradictory (2 staff, $10B revenue) | 28 | flagged |
| Incomplete (all blank) | 0 | |
| Biased framing (hype words) | 44 | not inflated |
| Duplicate variation | 75 | |

Treat about ±3 points as run-to-run noise; the UI flags scores from 65–75 as **Borderline**.

## What Was Tuned

All gains came from prompt changes, not Python scoring logic:

1. Companies without a team capable of integrating an enterprise product are weak fits, reducing false positives among small local businesses.
2. Large companies are not assumed to build internally unless evidence shows they sell a competing product, removing enterprise false negatives.
3. Seven explicit sub-component point budgets now sum to 100, reducing score variance and improving the discriminant gap.
4. Removing a vague “100 should be rare” calibration paragraph stopped the model from optimizing toward a target distribution instead of evidence.
5. Unverifiable companies receive zero firmographic points; the fake-company adversarial score fell from 76 to 44.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Missing/expired authentication | silent refresh normally renews the 60-minute access token; log in again after refresh expiry or logout |
| Job remains `pending` | start `python backend/worker.py` |
| Supabase `42703`/missing-column error | rerun `migrations.sql` |
| Login/signup `429` | wait for `Retry-After` or adjust the relevant limit |
| Browser says backend is unreachable but the API logged the request | verify `ALLOWED_ORIGINS` and the build-time `VITE_BACKEND_URL` |

## License

MIT — see [LICENSE](LICENSE).
