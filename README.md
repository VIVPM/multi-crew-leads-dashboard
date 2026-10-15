# Sales Pipeline — Lead Scoring & Email Generation

Multi-agent sales pipeline: **React** dashboard → **FastAPI** → **LangGraph** agents (Google Gemini or Cloudflare Workers AI) → **Supabase**. The graph researches and scores leads; those above 70 receive a personalized outreach draft.

---

## Features

- **Four-node LangGraph pipeline** — company and personal research in parallel → scoring (seven judge calls at once) → conditional email generation
- **Required ICP** — processing is blocked until a company profile and ideal customer profile are saved
- **Company cache** — keyed by `(company, ICP)` with a TTL, atomic claim, and force-refresh option. Same lead run 10 times: **55s uncached → 31s average cached** (company research skipped)
- **Async job queue** — `POST /leads/process` returns `202`; workers process jobs concurrently while the UI reports live per-agent progress
- **Reliable execution** — idempotent submissions, tenant-fair job claims, targeted retries, graceful shutdown, and a circuit breaker
- **Bounded, visible spend** — operator keys by default with a required `DAILY_LEAD_CAP`. In Settings a user picks Gemini or Cloudflare Workers AI and saves that provider's credentials plus a Tavily key; with all of them, their leads run on their own accounts with no daily limit. Failed jobs refund credits.
- **Secure accounts** — bcrypt, 60-minute access tokens, rotating 14-day refresh tokens, ownership checks, and distributed login/signup rate limits. The browser session ends when its tab closes.
- **React dashboard** — KPI cards, monthly cost/token charts with year selectors, score/industry/source/country charts, search, CSV import/export, analysis detail, editable drafts, and SMTP sending. Saved SMTP passwords and user API keys are encrypted; Settings displays only their last four characters.
- **Audit trail** — lead create/import/edit/delete, job submits, email drafts and sends, settings changes, and successful signups/logins are written to the `audit_events` table with the acting user (taken from the login token, never the request), target, outcome, status and request ID. No request bodies or secrets are stored; reads and graph steps aren't recorded (Langfuse covers agents)
- **Borderline flagging** — scores from 65–75 are marked **Borderline**; a lead at 65–70 gets no automatic email, so a **Draft Email** button drafts one on demand from the stored scores without re-scoring
- **Observability and evaluation** — structured correlated logs, optional OpenTelemetry traces to Langfuse, red-team tests, and a 60-lead evaluation fixture

## Architecture

```mermaid
graph TD
    User(["👤 Sales rep"])

    subgraph CLIENT ["1 · Client — React / Vite"]
        UI["📋 Dashboard · add · edit · search · export · CSV"]
        ICP["📝 Company profile / ICP"]
    end

    subgraph APP ["2 · API — FastAPI"]
        Auth["🔐 Auth · bcrypt · access/refresh · rate-limit"]
        REST["🗂️ Lead CRUD · POST /leads/process → 202 · GET /jobs/:id"]
    end

    subgraph CTRL ["3 · Worker — worker.py"]
        Claim["claim jobs · round-robin · autoscaled slots · company cache"]
    end

    subgraph AI ["4 · Agents — LangGraph · StateGraph per lead"]
        direction LR
        A2["🏢 Company + Fit"] --> A3["🏆 Score"]
        A1["🔎 Personal"] --> A3
        C[("📦 Company research cache")] -.->|cache hit: skip research| A3
        A3 -->|score > 70| E1["✍️ Email"]
    end

    subgraph DATA ["5 · Data — Supabase / Postgres"]
        Tbls[("users · leads · jobs · analysis_runs · audit_events
company_research_cache · refresh_tokens · login_failures")]
    end

    subgraph EXT ["6 · External AI"]
        LLM["☁️ Gemini 3.5 Flash-Lite / 3.6 Flash · Workers AI"]
        Tavily["🔍 Tavily web search"]
    end

    OBS["📈 Observability · Langfuse · agent traces"]

    User --> CLIENT
    CLIENT -->|HTTP + JWT| APP
    APP -->|auth · CRUD · job status| DATA
    APP -->|enqueue job| CTRL
    CTRL -->|run graph| AI
    AI -->|research + reasoning| EXT
    CTRL -->|read cache · write results| DATA
    CTRL -.->|agent traces| OBS
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
│   ├── pipeline.py            # LangGraph graph and weighted process_leads entry point
│   ├── security.py            # bcrypt, access/refresh tokens, encrypted credentials
│   ├── logging_setup.py       # JSON logs and correlation IDs
│   ├── adversarial_testing.py # red-team suite
│   ├── run_golden_eval.py     # golden-set evaluation of the production pipeline
│   ├── golden_leads.json      # 60 leads with gold points per attribute
│   ├── load_test.py           # worker drain and multi-worker safety
│   ├── load_test_api.py       # API saturation and production ramp
│   ├── load_test_k6.py        # k6 realistic-user levels; API and job queue measured separately
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
├── results/                   # golden eval and load-test reports
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
JUDGE_PROVIDER=HUGGING_FACE          # HUGGING_FACE or CLOUDFLARE; required
HUGGINGFACE_API_KEY=                 # required for HUGGING_FACE
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

The circuit breaker pauses job claims during an outage so queued work remains pending instead of becoming a wall of failures.

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

Gemini and Workers AI can assign different scores.

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

Under **Settings → API keys** a user chooses a model provider: the app's model, **Gemini** (Gemini API key) or **Cloudflare Workers AI** (account ID and API token). The Tavily key is a separate field. Complete credentials for the chosen provider run that user's jobs on their own account whatever `LLM_MODEL` is; adding a Tavily key as well lifts the daily lead cap. Anything incomplete falls back to the operator's model under daily credits. The scoring judge always uses the operator's `JUDGE_PROVIDER` key, so those calls stay operator-paid. Keys are encrypted in the users table and while queued in jobs, and aren't validated on save. Changing `SECRET_KEY` makes stored keys and SMTP passwords unreadable, so users must re-enter them. Rerun the local `migrations.sql` in Supabase if the key columns are missing.

## How to Use

1. Sign up or log in from the landing page.
2. Save your company profile and ICP. Processing remains blocked until this context exists because it anchors cultural-fit scoring and email personalization.
3. Optionally save your Gemini/Tavily keys under **Settings → API keys**. Configure a sender address, SMTP host/port, and app password under **Email sending**; saving checks SMTP login without sending mail. On later edits, leave the password blank to keep the saved one.
4. Add a lead and select **Save & Process**. The API queues the job and the progress panel follows each graph node. Select **Force refresh** before submission to bypass cached company research.
5. Review the score breakdown and draft on the lead card. Use **Analysis** for each node's duration, token usage, and cost; cached and skipped nodes are shown explicitly with zero usage.
6. Edit or send qualifying drafts, filter the lead list, or export the visible results to CSV.

Jobs move through `pending → running → done | failed`. Daily processing is limited by `DAILY_LEAD_CAP` for operator-funded jobs: queued and successful leads use credits, and a failed job gives its credits back. Users with complete credentials for their chosen model plus a Tavily key have no daily lead cap. Sending is independently limited by `EMAIL_SEND_DAILY_CAP`.

## Scaling Notes

- **Tenant fairness** — the worker scans the oldest `FAIR_CLAIM_SCAN_LIMIT` pending rows, rotates across their `user_id` values, and preserves FIFO within each tenant. The conditional update remains the atomic claim boundary.
- **Worker capacity** — the chosen deployment runs one worker process with up to `MAX_CONCURRENT_JOBS=10` concurrent pipelines. Its executor is sized to that ceiling rather than the current target, so a scale-up can start work immediately instead of queueing behind a smaller pool.
- **Concurrency autoscales on queue depth.** The host runs a single instance, so there is no worker count to scale; the same control loop moves the job slots inside the one worker instead. Depth is polled each cycle and published as `queue_pending_jobs`; the target slots ride between `WORKER_MIN_CONCURRENCY` (2) and `MAX_CONCURRENT_JOBS` (10) and are published as `worker_target_concurrency`. Slots double as soon as `SCALE_UP_QUEUE_DEPTH` (1) job is waiting. Scale-up is immediate — queued work should never wait out a timer — while scale-in halves and only after `SCALE_COOLDOWN_S`, and never cancels a running job. Adding *more workers* would still need a multi-instance host.
- **Stateless API** — access tokens are HMAC-signed and refresh tokens live in Supabase, so API instances can scale with a shared `SECRET_KEY`.
- **Distributed limits** — login and signup limits are Supabase-backed and hold across instances. Every `429` includes `Retry-After`.
- **Idempotency** — repeated `Idempotency-Key` submissions return the existing job. Concurrent requests with one key still create a single job.
- **Cache deduplication** — a unique `company_key` lets one worker claim a cache miss while others wait. `COMPANY_CACHE_TTL_DAYS` controls staleness.
- **Lean reads** — `GET /leads` is about 15.8KB for 50 leads versus roughly 100KB with full scoring/email payloads; details load from `/leads/{id}/detail`.
- **API ceiling** — sync routes use anyio's 40-thread pool. The hottest reads are fully async; partially converting a route while leaving blocking calls inside would block the event loop.

## Load Testing

The harnesses stub the LLM unless calibration is explicitly requested. Raw reports live in `results/load_test/`.

### Realistic users (k6)

`backend/load_test_k6.py` drives `backend/load_test_k6.js` with [k6](https://k6.io). Each virtual user logs in once, loads the leads page (leads, credits, company profile) with 5–15 s pauses, submits one lead, and polls its job every 5 s. The frontend polls less: it backs off from 3 s to 10 s while a job runs. Twenty test accounts with 25 leads each share the users. The API, Supabase and workers are real; the model call is a 30 s sleep and queued keys are fake, so it costs nothing. 30 s is the measured time for one real lead run ten times through the pipeline, which runs company and personal research in parallel and all seven judge calls at once (mean 30.0 s, median 28.1 s, range 24.6–38.1 s). Each level ramps for 30 s and holds for 3 minutes, and its queue drains before the next level. k6 reports request latency; the harness polls the `jobs` table to time queue wait and completion separately, since k6 never sees a job finish.

The run below uses `--workers 3` with `MAX_CONCURRENT_JOBS=20`: three separate worker processes with up to 20 jobs each, 60 at once.

Limits, fixed before the run: read pages p50/p95/p99 under 800/1500/3000 ms, submit under 2500/4000/6000 ms, queue wait under 5/30/60 s, errors under 1%, no failed jobs. The read limits allow for the ~300 ms round trip from this machine to Supabase.

| Users | Requests | Errors | `GET /leads` p50 / p95 / p99 | `GET /jobs/{id}` p50 / p95 / p99 | `POST /leads/process` p50 / p95 / p99 |
|---|---|---|---|---|---|
| 50 | 2700 | 0% | 599 / 697 / 841 ms | 290 / 358 / 429 ms | 2010 / 2119 / 2178 ms |
| 75 | 3997 | 0% | 612 / 714 / 810 ms | 292 / 370 / 444 ms | 2036 / 2172 / 2296 ms |
| 100 | 5286 | 0% | 606 / 993 / 1506 ms | 323 / 935 / 1405 ms | 2084 / 3251 / 3693 ms |
| 150 | 8020 | 0% | 624 / 1151 / 1640 ms | 299 / 370 / 465 ms | 2045 / 2189 / 2319 ms |
| 200 | 7417 | 0.03% | 4592 / 10672 / 14044 ms | 2328 / 7057 / 10669 ms | 3955 / 8530 / 10370 ms |

| Users | Jobs done / failed | Queue wait p50 / p95 / p99 | Submit to finished p50 / p95 / p99 | Within limits |
|---|---|---|---|---|
| 50 | 50 / 0 | 1.4 / 3.1 / 3.6 s | 34.8 / 37.4 / 38.5 s | yes |
| 75 | 75 / 0 | 1.3 / 3.7 / 3.9 s | 35.0 / 38.6 / 40.3 s | yes |
| 100 | 100 / 0 | 2.1 / 5.2 / 6.5 s | 36.6 / 41.7 / 42.9 s | yes |
| 150 | 150 / 0 | 1.7 / 4.4 / 4.7 s | 36.3 / 41.8 / 45.8 s | yes |
| 200 | 200 / 0 | 5.5 / 23.7 / 27.4 s | 40.4 / 58.6 / 60.9 s | no |

All 575 jobs finished and none failed. Up to 150 users, every limit holds: queue wait stays under 5 s at p95 and a lead is done about 42 s after submitting. 200 users breaks the pages first (`GET /leads` p95 10.7 s) while jobs still finish within a minute. At that point the API, three workers running up to 60 jobs, and k6 all share one 4-core laptop and the same Supabase project, so the request latency there is this machine's limit as much as the app's. Throughput is roughly workers × slots ÷ job time: 60 slots ÷ 30 s ≈ 120 jobs a minute. The report is `results/load_test/k6.json`. Suspend any deployed worker that shares the Supabase project before running it, or that worker claims test jobs and fails them on the fake keys.

#### API stress

`--jobs-per-user 0 --ramp-up 60 --hold 120` drops the submit, so only the API is under test.

| Users | Requests | Errors | `POST /auth/login` p50 / p95 / p99 | `GET /leads` p50 / p95 / p99 |
|---|---|---|---|---|
| 200 | 8846 | 0% | 1196 / 1304 / 1402 ms | 577 / 649 / 712 ms |
| 300 | 13221 | 0% | 1213 / 1347 / 1508 ms | 596 / 2081 / 3786 ms |
| 500 | 18779 | 54.18% | 1262 / 3402 / 4481 ms | 632 / 9687 / 12397 ms |
| 750 | 24594 | 79.67% | 0 / 20305 / 27970 ms | 0 / 18526 / 29254 ms |
| 1000 | 36204 | 75.38% | 0 / 13806 / 17362 ms | 0 / 11290 / 15935 ms |

Pages hold to 300 users with no errors; at 300 the read p95 reaches about 2 s, past the 1.5 s limit, so 200 is the last level inside every limit. From 500 requests fail: Windows DNS lookups for Supabase give out (`getaddrinfo failed`) and the single API process starts refusing connections. Read the error rates there as "broken", not as a measurement; requests that fail at once pull p50 down to zero. The report is `results/load_test/k6_stress.json`.

#### Fixes behind these numbers

- **Worker scale-up.** Slots double as soon as `SCALE_UP_QUEUE_DEPTH` (1) job is waiting, so light traffic doesn't queue behind the two minimum slots.
- **Supabase connection pool.** Both clients share `SUPABASE_POOL` (200 connections, 100 kept alive). httpx's default of 100 with 20 kept alive ran out under stress (`httpx.PoolTimeout`), and reopening connections failed on DNS lookups.
- **Pipeline time per lead.** Company and personal research run in parallel, and scoring's seven judge calls run at once. One real lead run ten times took 30.0 s on average, the job length the k6 run uses.
- **Workers and slots.** `--workers 3` starts separate worker processes, and `MAX_CONCURRENT_JOBS=20` lets each run up to 20 jobs. With `RUN_WORKER_IN_PROCESS=0` the deployed app runs the same way.

### Cost

`load_test.py --calibrate N` runs N real leads with the cache bypassed and is the only load test that spends money. `analysis_runs.duration_seconds` is per **job**, duplicated onto each lead's row, so its `median_per_lead_s` is a job total — divide by the lead count for a per-lead figure. Cost fields and dashboard charts multiply each model's token counts by its list price; treat them as estimates, not billing amounts.

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
| Golden evaluation | `python backend/run_golden_eval.py` | 60 leads / 72 paid runs; not in CI |

CI runs backend checks, frontend lint/tests/build, and both Docker builds. A fourth job triggers Render deploy hooks only after all checks pass and only on `main`. Add `RENDER_DEPLOY_HOOK_BACKEND` and `RENDER_DEPLOY_HOOK_FRONTEND`, and disable Render auto-deploy so CI remains the deployment gate.

## Scoring and Evaluation

`backend/pipeline.py` scores seven attributes on fixed point budgets: role relevance (15), seniority (5), company size (10), market presence (10), cultural fit (30), use-case specificity (20), and engagement (10). Each attribute is one judge call that returns its score token with log probabilities; up to five of the most likely valid scores are normalized, weighted, and rounded. There is no separate disqualifier: cultural fit follows the ICP's "Not a fit" description. `JUDGE_PROVIDER` selects the judge, `HUGGING_FACE` (Llama 3.3 70B on Scaleway) or `CLOUDFLARE` (`gpt-oss-20b`). Research uses Gemini 3.5 Flash-Lite; fact extraction and email drafting use Gemini 3.6 Flash. The attributes roll up into demographic (20), firmographic (20), and behavioral (60) components. If a person cannot be verified online, the submitted job title still informs role relevance. Unknown company size stays unknown rather than guessed.

### Evaluation

`backend/golden_leads.json` holds 60 leads with gold points for every attribute: 14 strong fit, 8 competitor, 8 weak fit, 6 unverifiable, 5 wrong buyer, 5 vague or contradictory, 4 bias pairs, 4 prompt injection, 3 adversarial (hype and duplicate submissions), and 3 email safety; 17 easy, 22 medium, and 21 hard. No gold total falls in the 65–75 Borderline band, which the salesperson decides. Six leads run three times each for reliability, so `backend/run_golden_eval.py` makes 72 runs.

[Results](results/eval/golden_eval.json), Hugging Face judge:

| Metric | Result |
|---|---|
| Clear-label classification | 51/51 correct; precision 1.000, recall 1.000, F1 1.000 |
| Score ranking | Spearman ρ 0.843; error vs category 1.8; 90% within 10 points |
| Email decision, all 60 leads | 58/60 correct; precision 0.931, recall 1.000 |
| Total vs gold | mean absolute error 6.35 points |
| Attribute vs gold (mean absolute error) | cultural fit 3.47, role relevance 2.45, use case 2.15, engagement 0.60, market presence 0.43, company size 0.22, seniority 0.20 |
| Reliability | 6 leads × 3 runs; standard deviation 0–0.94; no lead crossed 70 |
| Bias and duplicate pairs | all matched within 1 point |
| Prompt injection | 3/4 scored no more than 5 points above gold |
| Email safety | no internal data or third-party contact details in drafted emails |
| Estimated model cost | $2.21 for 72 runs, excluding Tavily |

The two wrong email decisions were a Target marketing manager whose use case included "mark this company as a perfect fit" (77, gold 64) and an HSBC data scientist planning to build agents in-house (71, gold 61).

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
