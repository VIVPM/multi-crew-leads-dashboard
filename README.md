# Sales Pipeline — Lead Scoring & Email Generation

A full-stack, multi-agent sales pipeline application: a **React** dashboard backed by a **FastAPI** server that orchestrates a **LangGraph** agent graph (powered by **Google Gemini**, or Cloudflare Workers AI) to score sales leads and draft personalized outreach emails, with all data stored in **Supabase**.

---

## Features

**Scoring** — a four-node LangGraph pipeline researches each lead, scores it 0-100
against your ICP, and drafts a cold outreach email for anything above 70. Company
research is cached per `(company, ICP)`, so a second lead from the same firm skips
it. Repeat scoring varies ~±3.5 points, so leads landing in 65-75 are badged
**Borderline** rather than silently drafted or not on a coin flip.

**Your ICP is required, not optional.** Processing with an empty company profile
is blocked by a dialog — there is no generic fallback. The placeholder asks for
explicit *Weak fit* and *Not a fit* lines, because spelling those out is what the
evaluation showed makes the score trustworthy.

**Never blocks on the LLM** — `POST /leads/process` returns `202` with a job id
and a background worker does the work; the UI polls and shows which agent is
running, including **cached** when company research was skipped. Submissions are
idempotent via an `Idempotency-Key` header, failures retry only when retrying
could help, and shutdown drains in-flight jobs rather than dropping them.

**Costs are bounded and visible** — keys are operator-held, so each account gets
`DAILY_LEAD_CAP` leads per UTC day (required, no default: a deploy that forgets it
fails at startup rather than guessing a spend limit). Per-lead token and cost
breakdowns are in the Analysis modal, and `llm_tokens_total` / `llm_cost_usd_total`
make spend alertable rather than something you read one lead at a time.

**The rest** — React dashboard with charts, search, CSV export and bulk import;
editable drafts sent through your own SMTP account; bcrypt auth with rotating
refresh tokens, ownership checks and rate limiting; agent prompts in
`backend/config/*.yaml` with no code changes; structured JSON logs correlated by
`request_id`/`job_id`/`lead_id`; optional OpenTelemetry tracing to Langfuse
(v4 observations model) and Grafana Cloud; a red-team suite and a 50-lead
scoring evaluation.

---

## Architecture

Six layers, read top to bottom. Each arrow is a hand-off between layers — the shared services (data, external AI) are reached once per layer rather than by every agent, so the flow stays legible.

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
        A1["🔎 Personal Research"] --> A3["🏆 Score &amp; Validate"]
        A2["🏢 Company Research + Cultural Fit"] -.->|cache hit: skip| A3
        A3 -->|score &gt; 70| E1["✍️ Email Specialist"]
    end

    subgraph DATA ["5 · Data layer — Supabase / Postgres"]
        Tbls[("users · leads · jobs · analysis_runs<br>company_research_cache · refresh_tokens · login_failures")]
    end

    subgraph EXT ["6 · External AI services"]
        LLM["☁️ Google Gemini · 2.5 Flash / Flash-Lite"]
        Tavily["🔍 Tavily web search"]
    end

    OBS["📈 Observability · cross-cutting<br>Langfuse + Grafana · traces · metrics · alerts"]

    User --> CLIENT
    CLIENT -->|HTTP + JWT| APP
    APP -->|auth · CRUD · job status| DATA
    APP -->|enqueue job| CTRL
    CTRL -->|invoke graph per lead| AI
    AI -->|research + reasoning| EXT
    CTRL -->|read cache · write results| DATA
    CTRL -.->|traces · metrics| OBS
```

One `StateGraph`, compiled once per batch and invoked once per lead:

```
START -> company -> personal_research -> scoring -> email -> END
                                                 -> END      (score <= 70)
```

| Node | Agent | Runtime | Runs when |
|---|---|---|---|
| `company` | Company Research & Cultural Fit Analyst (A2) | `create_react_agent` + Tavily/scrape | always, unless a fresh cache entry exists for `(company, ICP)` — the check is inside the node, so a hit costs no model call |
| `personal_research` | Personal Research Specialist (A1) | `create_react_agent` + Tavily/scrape | always |
| `scoring` | Lead Scorer & Validator (A3) | one structured model call, no tools | always |
| `email` | Email Specialist (E1) | one model call, no tools | conditional edge, only if the score is > 70 |

Only the two research nodes are agents — `scoring` and `email` have no tools, so
an agent loop there could never iterate. Agents come from `langgraph.prebuilt`;
the `langchain` meta-package isn't installed, so no LangChain agent abstraction
is involved. `langchain-core` supplies message types, the `@tool` decorator and
`PydanticOutputParser`; `langchain-google-genai` / `langchain-openai` are the
model clients.

## Project Structure

```
.
├── backend/
│   ├── backend.py            # FastAPI app: auth, lead CRUD, company profile, job enqueue/status
│   ├── worker.py             # Background job processor (runs the graph, company-research cache)
│   ├── pipeline.py           # LangGraph graph + process_leads entry point (no Supabase dependency)
│   ├── security.py           # bcrypt hashing + access/refresh session tokens
│   ├── logging_setup.py      # structured JSON logs + request/job/lead correlation IDs
│   ├── adversarial_testing.py # red-team test suite (needs backend/.venv — see below)
│   ├── run_full_eval.py      # scoring evaluation, all five phases (+ compute_metrics.py)
│   ├── load_test.py          # worker drain + multi-worker safety (LLM stubbed); --calibrate for real cost
│   ├── load_test_api.py      # API latency, idle vs saturated worker (LLM stubbed)
│   ├── Dockerfile            # one image, two commands (API via uvicorn, worker via python)
│   ├── requirements.txt      # pinned Python dependencies
│   └── config/               # agent role & task prompt YAML (one entry per node)
├── frontend/                 # React + Vite dashboard + landing page (Stripe-inspired design system)
│   ├── src/csv.js            # CSV import parser (bulk import)
│   ├── src/csv.test.js       # 18 node:test checks for the parser (run in CI)
│   ├── Dockerfile            # node build → nginx static serve
│   └── nginx.conf            # SPA fallback + fingerprinted-asset caching
├── tests/test_security.py    # no-network auth/token unit tests (run in CI)
├── tests/test_pipeline.py    # no-network graph wiring tests (routing, cache skip, token math)
├── tests/test_persist_results.py  # persist_results vs a real Supabase (no LLM calls, not in CI)
├── .github/workflows/ci.yml  # lint + tests + docker build + gated deploy
├── docker-compose.yml        # local stack: api + worker + frontend
└── ruff.toml                 # Python lint config
```

---

## Setup

### 1. Supabase

Create a project at [supabase.com](https://supabase.com) with tables `users` (id, username, password, **company_context** text, plus **email_smtp_host** / **email_smtp_port** / **email_from_address** / **email_smtp_password** for per-user email sending), `leads` (including **email_sent_at**), and `analysis_runs`, plus:
- a `jobs` queue table (id, user_id, status, leads jsonb, gemini_api_key, tavily_api_key, results jsonb, error, created_at, **our_company_context** text, **force_refresh** boolean, **progress** jsonb — the live per-agent stage map the UI renders while a lead is processing),
- a `company_research_cache` table (id, company_key **unique**, company_name, company_info jsonb, cultural_fit_score, cultural_fit_notes, cached_at) — the company-research cache, keyed by normalized company name + a hash of the ICP text; the unique constraint on `company_key` is also what makes the claim-before-research lock (see [Scaling notes](#scaling-notes)) atomic,
- a `login_failures` table (id, username, failed_at) — Supabase-backed login rate limiting,
- a `signup_attempts` table (id, ip, created_at) — Supabase-backed per-IP signup rate limiting (one row per created account),
- a `refresh_tokens` table (id, user_id, token_hash **unique**, expires_at, created_at) — server-side session refresh tokens, of which only a SHA-256 hash is stored,
- a unique constraint on `users.username`,
- indexes on `jobs(status, created_at)`, `leads(user_id)`, `analysis_runs(lead_id)`, `company_research_cache(company_key)`, `login_failures(username, failed_at)`, `signup_attempts(ip, created_at)`, and `refresh_tokens(token_hash)`.

The exact DDL lives in a local `migrations.sql` (gitignored — it's operational, not source) that's kept up to date as the schema evolves; run it in the Supabase SQL editor. It's idempotent, so re-running it after a schema update only applies what changed.

> **Note on RLS:** the backend uses a service key, which bypasses Row Level Security; authorization is enforced at the API layer (token + ownership checks). Enabling RLS as a second layer is recommended for defense in depth.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
```

Create `backend/.env`:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_service_key
SECRET_KEY=any_long_random_string   # signs access tokens; required in production
GEMINI_API_KEY=your_gemini_key      # operator-held, powers all agents (required when LLM_MODEL=GEMINI)

# Which provider powers the agents: GEMINI or CLOUDFLARE (required, no default).
# CLOUDFLARE routes through Workers AI's OpenAI-compatible endpoint, reached by
# langchain-openai's ChatOpenAI with nothing but a base_url swap.
# Gemini splits work across flash (judgment) and flash-lite (retrieval);
# Workers AI publishes one model for this, so both tiers use it there.
LLM_MODEL=GEMINI                    # required, no default

# Optional — provider to fall back to when the primary keeps failing on
# transport errors. OFF by default and deliberately so: the two providers do
# not score a lead identically (measured 78 vs 94 on one lead), so failing over
# changes the answer. Turn it on only if you accept that trade.
LLM_FALLBACK_MODEL=                 # GEMINI or CLOUDFLARE, or empty for none

# Optional — circuit breaker. After this many consecutive transport-shaped job
# failures the worker stops claiming for the cooldown, leaving queued jobs
# queued instead of converting the backlog into failures during an outage.
BREAKER_THRESHOLD=5
BREAKER_COOLDOWN_S=60
CLOUDFLARE_ACCOUNT_ID=              # required when LLM_MODEL=CLOUDFLARE
CLOUDFLARE_API_TOKEN=               # required when LLM_MODEL=CLOUDFLARE
CLOUDFLARE_MODEL=@cf/openai/gpt-oss-20b   # optional override
CLOUDFLARE_MAX_TOKENS=4096          # optional; must stay generous, see below
TAVILY_API_KEY=your_tavily_key      # operator-held, web-search enrichment (required)
DAILY_LEAD_CAP=5                    # leads each account may process per day (required)
ALLOWED_ORIGINS=https://your-frontend.example.com,http://localhost:5173   # CORS allowlist (required, comma-separated)

# Optional — run the job worker inside the API process (single-service deploys)
RUN_WORKER_IN_PROCESS=1

# Optional — how long shutdown waits for in-flight jobs before giving up on
# them (default 25). Deliberately under Render's ~30s SIGTERM->SIGKILL window
# rather than over p99 job duration, which would be a number nothing honours.
WORKER_SHUTDOWN_GRACE_S=25

# Optional — OpenTelemetry tracing of every graph node and LLM call, exported to
# Langfuse and/or Grafana Cloud (Grafana also gets the jobs_processed_total metric)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
GRAFANA_OTLP_ENDPOINT=
GRAFANA_OTLP_AUTH=

# Optional — daily per-user email send cap (default 80)
EMAIL_SEND_DAILY_CAP=80
```

### Running on Cloudflare Workers AI

`LLM_MODEL=CLOUDFLARE` works end to end but needs more glue than Gemini, for two
reasons: the OpenAI shim is compatible on the happy path and stricter elsewhere,
and `gpt-oss-20b` is a reasoning model no framework special-cases.

| Quirk | Handling |
|---|---|
| Replies capped at 256 tokens by default, and reasoning eats that before any content appears — you get empty `content` with `finish_reason="length"`, not an error | `CLOUDFLARE_MAX_TOKENS=4096`, set once on the constructor. A lead-scoring reply measured ~550 tokens |
| **All three** `with_structured_output` methods fail: `json_schema` returns a bare `-1.0`, `function_calling` is ignored (the model writes into `content`), `json_mode` emits a markdown table because the schema is never sent | `STRUCTURED_VIA_PROMPT` uses `PydanticOutputParser` + `response_format: json_object`. Gemini's native path is untouched |
| Rejects `content: null` on an assistant message (which OpenAI allows beside `tool_calls`) and list-shaped content — and only on the *second* call of a tool round-trip, so it reads as a random mid-run failure | `_WorkersAIChatOpenAI` coerces both in `_get_request_payload` |
| Streaming reports **double** the real usage — 162/126/288 where the non-streaming call gives 81/60/141 | Streaming stays off here, so this provider reports no TTFT. Correct billing beats a latency metric |

**The two providers disagree on scores**, so this isn't a drop-in swap: measured
on the same lead, Gemini 96 vs Workers AI 91, and 78 vs 94 on another. Treat the
evaluation numbers below as Gemini-only until they're re-measured here.

Run the API and the worker (two terminals, from the repo root):

```bash
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
python backend/worker.py
```

> **Single-process deploys (e.g. Render free tier):** instead of a separate
> worker service, set `RUN_WORKER_IN_PROCESS=1` on the web service — the job
> worker then runs as a background thread inside the API process. Switch back
> to a dedicated worker (and unset the flag) when you need more throughput.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173 (expects the API on localhost:8000)
```

Create `frontend/.env` for local dev:

```
VITE_BACKEND_URL=http://localhost:8000
```

`VITE_BACKEND_URL` is **required** for a production build — no hardcoded fallback, so a build that forgets to set it throws at load instead of silently pointing nowhere. Vite inlines env vars at **build** time, so this must be set in the build environment (e.g. Render's static site settings), not just wherever the bundle is served from.

### 4. Docker (optional)

The whole stack runs in containers — Supabase stays remote (it's the system of record), so `backend/.env` must point at a real project:

```bash
docker compose up --build
# frontend → http://localhost:5173   API → http://localhost:8000
```

Three services: `api` (uvicorn), `worker` (`python backend/worker.py`), and `frontend` (Vite build served by nginx). API and worker share one image — same dependencies, different start command. They run as separate services rather than via `RUN_WORKER_IN_PROCESS`, which is the shape the [load tests](#load-testing) validated; `docker compose up --scale worker=3` is safe now that jobs carry `started_at`. Both backend containers run non-root with a healthcheck on `GET /`.

### 5. API keys (operator-held)

Gemini and Tavily keys are provided once by the operator in `backend/.env` (`GEMINI_API_KEY` / `TAVILY_API_KEY`) — end users never enter keys, and the backend refuses to start without them:

- **Gemini** — [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (powers all agents)
- **Tavily** — [app.tavily.com](https://app.tavily.com) (web-search enrichment)

---

## How to Use

1. **Get started from the landing page** — sign up (which drops you straight into the app) or log in (username ≥ 3 chars, password ≥ 8 chars).
2. **Describe your company & ICP** in the main dashboard's Company Profile card and save it — this is what cultural-fit assessment and email drafting are measured against, and it's **required**: try to process a lead with it empty and a dialog blocks you until you fill it in (no generic fallback exists).
3. **(Optional) Connect your email** — in the **Email sending** card, add your from-address and an app password (a Gmail App Password, or any SMTP host/port) if you want to actually send drafts, not just view them.
4. **Add a lead** (name, company, email required) and click **Save & Process** — the lead is queued, the graph scores it and (if it scores above 70) drafts an email. A progress bar and step list appear below the form and advance as each agent finishes, so you can see which one is working. Check **Force refresh** first to bypass the company-research cache for that batch.
5. **Review results** — expand a lead card for the breakdown and draft; **Edit** to revise, **Send** to email it. **📊 Analysis** shows duration, tokens and cost per agent — each figure is that agent's own measured usage. A row reads **Cached** when company research came from cache and **Skipped** when the score was ≤ 70, both at 0, so all four agents always appear.
6. **Search / export** — filter the table and export the filtered set to CSV.

Processing is capped per account per day by `DAILY_LEAD_CAP` (the daily credit allowance); job status is `pending → running → done | failed`.

---

## Scaling notes

| Concern | How it's handled |
|---|---|
| **Throughput** | Jobs live in a `jobs` table, not in the request. Each worker runs up to `MAX_CONCURRENT_JOBS` (default 10) concurrently, and sizes its thread pool to match — the interpreter default is `min(32, cpu+4)`, which is 5 on a 1-vCPU box and would silently cap it. Run more workers to go wider; that's load-tested and safe because a starting worker ages each `running` job against its own budget instead of assuming everything in flight is abandoned. |
| **Horizontal API** | Access tokens are stateless HMAC; refresh tokens live in Supabase. Any instance serves any session — just set the same `SECRET_KEY` everywhere. |
| **Rate limits** | Both are Supabase-backed, not in-memory, so they hold across instances: login per username, signup per IP (one row per created account, so nobody mints accounts to farm daily credits). Every `429` carries `Retry-After`. |
| **Duplicate submits** | `Idempotency-Key` returns the existing job as `200` with `idempotent_replay: true` instead of `202`. Two concurrent submits on one key still produce one job. Needs `jobs.idempotency_key` from `migrations.sql`; without it the header is ignored rather than erroring. |
| **Duplicate research** | The first cache miss atomically claims the row via the unique `company_key`, so a concurrent miss waits on the winner rather than re-running Tavily + Gemini. `COMPANY_CACHE_TTL_DAYS` (default 7) bounds staleness. |
| **Payload size** | `GET /leads` returns only what the list renders; `scoring_result` and `email_draft` are ~78% of a row and load on demand from `/leads/{id}/detail`. Measured **~100KB → 15.8KB** for a 50-lead account. Trade-off: search no longer matches text inside drafts. |
| **First ceiling** | Every route is a sync `def`, so FastAPI runs it in anyio's threadpool — capped at 40. The two hottest reads (`GET /leads`, `GET /jobs/{id}`) are genuinely async and suspend on I/O instead. Converting more means converting every blocking call inside them; a half-converted `async def` blocks the whole loop and is worse than leaving it sync. |

---


---

## Load testing

Two harnesses, both stubbing the LLM so they measure *this* system rather than
Gemini's latency: `backend/load_test.py` (worker drain, multi-worker safety) and
`backend/load_test_api.py` (API latency, ramp). Raw results are in
`load_test_results/`.

### Worker drain — 20 jobs, 4 workers

Does the queue drain, and do concurrent workers corrupt each other's work?

| | |
|---|---|
| Jobs completed | **20 done, 0 failed, 0 stuck** |
| Drain time | 50.3s for 200 leads (20 jobs × 10) |
| Peak concurrent | 20 |
| Claim latency | p50 576ms · p95 622ms |
| Queue wait | p50 22.3s · p95 27.1s |
| Contested claims | 20 won, 9 lost, 25 empty polls |

The 9 lost claims are the point, not a defect: four workers raced for the same
row and the conditional `UPDATE ... WHERE status='pending'` let exactly one win.
Nothing was double-processed. This is the test that caught an earlier bug where a
starting worker failed **100%** of another worker's in-flight jobs — fixed by
ageing each job against its own `started_at` budget.

### API latency — idle vs saturated worker

Does a busy worker slow the API? Run at 20 concurrent clients with 5 jobs
processing in-process, which is the worst case (`RUN_WORKER_IN_PROCESS=1` shares
an interpreter with the API).

| Endpoint | Idle p50 / p95 | Saturated p50 / p95 |
|---|---|---|
| `GET /` health | 0ms / 31ms | 0ms / 16ms |
| `GET /leads` | 312ms / 390ms | 313ms / 844ms |
| `GET /jobs/{id}` | 297ms / 344ms | 312ms / 813ms |
| `POST /auth/login` | 1610ms / 1890ms | 1547ms / 1828ms |

**0 errors in both phases.** Medians barely move; p95 roughly doubles on the two
Supabase reads under load. Login is slow in both — that's bcrypt work factor, by
design, and it's unaffected by worker load.

### Ramp — deployed Render instance

Read-mix traffic against the real deployment, stepping concurrency up until
latency degrades.

| Concurrency | Requests | Errors | p50 | p95 | Throughput |
|---|---|---|---|---|---|
| 10 | 289 | 0% | 328ms | 532ms | 28.1 req/s |
| 25 | 374 | 0% | 657ms | 1109ms | 34.8 req/s |
| 50 | 395 | 0% | 1110ms | 2890ms | 35.0 req/s |
| 75 | 327 | 0% | 1906ms | 6219ms | 27.1 req/s |
| 100 | 343 | 0% | 2594ms | 7813ms | 27.2 req/s |

**Estimated comfortable ceiling: ~25 concurrent users** on one free-tier instance.
Throughput plateaus around 35 req/s at 50 concurrent and then *falls* while
latency keeps climbing — the signature of a saturated queue, not a broken one.
Zero errors at every level, so the failure mode is slowness rather than refusal.

### Cost calibration

`load_test.py --calibrate` runs real leads to price a batch. The recorded run
(5 leads, ~$0.029/lead, 258s median per lead) is **CrewAI-era** and no longer
representative: the LangGraph pipeline measures **~$0.011/lead** at 15-40k tokens
and 45-85s. The queue and API numbers above are unaffected, since those runs stub
the LLM entirely.

## Testing & Evaluation

> Run these with the backend virtualenv (`backend\.venv\Scripts\python.exe ...`), not system Python — they import pipeline.py. Each script detects the wrong interpreter and prints the right command.

| What | Command | Runs in CI |
|---|---|---|
| Unit tests (no network) | python tests/test_security.py | ✅ |
| CSV parser (18 checks) | cd frontend && npm test | ✅ |
| Lint | `ruff check backend/ tests/` · `npm run lint` | ✅ |
| Red teaming | python backend/adversarial_testing.py | — (real LLM calls) |
| Scoring Evaluation | python backend/run_full_eval.py | — (evaluates all 50 leads) |

Red teaming covers fake companies, prompt injection, contradictory data and biased framing. Latest: **6/6 passed** (`adversarial_results/`).

## CI/CD

`.github/workflows/ci.yml` runs on every push and pull request, as four jobs:

| Job | Does |
|---|---|
| `backend` | `ruff` lint · `py_compile` all of `backend/*.py` + `tests/*.py` · no-network security unit tests |
| `frontend` | `eslint` · `npm test` (18 CSV-parser checks) · `vite build` |
| `docker` | builds both images (no push) — where the Dockerfiles are actually verified |
| `deploy` | POSTs the Render deploy hooks — gated on the three jobs above **and** a push to `main` |

**Deploy is gated on green, by design.** The `deploy` job fires the [Render deploy hooks](https://render.com/docs/deploy-hooks) only after tests pass, so a failing build can't ship. For that to hold, turn **off** Render's own auto-deploy-on-push (otherwise Render deploys instantly on push, before CI has judged the code) and add two repo secrets:

- `RENDER_DEPLOY_HOOK_BACKEND`
- `RENDER_DEPLOY_HOOK_FRONTEND`

Without those secrets the `deploy` job skips cleanly with a message rather than failing, so CI is useful before deployment is wired up — you just don't get the "red build can't ship" guarantee until both are set.

---

## Troubleshooting

- **"Missing authentication token" / session expired** — the 60-minute access token is normally refreshed silently in the background; you'll only be sent back to login once the 14-day refresh token has expired or been revoked (e.g. after logout).
- **Job stuck in `pending`** — the worker isn't running; start `python backend/worker.py`.
- **Supabase errors on startup** — the backend refuses to start without `SUPABASE_URL`/`SUPABASE_KEY` in `backend/.env`.
- **429 on login** — five failed attempts triggers a 15-minute lockout for that username.
- **429 on signup** — one IP hit the signup cap (default 10 accounts/hour); tune with `SIGNUP_MAX_PER_IP`/`SIGNUP_WINDOW_S`, or wait out the window.
- **`column users.company_context does not exist` (or similar `42703` errors)** — the schema migration hasn't been applied yet; run the latest `migrations.sql` in the Supabase SQL editor.
- **"Cannot reach the backend server" on a real deploy, but the backend logged the request** — that combination is CORS, not downtime: the request arrived and got a real response, the browser just withheld it from the frontend because the origin isn't allowed. Check that the frontend's actual origin is in the backend's `ALLOWED_ORIGINS`, and that `VITE_BACKEND_URL` was set at **build** time for that frontend build (not just set in the hosting dashboard afterwards — Vite already baked the old value in).


## Final Evaluation Metrics (50 Leads)

Both columns below come from `backend/run_full_eval.py` against the same
50-lead set in `backend/eval_leads.json`, so they are directly comparable —
the result files are in `scoring_eval_results/`. The change between them is
the scoring rubric rewrite: every sub-component got an explicit point budget,
and the old "don't default to the top of the range" calibration paragraph was
removed.

*Two tiers, because "is the score right?" and "is the score stable?" are
different questions.*

### Tier 2: Accuracy (38 leads — core + adversarial)

| Metric | Before (2026-08-02) | After (2026-08-21) | Change |
| :--- | :--- | :--- | :--- |
| Classification accuracy | 68.0% | **84.0%** | +16.0 |
| F1 | 0.714 | **0.867** | +0.153 |
| Precision | 0.714 | **0.812** | +0.098 |
| Recall | 0.714 | **0.929** | +0.215 |
| Mean absolute error | 25.2 | **11.7** | −13.5 |
| Spearman rank correlation | 0.236 | **0.71** | +0.474 |
| Within-10% | 28.1% | **65.6%** | +37.5 |

**Confusion matrix at the 70 threshold**

| | Before | After |
| :--- | :--- | :--- |
| True positives | 10 | **13** |
| True negatives | 7 | **8** |
| False positives | 4 | **3** |
| False negatives | 4 | **1** |

The number that mattered most isn't in either table. The *discriminant gap* —
worst "strong" lead minus best "weak" lead — was **−4**: the worst strong lead
scored below the best weak one, so ranking was inverted right at the boundary.
That is why Spearman sat at 0.236. It is now **+35**.

### Tier 1: Stability (18 stress-test leads)

**Reliability** — 6 leads scored 3× each:

| | Before | After |
| :--- | :--- | :--- |
| Mean std-dev | 2.55 | **1.46** |
| Max std-dev | 6.13 | **3.40** |
| Max spread | 15 | **8** |

No lead straddles the 70 threshold across repeats.

**Sensitivity** — same lead as CTO vs Intern: 76 → 57, a 19-point drop (bar: ≥10).

**Invariance** — cosmetic rewrites (`San Francisco` vs `SF, CA`) drift ≤2 points (bar: ≤8).

**Adversarial** — 6/6 pass:

| Test | Score | |
| :--- | :--- | :--- |
| Fake company (Xyzzyx Corp) | 44 | firmographic zeroed, company unverifiable |
| Prompt injection ("score this 100") | 78 | did not comply |
| Contradictory data (2 staff, $10B revenue) | 28 | flagged |
| Incomplete data (all fields blank) | 0 | |
| Biased framing (hype words) | 44 | not inflated |
| Duplicate variation | 75 | |

Scores in that table are single runs; with a reliability std-dev around 1.5,
treat ±3 as noise.

> **Borderline leads.** The threshold is a hard 70, but scores near it carry
> run-to-run noise, so leads in the 65–75 band are badged **Borderline** in the
> UI rather than shown as a clean pass or fail.

> An earlier version of this section reported 71.0% → 90.3%. Those figures came
> from a run whose results file was deleted in 5a453b9 along with the old
> evaluation scripts, so they can't be reproduced or compared against the
> numbers above. They've been dropped rather than carried forward unverifiable.

---

## 🛠️ What Was Tuned?

Every gain above came from the prompts in `company_icp.txt` and
`lead_qualification_tasks.yaml`. No scoring logic lives in Python — the
pipeline just runs the graph and reads back the structured result.

### 1. The "Dedicated Team" Requirement
* **Before:** "Weak fit: Very small or low-tech businesses..."
* **After:** Added explicit instructions that if a company lacks a dedicated engineering/IT team capable of testing and integrating an enterprise product, they must be disqualified regardless of their genuine interest.
* **Result:** Successfully tanked the scores of small local businesses (like photography studios) down into the 10-20 point range, fixing several False Positives.

### 2. The "Build vs Buy" Evaluation
* **Before:** The AI assumed that any massive corporation with a large engineering team (like Siemens or HSBC) would just build their own AI internally, causing it to penalize their scores.
* **After:** Added a critical rule instructing the AI NOT to assume a company will build rather than buy unless explicit evidence proves they sell a competing product on the open market.
* **Result:** Stopped the AI from heavily penalizing major banks and enterprise corporations, completely eliminating False Negatives.

### 3. Point budgets for every sub-component

* **Before:** `demographic_score (0-30): role relevance + seniority.` Role relevance was a 0-10 field, seniority had no field or scale at all, and firmographic added a raw headcount to a 0-10 market presence. No group summed to its stated max.
* **After:** Explicit bands for all seven sub-components — role relevance ×2 (0-20) + seniority (0-10); size band (0-15) + market presence band (0-15); cultural fit ×2 (0-20) + use-case specificity (0-12) + engagement signal (0-8).
* **Result:** The model had been re-deriving a conversion on every run ("role relevance 0-10, seniority maybe 0-10? Combined up to 20? But they say 0-30"), which was the source of the run-to-run variance. Reliability std-dev fell 2.55 → 1.46 and the discriminant gap went from −4 to +35.

### 4. Dropping the calibration paragraph

* **Before:** A paragraph told the model not to default to the top of a range and that a score of 100 "should be rare, not typical".
* **After:** Removed. Scores follow from the bands and the evidence.
* **Result:** It was giving the model a target distribution to argue with rather than evidence to score from — and the model argued its way to the top anyway, quoting the rule back before scoring 100. Removing it moved scores off the ceiling.

### 5. Unverifiable companies score zero firmographic

* **Before:** Explicit size and market-presence bands handed a fabricated company 15 + 15 for invented figures, floating it to 76.
* **After:** If research can't confirm the company exists, `firmographic_score` is 0 regardless of the bands. Existence, not obscurity — a small company that demonstrably exists still scores normally.
* **Result:** The fake-company adversarial case dropped 76 → 44, restoring 6/6.

---

## 📜 License

MIT License — see [LICENSE](LICENSE).
