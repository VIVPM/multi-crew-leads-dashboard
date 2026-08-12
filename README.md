# Sales Pipeline — Lead Scoring & Email Generation

A full-stack, multi-agent sales pipeline application: a **React** dashboard backed by a **FastAPI** server that orchestrates **CrewAI** agent crews (powered by **Google Gemini**) to score sales leads and draft personalized outreach emails, with all data stored in **Supabase**.

---

## Features

- **Landing page** — Stripe-inspired marketing page with an animated product demo: the agent pipeline lights up in sequence, the score counts up, and the email draft types itself.
- **React dashboard** — a menu (top-right) switches between pages: **Add a lead**, **Dashboard** (charts — industry, source, score distribution, leads over time), **Lead details** (search, edit, delete, export; per-lead analysis modal with token/cost/timing breakdowns), and **Settings** (company & ICP, email sending). The menu also shows the signed-in account and Log out.
- **Required ICP** — your company profile & ideal customer profile live in a main-dashboard card and are required, not optional: processing a lead with none set pops a blocking dialog until you fill it in. No generic fallback exists. The placeholder guides you to write not just a strong-fit ICP but explicit **Weak fit** and **Not a fit** lines — spelling those out is what the scoring evaluation showed makes the score trustworthy.
- **Four-agent, three-crew pipeline** — a `company` crew (Company Research & Cultural Fit, cacheable), a `personal_scoring` crew (Personal Research → Lead Scoring & Validation), and an `email` crew (Email Specialist, draft + optimize) for leads scoring above 70.
- **Company research caching** — company facts + cultural fit are cached per (company, ICP) pair for repeat leads from the same company, with a short TTL (company status can change) and a **Force refresh** checkbox on the lead form to bypass it.
- **Web-search enrichment** — agents research leads live via **Tavily** search and website scraping.
- **Asynchronous job queue** — processing runs in a background worker; the API responds instantly and the UI polls job status, so long LLM runs never block requests.
- **Live per-agent progress** — while a lead is processing, a progress bar and step list below the form show which agent is working: Company research & cultural fit → Personal research → Lead scoring → Email draft. Each step ticks over as that agent's task actually completes (the worker publishes stages to `jobs.progress`), so it reflects real pipeline state rather than an animated guess — including **cached** when company research was skipped.
- **Token-based auth** — signup/login with bcrypt password hashing (signing up logs you straight in, no second login step); a short-lived 60-minute access token plus a revocable 14-day refresh token (stored server-side as a hash) that the frontend renews silently, so a session lasts without a hard hourly logout and logout revokes it server-side; every lead endpoint is ownership-checked; login is rate-limited (5 failed attempts → 15-minute lockout) and signups are capped per IP (10 accounts/hour) so nobody farms daily credits by scripting throwaway accounts.
- **Operator-held API keys** — Gemini and Tavily keys are supplied once by the operator in `backend/.env`; end users never enter keys.
- **Daily lead credits** — since the keys are operator-held, every processed lead is the operator's LLM spend on one shared quota, so each account gets a daily allowance set by `DAILY_LEAD_CAP` (1 credit = 1 lead). It's required with no default: a deploy that forgets it fails at startup rather than silently running on a guessed spend limit. The dashboard shows credits remaining, over-limit requests are rejected before any spend, and the count resets at UTC midnight — no credits table or reset job, since "remaining" is just `cap − leads-processed-today`.
- **Editable & sendable email drafts** — click **Edit** on a lead's drafted email to revise it inline, or **Send** to deliver it through your own connected email account (per-user SMTP settings — a Gmail App Password or any SMTP provider; a daily cap guards the account).
- **Structured JSON logging** — every log line is correlated by `request_id`/`job_id`/`lead_id`, so one request's or job's full story is a single grep away.
- **Observability** — optional OpenTelemetry tracing of every crew/agent/LLM call via targeted openinference instrumentors, exported to Langfuse and/or Grafana Cloud (which also receives a `jobs_processed_total` metric powering no-activity/failure alerts), enabled automatically when the matching env vars are set.
- **YAML-driven agents** — all agent roles, task prompts, and workflow logic configurable in `backend/config/` without code changes.
- **Bulk CSV import** — add leads one at a time or upload a CSV; rows are validated per line and duplicates (by email) are skipped. An import must fit the day's remaining credits: uploading more leads than credits left is blocked upfront with a "upload at most N" message, rather than partially processing. During a run each lead is scored one at a time with a live per-lead list (queued / scoring… / scored / failed), so the team watches them complete one after another.
- **Borderline scores are flagged, not hidden** — repeat scoring of the same lead varies by ~±3.5 points, so leads landing within 65–75 of the 70 email cutoff are badged **Borderline** with an explanation, rather than silently drafting (or not) on a coin flip.
- **Red-team test suite** — adversarial inputs (fake companies, prompt injection, contradictory data) with saved pass/fail reports.
- **Scoring evaluation harness** — reliability checks (repeatability, discrimination, seniority sensitivity, invariance) plus a human gold-set comparison for accuracy.

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

    subgraph AI ["4 · Reasoning layer — CrewAI · 4 agents"]
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
    CTRL -->|run crews| AI
    AI -->|research + reasoning| EXT
    CTRL -->|read cache · write results| DATA
    CTRL -.->|traces · metrics| OBS
```

Three separate `Crew` objects (not one monolith), because `company` needs to be independently skippable on a cache hit:

| Crew | Agents | Runs when |
|---|---|---|
| `company` | Company Research & Cultural Fit Analyst (A2) | always, unless a fresh cache entry exists for `(company, ICP)` |
| `personal_scoring` | Personal Research Specialist (A1) → Lead Scorer & Validator (A3) | always |
| `email` | Email Specialist (E1) | only if the score from `personal_scoring` is > 70 |

## Project Structure

```
.
├── backend/
│   ├── backend.py            # FastAPI app: auth, lead CRUD, company profile, job enqueue/status
│   ├── worker.py             # Background job processor (runs the crews, company-research cache)
│   ├── pipeline.py           # CrewAI crews + process_leads entry point (no Supabase dependency)
│   ├── security.py           # bcrypt hashing + access/refresh session tokens
│   ├── logging_setup.py      # structured JSON logs + request/job/lead correlation IDs
│   ├── adversarial_testing.py # red-team test suite (needs backend/.venv — see below)
│   ├── scoring_eval.py       # Tier 1: scoring reliability (repeatability, sensitivity)
│   ├── scoring_gold_set.py   # Tier 2: agent vs human-scored gold set (accuracy)
│   ├── load_test.py          # worker drain + multi-worker safety (LLM stubbed); --calibrate for real cost
│   ├── load_test_api.py      # API latency, idle vs saturated worker (LLM stubbed)
│   ├── test_crews.py         # crew smoke test + CrewAI eval runs (needs backend/.venv — see below)
│   ├── Dockerfile            # one image, two commands (API via uvicorn, worker via python)
│   ├── requirements.txt      # pinned Python dependencies
│   └── config/               # agent & task YAML definitions (all three crews)
├── frontend/                 # React + Vite dashboard + landing page (Stripe-inspired design system)
│   ├── src/csv.js            # CSV import parser (bulk import)
│   ├── src/csv.test.js       # 18 node:test checks for the parser (run in CI)
│   ├── Dockerfile            # node build → nginx static serve
│   └── nginx.conf            # SPA fallback + fingerprinted-asset caching
├── tests/test_security.py    # no-network unit tests (run in CI)
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
GEMINI_API_KEY=your_gemini_key      # operator-held, powers all agents (required)
TAVILY_API_KEY=your_tavily_key      # operator-held, web-search enrichment (required)
DAILY_LEAD_CAP=5                    # leads each account may process per day (required)
ALLOWED_ORIGINS=https://your-frontend.example.com,http://localhost:5173   # CORS allowlist (required, comma-separated)

# Optional — run the job worker inside the API process (single-service deploys)
RUN_WORKER_IN_PROCESS=1

# Optional — OpenTelemetry tracing of every crew/agent/LLM call, exported to
# Langfuse and/or Grafana Cloud (Grafana also gets the jobs_processed_total metric)
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
GRAFANA_OTLP_ENDPOINT=
GRAFANA_OTLP_AUTH=

# Optional — daily per-user email send cap (default 80)
EMAIL_SEND_DAILY_CAP=80
```

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
4. **Add a lead** (name, company, email required) and click **Save & Process** — the lead is queued, the crews score it and (if it scores above 70) draft an email. A progress bar and step list appear below the form and advance as each agent finishes, so you can see which one is working. Check **Force refresh** first to bypass the company-research cache for that batch.
5. **Review results** — expand a lead card for the scoring breakdown and email draft; click **Edit** to revise the draft, or **Send** to email it (once your email account is connected). Open **📊 Analysis** for duration, token, and cost details (tokens are measured per crew: the single-agent company and email crews are exact, while the scoring crew's two agents split its total, so those two rows are estimates; a row marked **Cached** means company research was served from cache, **Skipped** means the email crew never ran because the score was ≤ 70 — both always listed at 0, so the breakdown always shows all 4 agents).
6. **Search / export** — filter the table and export the filtered set to CSV.

Processing is capped per account per day by `DAILY_LEAD_CAP` (the daily credit allowance); job status is `pending → running → done | failed`.

---

## Scaling notes

- Lead processing is decoupled from HTTP via the `jobs` table. Each `worker.py` process runs up to `MAX_CONCURRENT_JOBS` (default 10) jobs concurrently via `asyncio` — unrelated users' jobs run in parallel, not queued behind each other one at a time. Add more `worker.py` processes to raise the ceiling further; running several is safe (load-tested), since a starting worker ages each `running` job against its own time budget instead of assuming everything in flight is abandoned. Requires the `started_at` column from `migrations.sql`.
- Access tokens are stateless (HMAC-signed) and refresh tokens live in the shared `refresh_tokens` table, so any API instance can serve or refresh any session — instances scale horizontally behind a load balancer; set the same `SECRET_KEY` on every one.
- Both rate limiters are Supabase-backed, not in-memory, so they hold correctly across multiple API instances: login (`login_failures`, per username) and signup (`signup_attempts`, per IP). The signup cap (`SIGNUP_MAX_PER_IP`/`SIGNUP_WINDOW_S`, default 10/hour) counts one row per created account, so a spammer can't mint accounts to farm each one's daily credits.
- Company research (facts + cultural fit) is cached per `(company, ICP)` pair — `COMPANY_CACHE_TTL_DAYS` (default 7) bounds how long a company shutting down / getting acquired could go undetected; `ProcessLeadsRequest.force_refresh` bypasses the cache for a batch, exposed in the UI as a **Force refresh** checkbox on the lead form.
- Concurrent requests for the same brand-new company don't duplicate the research: the first cache miss atomically claims the row (`company_key` is unique), so a second concurrent miss waits on the winner instead of re-running Tavily + Gemini itself.
- `GET /leads` returns only the columns the list renders. `scoring_result` and `email_draft` are ~78% of a lead row but are only shown once a card is expanded, so they come from `GET /leads/{lead_id}/detail` on demand — measured **~100KB → 15.8KB** for a 50-lead account on the most-requested endpoint. Trade-off: search no longer matches text *inside* drafts or scoring JSON (server-side search would be the fix if that's ever wanted).
- Every route in `backend.py` is a sync `def`, so FastAPI runs each in a worker thread — anyio caps that pool at 40 by default, which is the first ceiling under load. The two hottest read endpoints (`GET /leads`, `GET /jobs/{id}`) use a genuinely async Supabase client instead, so they suspend on I/O rather than holding a thread. Converting more routes means converting *every* blocking call inside them too; a half-converted `async def` blocks the whole event loop, which is worse than leaving it sync.

---


## Testing & Evaluation

> Run these with the backend virtualenv (ackend\.venv\Scripts\python.exe ...), not system Python — they import pipeline.py. Each script detects the wrong interpreter and prints the right command.

| What | Command | Runs in CI |
|---|---|---|
| Unit tests (no network) | python tests/test_security.py | ✅ |
| CSV parser (18 checks) | cd frontend && npm test | ✅ |
| Lint | 
uff check backend/ tests/ · 
pm run lint | ✅ |
| Red teaming | python backend/adversarial_testing.py | — (real LLM calls) |
| Scoring Evaluation | python backend/run_full_eval.py | — (evaluates all 50 leads) |

Red teaming covers fake companies, prompt injection, contradictory data and biased framing. Latest: **6/6 passed** (dversarial_results/).

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
- **"Cannot reach the backend server" on a real deploy, backend logs show `403`, not silence** — that combination means CORS, not downtime: the request reached the server and got a real response, but the browser withheld it from the frontend because the response had no `Access-Control-Allow-Origin` header. Check that the frontend's actual origin is in the backend's `ALLOWED_ORIGINS`, and that `VITE_BACKEND_URL` was set at **build** time for that frontend build (not just set in the hosting dashboard after the fact — Vite already baked the old value in).


## Final Evaluation Metrics (50 Leads)

# Prompt Tuning Evaluation Comparison

This document provides a side-by-side comparison of the CrewAI lead scoring system's performance before and after prompt tuning. 

*Note on Tiers: We separate testing into two tiers because "is the score right?" and "is the score stable?" are different questions.*

## 🎯 Tier 2: Accuracy Metrics (38 Leads - Includes Core + Adversarial)
*Tier 2 tests whether the score is **right** by comparing it against human gold-set scores.*

| Metric | Before Tuning (Baseline) | After Tuning (Optimized) | Change |
| :--- | :--- | :--- | :--- |
| **Classification Accuracy** | 71.0% | **90.3%** | 🟢 +19.3% |
| **F1 Score** | 0.710 | **0.903** | 🟢 +0.193 |
| **Precision** | 0.688 | **0.824** | 🟢 +0.136 |
| **Recall** | 0.733 | **1.000** | 🟢 +0.267 |
| **Regional Mean Absolute Error (MAE)** | 22.9 | **16.6** | 🟢 -6.3 |
| **Spearman Rank Correlation** | 0.386 | **0.806** | 🟢 +0.420 |
| **Within-10% Accuracy** | 26.3% | **39.5%** | 🟢 +13.2% |

## 📊 Confusion Matrix (Threshold = 70)

| Category | Before Tuning | After Tuning | Impact |
| :--- | :--- | :--- | :--- |
| **True Positives (TP)** | 11 | **14** | Found 3 more high-quality leads. |
| **True Negatives (TN)** | 11 | **14** | Correctly disqualified 3 more weak/fake leads. |
| **False Positives (FP)** | 5 | **3** | Handled tricky framing better. |
| **False Negatives (FN)** | 4 | **0** | **Eliminated all missed opportunities!** |

> **Borderline Leads:** The AI operates on a hard threshold of 70. However, scores between 65-75 have structural run-to-run noise (a ~±3.5 point variance could swing a 72 to a 68 on a re-run). To handle this, 7 leads in our dataset fall into this edge-case range and are gracefully badged as **Borderline** in the UI rather than purely Pass/Fail.

---

## 🛡️ Tier 1 & Adversarial Metrics (18 Stress-Test Leads)
*Tier 1 tests whether the score is **stable** by checking repeatability, variance, and sensitivity to cosmetic changes.*

The remaining 18 leads were specifically crafted to test logic traps, hallucinations, and reliability. Because they are edge cases, they are scored separately from the core accuracy metrics above.

### 1. Reliability (6 Leads run 3x each)
Tests if the AI gives the exact same score for the exact same lead across multiple runs.
* **Before Tuning:** `rel_04` **FAILED** with a massive 15-point variance (scores swung wildly between 76 and 91 because the AI was just guessing without strict rules).
* **After Tuning:** **ALL PASS**. The strict, evidence-based rules caused the maximum variance across all 6 leads to tighten to just **4.03 points**!

### 2. Sensitivity (2 Leads)
Tests if the AI properly identifies seniority by comparing an identical lead pitched by a 'CTO' vs an 'Intern'.
* **Before Tuning:** CTO scored 93, Intern scored 71 (Drop of 22 points). **PASS**
* **After Tuning:** CTO scored 89, Intern scored 71 (Drop of 18 points). **PASS**

### 3. Invariance (4 Leads)
Tests if purely cosmetic changes (like 'San Francisco' vs 'SF, CA' or capitalizing different letters) confuses the AI.
* **Before Tuning:** Score drifted by up to **4 points**. **PASS**
* **After Tuning:** Score drifted by only **1 point**. **PASS**

### 4. Adversarial Edge Cases (6 Leads)
Tests if the AI can be tricked by fake data, prompt injections, or hype words.

| Adversarial Test | Before Tuning (Score) | After Tuning (Score) | Status |
| :--- | :--- | :--- | :--- |
| **Fake Company** (Xyzzyx Corp) | 60 | **45** | Both Pass (Significantly better after tuning) |
| **Prompt Injection** (Hack to 100) | 55 | **62** | Both Pass |
| **Contradictory Data** (Enterprise + 2 employees) | 32 | **35** | Both Pass |
| **Incomplete Data** (Blank fields) | 0 | **0** | Both Pass |
| **Biased Framing** (Extreme hype words) | 74 | **78** | Both Pass |
| **Duplicate Variation** | 85 | **82** | Both Pass |

As you can see, the tuning not only fixed the classification accuracy of the 32 normal leads, but it also completely stabilized the reliability and cosmetic invariance of the 18 stress-test leads!

---

## 🛠️ What Was Tuned?

The massive improvement in performance was driven entirely by modifying the LLM's instructions in `company_icp.txt` and `lead_qualification_tasks.yaml`.

### 1. The "Dedicated Team" Requirement
* **Before:** "Weak fit: Very small or low-tech businesses..."
* **After:** Added explicit instructions that if a company lacks a dedicated engineering/IT team capable of testing and integrating an enterprise product, they must be disqualified regardless of their genuine interest.
* **Result:** Successfully tanked the scores of small local businesses (like photography studios) down into the 10-20 point range, fixing several False Positives.

### 2. The "Build vs Buy" Evaluation
* **Before:** The AI assumed that any massive corporation with a large engineering team (like Siemens or HSBC) would just build their own AI internally, causing it to penalize their scores.
* **After:** Added a critical rule instructing the AI NOT to assume a company will build rather than buy unless explicit evidence proves they sell a competing product on the open market.
* **Result:** Stopped the AI from heavily penalizing major banks and enterprise corporations, completely eliminating False Negatives.

---

## 📜 License

MIT License — see [LICENSE](LICENSE).
