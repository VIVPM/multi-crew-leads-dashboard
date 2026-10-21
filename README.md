# Sales Pipeline — Lead Scoring & Email Generation

Multi-agent sales pipeline: **React** dashboard → **FastAPI** → **LangGraph** agents (Google Gemini or Cloudflare Workers AI) → **Supabase**. The graph researches and scores leads; those above 70 get a personalized outreach draft.

---

## Features

- **LangGraph pipeline** — company and personal research in parallel → scoring (seven judge calls at once) → email for leads above 70
- **Async job queue** — `POST /leads/process` returns `202`; workers pull jobs and the UI shows live per-agent progress
- **Reliable execution** — idempotent submissions, tenant-fair claims, transport-only retries, graceful shutdown and a circuit breaker
- **Bounded spend** — operator keys with a required `DAILY_LEAD_CAP`; users can bring their own Gemini or Cloudflare credentials plus a Tavily key to run without the cap
- **Company cache** — keyed by `(company, ICP)` with a TTL, atomic claim and force refresh
- **Secure accounts** — bcrypt, 60-minute access tokens, rotating 14-day refresh tokens, ownership checks, login/signup rate limits; saved keys and SMTP passwords are encrypted
- **Dashboard** — KPIs, cost/token charts, search, CSV import/export, analysis detail, editable drafts and SMTP sending; scores 65–75 are flagged **Borderline** with an on-demand **Draft Email** button
- **Audit trail** — writes, job submits, email sends, settings changes and logins go to `audit_events` with the acting user from the token; no bodies or secrets
- **Observability and evaluation** — correlated JSON logs, optional Langfuse traces, red-team tests and a 60-lead golden set

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

    subgraph CTRL ["3 · Workers — worker.py"]
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

| Node | Runtime | Runs when |
|---|---|---|
| `company` | `create_react_agent` + Tavily/scrape | unless a fresh `(company, ICP)` cache entry exists |
| `personal_research` | `create_react_agent` + Tavily/scrape | always, alongside `company` |
| `scoring` | one extraction call + seven parallel judge calls | always |
| `email` | model call | only when score > 70 |

Time per stage for one strong-fit lead (score 92), taken from its Langfuse trace:

| Stage | Time (s) |
|---|---:|
| `company` | 7.6 |
| `personal_research` | 7.0 |
| `scoring` | 9.5 |
| `email` | 10.3 |
| **Total** | **27.4** |

The two research nodes run in parallel, so the research stage takes as long as the slower one. Leads scoring 70 or below skip `email`.

## Project Structure

```text
.
├── backend/
│   ├── backend.py             # FastAPI: auth, leads, settings, job queue
│   ├── worker.py              # job claims, slots, company cache
│   ├── queue_policy.py        # round-robin claims and slot autoscaling
│   ├── pipeline.py            # LangGraph graph and weighted scoring
│   ├── security.py            # bcrypt, tokens, encrypted credentials
│   ├── logging_setup.py       # JSON logs and correlation IDs
│   ├── adversarial_testing.py # red-team suite
│   ├── run_golden_eval.py     # golden-set evaluation
│   ├── golden_leads.json      # 60 leads with gold points per attribute
│   ├── run_real_agent_locust.py # real-provider Locust runner
│   ├── real_agent_locust.py     # Locust virtual-user flow
│   └── config/                # agent and task prompt YAML
├── frontend/                  # React + Vite dashboard
├── tests/                     # security, settings, dashboard, pipeline, queue, persistence
├── results/                   # golden eval and load-test reports
├── .github/workflows/ci.yml
└── docker-compose.yml
```

---

## Setup

### 1. Supabase

Create a project at [supabase.com](https://supabase.com) and run the local, gitignored `migrations.sql` in the SQL editor. It's idempotent; rerun it after schema changes. The backend uses a service key and checks ownership in the API; RLS is recommended as a second layer.

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

LLM_MODEL=GEMINI                     # GEMINI or CLOUDFLARE
GEMINI_API_KEY=your_gemini_key       # for GEMINI
CLOUDFLARE_ACCOUNT_ID=               # for CLOUDFLARE, or JUDGE_PROVIDER=CLOUDFLARE
CLOUDFLARE_API_TOKEN=
JUDGE_PROVIDER=HUGGING_FACE          # HUGGING_FACE or CLOUDFLARE
HUGGINGFACE_API_KEY=                 # for HUGGING_FACE
TAVILY_API_KEY=your_tavily_key

DAILY_LEAD_CAP=5
ALLOWED_ORIGINS=https://your-frontend.example.com,http://localhost:5173
RUN_WORKER_IN_PROCESS=1              # 1 runs the worker inside the API; 0 for separate workers

LANGFUSE_PUBLIC_KEY=                 # optional tracing
LANGFUSE_SECRET_KEY=
LANGFUSE_HOST=
```

Run from the repository root:

```bash
uvicorn backend.backend:app --host 0.0.0.0 --port 8000
python backend/worker.py             # only when RUN_WORKER_IN_PROCESS=0; run as many as you need
```

Any number of workers can run against one database: a conditional update lets exactly one claim each job.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

Set `VITE_BACKEND_URL=http://localhost:8000` in `frontend/.env`. Vite inlines it, so production builds need the real API URL at build time.

### 4. Docker (optional)

```bash
docker compose up --build              # frontend → :5173, API → :8000
docker compose up --scale worker=3     # more workers
```

### 5. API keys

Operator keys in `backend/.env` work by default. Under **Settings → API keys** a user can choose **Gemini** (API key) or **Cloudflare Workers AI** (account ID and API token), with Tavily as a separate field. Complete credentials for the chosen provider run that user's jobs on their own account; with a Tavily key too, the daily cap is lifted. The scoring judge always uses the operator's key. Keys aren't validated on save, and changing `SECRET_KEY` makes saved keys unreadable.

### Cloudflare Workers AI

`gpt-oss-20b` needs a few workarounds, all inside the model wrapper: a 4096-token reply limit (the default `CLOUDFLARE_MAX_TOKENS`) so reasoning doesn't use up the reply, the JSON schema in the prompt because native structured output fails, `content: null` coerced to `""`, and streaming off because it double-counts tokens. Gemini and Workers AI can score the same lead differently.

## How to Use

1. Sign up, then save your company profile and ICP. Processing is blocked until it exists, since it anchors cultural fit and the email.
2. Optionally add your own keys under **Settings → API keys** and SMTP details under **Email sending**.
3. Add a lead and select **Save & Process**. The progress panel follows each graph node; **Force refresh** bypasses cached company research.
4. Review the score breakdown and draft, open **Analysis** for per-node time, tokens and cost, then edit or send the email.

Jobs move `pending → running → done | failed`. Queued and successful leads use daily credits; a failed job gives them back.

## Scaling Notes

- **Workers pull jobs.** Each worker checks the queue every 3 s and claims the oldest waiting job across users in turn while it has a free slot.
- **Slots autoscale.** A worker runs between `WORKER_MIN_CONCURRENCY` (2) and `MAX_CONCURRENT_JOBS` jobs; at each queue check, waiting jobs can double its slots. After `SCALE_COOLDOWN_S` with an empty queue, slots halve without cancelling running jobs.
- **More workers** raise the job ceiling, but actual throughput also depends on provider quotas and database capacity.
- **Stateless API** — HMAC access tokens and Supabase-stored refresh tokens, so API copies can scale behind a load balancer with a shared `SECRET_KEY`.
- **Supabase connection pool** — both clients share `SUPABASE_POOL` (200 connections, 100 kept alive); httpx's default 100/20 ran out under stress.
- **Request limits** — Supabase enforces them across API instances: signup 5 attempts/minute per username and 30/minute in total; login 30 requests/minute per username plus 5 failed attempts/15 minutes; token refresh 120 requests/minute per refresh token; lead processing 30 leads/minute/user, including user-funded runs; on-demand drafts 5/minute/user. Rejections return `429` with `Retry-After`. Apply the new block in the local `migrations.sql` before deploying the API or these routes return `503`.
- **Tavily search pacing** — one worker process reserves at most 90 searches per minute per API key (`TAVILY_RPM_LIMIT`, default 90), leaving room below a development key's 100 RPM cap. Searches wait for capacity; an explicit rate rejection waits 60 s and retries once, while credit exhaustion fails immediately. Use a higher setting only after confirming the key's limit. Multiple worker processes do not share this in-memory budget.
- **Polling backs off** from 3 s to 10 s while a job runs.

## Real-Agent Load Testing

`backend/run_real_agent_locust.py` runs [Locust](https://locust.io) against a local copy of the real API, Supabase and worker, with real calls to Gemini, Tavily and the Hugging Face judge — this costs money. Each virtual user logs in, submits one lead with `force_refresh: true`, and polls `/jobs/{id}` to completion; `backend/real_agent_locust.py` defines that user flow. The runner starts each user level in order and stops after the first level with a request error, failed job or unfinished job. It deletes the tagged benchmark accounts, leads, jobs and cache rows afterward. The Tavily key comes from `backend/.env` (`TAVILY_API_KEY`); `--tavily-key-file` can supply a different key for one run without changing `.env`.

Suspend any deployed worker sharing the Supabase project before running this test; otherwise that worker may claim benchmark jobs.

```bash
python backend/run_real_agent_locust.py --levels 1,2,5,10,20,40,80,160,250
```

**1, 2, 5, 10 users** (production models, `MAX_CONCURRENT_JOBS=10`, 18 golden leads total):

| Users at once | Jobs done | Submit to result p50 / p95 / p99 (s) | Queue wait p50 / p95 / p99 (s) | Model cost (USD) |
|---:|---:|---|---|---:|
| 1  | 1/1   | 46.4 / 46.4 / 46.4 | 2.7 / 2.7 / 2.7   | 0.0336 |
| 2  | 2/2   | 41.1 / 50.6 / 50.6 | 2.1 / 2.4 / 2.4   | 0.0946 |
| 5  | 5/5   | 45.2 / 49.2 / 49.2 | 2.1 / 8.8 / 8.8   | 0.1952 |
| 10 | 10/10 | 35.5 / 51.5 / 51.5 | 7.6 / 12.4 / 12.4 | 0.2419 |

Total model cost: $0.57 for 18 leads (Tavily not included). No failed jobs, no request errors, no rate-limit 429s at any level.

**20, 40, 80 users** (`MAX_CONCURRENT_JOBS=40`). Both Gemini models were temporarily switched to `gemini-3.1-flash-lite` for these runs; the current code uses production models again. The 80-user level reuses the 60 golden leads with new names/emails for leads 61–80.

| Users at once | Jobs done | Submit to result p50 / p95 / p99 (s) | Queue wait p50 / p95 / p99 (s) |
|---:|---:|---|---|
| 1 (check) | 1/1   | 29.9 / 29.9 / 29.9    | 3.2 / 3.2 / 3.2       |
| 20        | 20/20 | 76.6 / 86.8 / 86.9    | 31.2 / 60.8 / 61.2    |
| 40        | 40/40 | 117.8 / 143.7 / 147.1 | 48.7 / 104.2 / 105.3  |
| 80        | 80/80 | 146.5 / 215.9 / 218.9 | 75.1 / 182.8 / 183.6  |

About 16K tokens/lead, about 2.1 Tavily searches/lead. No failed jobs, no request errors, no rate-limit 429s.

**160 users**, same model/slot setup, after moving the worker's Supabase calls off the asyncio event loop (synchronous calls were blocking it; moved to `asyncio.to_thread`):

<table>
<tr><th rowspan="2">Users</th><th rowspan="2">Done</th><th rowspan="2">Failed</th><th colspan="3">Submit to result (s)</th><th colspan="3">Queue wait (s)</th><th rowspan="2">Throughput (jobs/min)</th></tr>
<tr><th>p50</th><th>p95</th><th>p99</th><th>p50</th><th>p95</th><th>p99</th></tr>
<tr align="right"><td>160</td><td>160</td><td>0</td><td>154.5</td><td>317.9</td><td>322.2</td><td>131.8</td><td>296.1</td><td>301.4</td><td>28.7</td></tr>
</table>

Worker claim-to-done median/p95 was 19.0/28.0 s; the highest number of claimed, unfinished jobs was 14, below the 40-slot ceiling. One `GET /jobs/{id}` returned HTTP 500 from a Supabase 502; the job finished anyway.

**250 users**, same model/slot setup, after adding Tavily request pacing (90 requests/minute/key, below the Tavily development key's 100 RPM cap):

<table>
<tr><th rowspan="2">Users</th><th rowspan="2">Done</th><th rowspan="2">Failed</th><th colspan="3">Submit to result (s)</th><th colspan="3">Queue wait (s)</th><th rowspan="2">Throughput (jobs/min)</th></tr>
<tr><th>p50</th><th>p95</th><th>p99</th><th>p50</th><th>p95</th><th>p99</th></tr>
<tr align="right"><td>250</td><td>235</td><td>15</td><td>394.9</td><td>522.8</td><td>536.9</td><td>346.8</td><td>475.0</td><td>476.1</td><td>24.5</td></tr>
</table>

Zero Tavily failures occurred. The 15 failed jobs hit the Hugging Face/Scaleway judge's separate requests-per-minute quota (`429 INSUFFICIENT QUOTA`); that quota is not yet paced. Ten job-status polls returned HTTP 500 after Supabase timeouts, and one submission returned an idempotent replay (HTTP 200) that the harness counted as a client error although its job finished. These request errors are not additional failed jobs. Worker claim-to-done: median/p95/p99 22.0/52.0/78.0 s; peak claimed-unfinished jobs 35, below the 40-slot ceiling. Analysis records for 234 completed jobs estimated $4.29 in model cost; the estimate uses production-model rates rather than the `gemini-3.1-flash-lite` model used for this test, so it is not the billed amount. No 500-user real-agent run has been attempted.

## Testing & CI/CD

| What | Command | CI |
|---|---|---|
| Unit tests | `python tests/test_security.py` · `test_settings.py` · `test_rate_limits.py` · `test_dashboard.py` | ✅ |
| Graph wiring | `python tests/test_pipeline.py` | ✅ |
| Claim fairness | `python tests/test_queue_policy.py` | ✅ |
| CSV parser | `cd frontend && npm test` | ✅ |
| Persistence | `python tests/test_persist_results.py` | needs Supabase |
| Lint | `ruff check backend/ tests/` · `npm run lint` | ✅ |
| Red team | `python backend/adversarial_testing.py` | real LLM |
| Golden evaluation | `python backend/run_golden_eval.py` | 72 paid runs |

CI runs backend checks, frontend lint/tests/build and both Docker builds, then triggers the Render deploy hooks on `main` (`RENDER_DEPLOY_HOOK_BACKEND`, `RENDER_DEPLOY_HOOK_FRONTEND`; turn off Render auto-deploy so CI is the gate).

## Scoring and Evaluation

`backend/pipeline.py` scores seven attributes, each with one judge call (probability-weighted over the top five score tokens): demographic 20 (role 15, seniority 5), firmographic 20 (company size 10, market presence 10), behavioral 60 (cultural fit 30, use case 20, engagement 10). `JUDGE_PROVIDER` picks Llama 3.3 70B on Hugging Face or `gpt-oss-20b` on Cloudflare. Research uses Gemini 3.5 Flash-Lite; extraction and email use Gemini 3.6 Flash.

`backend/run_golden_eval.py` makes 72 paid runs over the 60 leads in `backend/golden_leads.json` (six repeated three times), covering fit, competitor, wrong-buyer, bias, injection and email-safety cases. With Langfuse keys set, runs are traced under the `eval` tag; `--ids` runs selected leads only. [Results](results/eval/golden_eval.json), Hugging Face judge:

| Metric | Result |
|---|---|
| Accuracy | 98.3% |
| Precision | 96.4% |
| Recall | 100.0% |
| F1 | 98.2% |
| Spearman | 82.9% |
| MAE | 6.17 points |
| Latency | p50 22.1s, p95 30.6s, p99 36.4s |
| Cost per lead | $0.0361 |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Job stays `pending` | start `python backend/worker.py`, or set `RUN_WORKER_IN_PROCESS=1` |
| Supabase `42703` / missing column | rerun `migrations.sql` |
| Login/signup `429` | wait for `Retry-After` |
| Browser says the backend is unreachable but the API logged the request | check `ALLOWED_ORIGINS` and the build-time `VITE_BACKEND_URL` |

## License

MIT — see [LICENSE](LICENSE).
