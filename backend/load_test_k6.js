// load_test_k6.js — one load level, or one spike, of realistic users against the stubbed local API; run by load_test_k6.py.
import http from 'k6/http'
import { sleep } from 'k6'
import exec from 'k6/execution'
import { Trend, Rate, Counter } from 'k6/metrics'

const BASE = __ENV.BASE
const VUS = Number(__ENV.VUS)
const RAMP_S = Number(__ENV.RAMP_S)
const HOLD_S = Number(__ENV.HOLD_S)
const MAX_JOBS = Number(__ENV.MAX_JOBS_PER_USER)
const POLL_S = 5
const STAGES = __ENV.STAGES
  ? JSON.parse(__ENV.STAGES)
  : [{ seconds: RAMP_S, target: VUS }, { seconds: HOLD_S, target: VUS }]
const SUBMIT_WINDOW_S = Number(__ENV.SUBMIT_WINDOW_S || HOLD_S * 0.6)
const PHASES = __ENV.PHASES ? JSON.parse(__ENV.PHASES) : []
const PHASE_ENDPOINTS = ['login', 'leads', 'job_status', 'process']
const accounts = JSON.parse(open(__ENV.ACCOUNTS))

const ENDPOINTS = ['login', 'leads', 'credits', 'company_context', 'create_lead', 'process', 'job_status']
const latency = Object.fromEntries(ENDPOINTS.map(e => [e, new Trend(`api_${e}`, true)]))
const failed = Object.fromEntries(ENDPOINTS.map(e => [e, new Rate(`err_${e}`)]))
const jobsSubmitted = new Counter('jobs_submitted')
const phaseLatency = Object.fromEntries(PHASES.flatMap(ph =>
  PHASE_ENDPOINTS.map(e => [`${ph.name}_${e}`, new Trend(`phase_${ph.name}_${e}`, true)])))

export const options = {
  scenarios: {
    users: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: STAGES.map(st => ({ duration: `${st.seconds}s`, target: st.target })),
      gracefulRampDown: '10s',
      gracefulStop: '30s',
    },
  },
  summaryTrendStats: ['p(50)', 'p(95)', 'p(99)', 'avg', 'max', 'count'],
  thresholds: {
    api_leads: ['p(50)<800', 'p(95)<1500', 'p(99)<3000'],
    api_credits: ['p(50)<800', 'p(95)<1500', 'p(99)<3000'],
    api_company_context: ['p(50)<800', 'p(95)<1500', 'p(99)<3000'],
    api_job_status: ['p(50)<800', 'p(95)<1500', 'p(99)<3000'],
    api_process: ['p(50)<2500', 'p(95)<4000', 'p(99)<6000'],
    http_req_failed: ['rate<0.01'],
  },
}

// Names the spike phase (before, spike, after) the run is in, from seconds since the scenario started.
function currentPhase() {
  const elapsed = (Date.now() - exec.scenario.startTime) / 1000
  const phase = PHASES.find(ph => elapsed < ph.end_s)
  return phase ? phase.name : null
}

// Times one request under its endpoint name, and its spike phase if any, and counts anything but the expected status as an error.
function call(name, res, ok) {
  latency[name].add(res.timings.duration)
  const phase = currentPhase()
  if (phase && phaseLatency[`${phase}_${name}`]) phaseLatency[`${phase}_${name}`].add(res.timings.duration)
  const bad = res.status !== ok
  failed[name].add(bad)
  return bad ? null : res
}

let session = null
let job = null
let submitted = 0
let submitAt = null

// Each VU is one person with a browser tab: log in once, browse, submit a bounded number of leads, poll until done.
export default function () {
  if (!session) {
    const acct = accounts[(__VU - 1) % accounts.length]
    const res = call('login', http.post(`${BASE}/auth/login`, JSON.stringify(acct),
      { headers: { 'Content-Type': 'application/json' }, tags: { name: 'login' } }), 200)
    if (!res) { sleep(5); return }
    const body = res.json()
    session = { userId: body.user_id, headers: { Authorization: `Bearer ${body.token}`, 'Content-Type': 'application/json' } }
    submitAt = Date.now() + Math.random() * SUBMIT_WINDOW_S * 1000
  }
  const params = name => ({ headers: session.headers, tags: { name } })

  if (job) {
    const res = call('job_status', http.get(`${BASE}/jobs/${job}`, params('job_status')), 200)
    if (res && ['done', 'failed'].includes(res.json('status'))) job = null
    sleep(POLL_S)
    return
  }

  if (submitted < MAX_JOBS && Date.now() >= submitAt) {
    const n = `${__VU}-${__ITER}`
    const lead = call('create_lead', http.post(`${BASE}/leads`, JSON.stringify({
      name: `K6 Lead ${n}`, job_title: 'VP Engineering', company: `K6co ${n}`,
      email: `k6-${n}@example.com`, use_case: 'Automate lead qualification', source: 'Website',
    }), params('create_lead')), 200)
    if (lead) {
      const res = call('process', http.post(`${BASE}/leads/process`, JSON.stringify({ leads: [lead.json()] }),
        { headers: { ...session.headers, 'Idempotency-Key': `k6-${n}-${Date.now()}` }, tags: { name: 'process' } }), 202)
      if (res) { job = res.json('job_id'); submitted += 1; jobsSubmitted.add(1) }
    }
    sleep(POLL_S)
    return
  }

  const page = http.batch([
    ['GET', `${BASE}/leads/${session.userId}`, null, params('leads')],
    ['GET', `${BASE}/account/credits`, null, params('credits')],
    ['GET', `${BASE}/account/company-context`, null, params('company_context')],
  ])
  call('leads', page[0], 200)
  call('credits', page[1], 200)
  call('company_context', page[2], 200)
  sleep(5 + Math.random() * 10)
}

// Writes the end-of-run summary as JSON for load_test_k6.py.
export function handleSummary(data) {
  return { [__ENV.SUMMARY]: JSON.stringify(data) }
}
