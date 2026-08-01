import { useState } from 'react'

const EMAIL_RE = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/
const MAX_LEN = 255
const SOURCES = ['Website', 'Referral', 'Event', 'Social Media', 'Other']

// Tooltip shown on the ⓘ beside the Use Case field
const USE_CASE_HINT =
  'What this lead is trying to achieve that your product can help with — written ' +
  'as a short goal in their own words, not a description of your product.\n\n' +
  'e.g. "automate customer-support triage" or "cut lead-response time"\n\n' +
  'The crew uses it to judge how well your product fits this lead, and to explain ' +
  'in the drafted email why your product suits that goal.'

// Pipeline stages in run order; keys match pipeline.py's on_stage names
const PROCESS_STEPS = [
  ['company', 'Company research & cultural fit'],
  ['personal_research', 'Personal research'],
  ['scoring', 'Lead scoring'],
  ['email', 'Email draft'],
]

// Live step tracker. The first stage the worker hasn't reported yet is the
// one currently running.
function StepTracker({ steps }) {
  const p = steps || {}
  const currentIdx = PROCESS_STEPS.findIndex(([key]) => !p[key])
  const doneCount = PROCESS_STEPS.filter(([key]) => p[key] === 'done' || p[key] === 'cached').length
  const pct = Math.round((doneCount / PROCESS_STEPS.length) * 100)
  return (
    <div className="alert alert-info bulk-progress">
      <div className="bulk-progress-head">
        Processing this lead — keep this tab open. {doneCount} of {PROCESS_STEPS.length} stages done.
      </div>
      <div className="bulk-progress-track">
        <div className="bulk-progress-bar" style={{ width: `${pct}%` }} />
      </div>
      <ul className="bulk-progress-list">
        {PROCESS_STEPS.map(([key, label], i) => {
          const state = p[key]
          const status =
            state === 'done' || state === 'cached' ? 'done'
            : i === currentIdx ? 'processing'
            : 'pending'
          return (
            <li key={key} className={`bulk-progress-item is-${status}`}>
              <span className="bulk-progress-icon">
                {status === 'processing'
                  ? <span className="spinner spinner-sm" aria-hidden="true" />
                  : status === 'done' ? '✓' : '○'}
              </span>
              <span className="bulk-progress-name">{label}</span>
              <span className="bulk-progress-state">
                {state === 'cached' ? 'cached'
                  : status === 'done' ? 'done'
                  : status === 'processing' ? 'working…'
                  : 'queued'}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

const INDUSTRIES = [
  'Technology & Software',
  'Finance & Banking',
  'Healthcare & Life Sciences',
  'E-commerce & Retail',
  'Media & Entertainment',
  'Manufacturing & Industrial',
  'Education & EdTech',
  'Real Estate & Construction',
  'Logistics & Supply Chain',
  'Energy & Utilities',
  'Telecommunications',
  'Professional Services',
  'Aerospace & Defense',
  'Government & Public Sector',
  'Other',
]

function validate(fields) {
  const errors = []
  if (!fields.name?.trim()) errors.push('Name is required.')
  if (!fields.company?.trim()) errors.push('Company is required.')
  if (!fields.email?.trim()) errors.push('Email is required.')
  else if (!EMAIL_RE.test(fields.email.trim())) errors.push('Invalid email format.')
  for (const [label, val] of [
    ['Name', fields.name], ['Job Title', fields.job_title],
    ['Company', fields.company], ['Email', fields.email],
    ['Use Case', fields.use_case], ['Industry', fields.industry],
    ['Location', fields.location],
  ]) {
    if (val && val.length > MAX_LEN)
      errors.push(`${label} must be under ${MAX_LEN} characters.`)
  }
  return errors
}

export default function LeadForm({ lead, onSave, onCancel }) {
  // The parent keys this component by lead id, so a different lead remounts it
  const [fields, setFields] = useState(() => ({
    name: lead?.name || '',
    job_title: lead?.job_title || '',
    company: lead?.company || '',
    email: lead?.email || '',
    use_case: lead?.use_case || '',
    industry: lead && INDUSTRIES.includes(lead.industry) ? lead.industry : '',
    location: lead?.location || '',
    source: lead?.source || 'Website',
  }))
  const [errors, setErrors] = useState([])
  const [status, setStatus] = useState(null) // 'saving' | 'processing' | null
  const [steps, setSteps] = useState(null)   // live {stage: state} progress map
  const [forceRefresh, setForceRefresh] = useState(false)

  function set(key, val) {
    setFields(f => ({ ...f, [key]: val }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const errs = validate(fields)
    // Show one problem at a time, in field order
    if (errs.length) { setErrors([errs[0]]); return }
    setErrors([])
    try {
      await onSave(fields, setStatus, forceRefresh, setSteps)
    } catch (err) {
      setErrors([err.message || String(err)])
      setStatus(null)
    }
  }

  const loading = status !== null

  return (
    <div className="lead-form-body">
      {errors.map((e, i) => (
        <div key={i} className="alert alert-error">{e}</div>
      ))}

      <form onSubmit={handleSubmit}>
        <div className="form-grid">
          <div className="form-group">
            <label>Name *</label>
            <input value={fields.name} onChange={e => set('name', e.target.value)} placeholder="Full name" disabled={loading} />
          </div>
          <div className="form-group">
            <label>Job Title</label>
            <input value={fields.job_title} onChange={e => set('job_title', e.target.value)} placeholder="e.g. CTO" disabled={loading} />
          </div>
          <div className="form-group">
            <label>Company *</label>
            <input value={fields.company} onChange={e => set('company', e.target.value)} placeholder="Company name" disabled={loading} />
          </div>
          <div className="form-group">
            <label>Email *</label>
            <input type="email" value={fields.email} onChange={e => set('email', e.target.value)} placeholder="email@company.com" disabled={loading} />
          </div>
          <div className="form-group">
            <label>
              Use Case
              <span className="field-hint" tabIndex={0} role="img" aria-label={USE_CASE_HINT}>
                i
                <span className="field-hint-pop" aria-hidden="true">
                  <strong>Use case</strong> — what this lead is trying to achieve that your
                  product can help with, in their own words (not a pitch of your product).
                  <span className="field-hint-eg">"automate customer-support triage" · "cut lead-response time"</span>
                  The crew uses it to score fit and to explain, in the drafted email, why your
                  product suits that goal.
                </span>
              </span>
            </label>
            <input value={fields.use_case} onChange={e => set('use_case', e.target.value)} placeholder="e.g. Automate support" disabled={loading} />
          </div>
          <div className="form-group">
            <label>Industry</label>
            <select value={fields.industry} onChange={e => set('industry', e.target.value)} disabled={loading}>
              <option value="">— Select industry —</option>
              {INDUSTRIES.map(i => <option key={i}>{i}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label>Location</label>
            <input value={fields.location} onChange={e => set('location', e.target.value)} placeholder="City, Country" disabled={loading} />
          </div>
          <div className="form-group">
            <label>Lead Source</label>
            <select value={fields.source} onChange={e => set('source', e.target.value)} disabled={loading}>
              {SOURCES.map(s => <option key={s}>{s}</option>)}
            </select>
          </div>
        </div>

        <label className="checkbox-row" title="Company research is cached across leads from the same company. Check this to ignore the cache and re-research it now.">
          <input
            type="checkbox"
            checked={forceRefresh}
            onChange={e => setForceRefresh(e.target.checked)}
            disabled={loading}
          />
          Force refresh company research (ignore cache)
        </label>

        <div className="form-actions">
          <button type="submit" className="btn btn-success" disabled={loading}>
            {status === 'saving' ? 'Saving…' : status === 'processing' ? 'Processing…' : 'Save and process'}
          </button>
          <button type="button" className="btn btn-outline" onClick={onCancel} disabled={loading}>
            Cancel
          </button>
        </div>
      </form>

      {status === 'processing' && <StepTracker steps={steps} />}
    </div>
  )
}
