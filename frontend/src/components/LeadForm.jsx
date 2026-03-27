import { useState, useEffect } from 'react'

const EMAIL_RE = /^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$/
const MAX_LEN = 255
const SOURCES = ['Website', 'Referral', 'Event', 'Social Media', 'Other']

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

export default function LeadForm({ lead, onSave, onCancel, keysReady }) {
  const [fields, setFields] = useState({
    name: '', job_title: '', company: '', email: '',
    use_case: '', industry: '', location: '', source: 'Website',
  })
  const [errors, setErrors] = useState([])
  const [status, setStatus] = useState(null) // 'saving' | 'processing' | null

  useEffect(() => {
    if (lead) {
      setFields({
        name: lead.name || '',
        job_title: lead.job_title || '',
        company: lead.company || '',
        email: lead.email || '',
        use_case: lead.use_case || '',
        industry: INDUSTRIES.includes(lead.industry) ? lead.industry : '',
        location: lead.location || '',
        source: lead.source || 'Website',
      })
    }
  }, [lead])

  function set(key, val) {
    setFields(f => ({ ...f, [key]: val }))
  }

  async function handleSubmit(e) {
    e.preventDefault()
    const errs = validate(fields)
    if (errs.length) { setErrors(errs); return }
    setErrors([])
    try {
      await onSave(fields, setStatus)
    } catch (err) {
      setErrors([err.message || String(err)])
      setStatus(null)
    }
  }

  const loading = status !== null

  return (
    <div className="card lead-form-card">
      <h3 className="card-title">{lead ? '✏️ Edit Lead' : '➕ Add New Lead'}</h3>

      {!keysReady && (
        <div className="alert alert-warning">
          Enter your Sambanova &amp; Tavily API keys in the sidebar before saving.
        </div>
      )}

      {errors.map((e, i) => (
        <div key={i} className="alert alert-error">{e}</div>
      ))}

      {status === 'processing' && (
        <div className="alert alert-info">
          ⏳ AI crew is scoring and writing email — this may take a few minutes…
        </div>
      )}

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
            <label>Use Case</label>
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

        <div className="form-actions">
          <button type="submit" className="btn btn-success" disabled={loading || !keysReady}>
            {status === 'saving' ? '💾 Saving…' : status === 'processing' ? '⚡ Processing…' : '⚡ Save & Process'}
          </button>
          <button type="button" className="btn btn-outline" onClick={onCancel} disabled={loading}>
            Cancel
          </button>
        </div>
      </form>
    </div>
  )
}
