import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

// Controlled by App.jsx: `value` is the saved profile, `expanded` is lifted
// so the "ICP required" dialog can force this open and scroll to it.
export default function CompanyProfile({ value, loaded, expanded, onToggleExpanded, onSaved, onMessage }) {
  const [text, setText] = useState(value)
  const [status, setStatus] = useState(null) // 'saving' | 'saved' | 'error' | null
  const [error, setError] = useState(null)

  useEffect(() => { setText(value) }, [value])

  async function handleSave() {
    setStatus('saving')
    setError(null)
    try {
      await api('PUT', '/account/company-context', { company_context: text })
      setStatus('saved')
      onSaved(text)
      onToggleExpanded(false)
      onMessage?.('Company profile saved.')
      setTimeout(() => setStatus(null), 2000)
    } catch (e) {
      setStatus('error')
      setError(friendlyError(e))
    }
  }

  const summary = text.split('\n')[0].slice(0, 80)

  return (
    <div className="card company-profile-card" id="company-profile-card">
      <div className="company-profile-header" onClick={() => onToggleExpanded(!expanded)}>
        <div>
          <h3 className="card-title" style={{ marginBottom: 0 }}>
            Your company &amp; ICP <span className="required-star">*</span>
          </h3>
          {!expanded && (
            <p className="muted company-profile-summary">
              {summary || 'Not set — required before processing leads.'}
            </p>
          )}
        </div>
        <span className="lead-card-chevron">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <>
          <p className="muted" style={{ marginTop: '0.5rem' }}>
            Describe your company, product, and ideal customer profile — every
            lead's cultural-fit score and outreach email are measured against this.
            Required before you can process a lead.
          </p>
          <div className="form-group" style={{ marginTop: '0.75rem' }}>
            <textarea
              rows={6}
              value={text}
              onChange={e => setText(e.target.value)}
              placeholder={'Company Name: Acme Inc.\nProduct: ...\nICP: ...\nPitch: ...'}
              disabled={!loaded}
            />
          </div>
          {error && <div className="alert alert-error">{error}</div>}
          <button className="btn btn-primary" onClick={handleSave} disabled={status === 'saving' || !loaded || !text.trim()}>
            {status === 'saving' ? 'Saving…' : status === 'saved' ? 'Saved ✓' : 'Save company profile'}
          </button>
        </>
      )}
    </div>
  )
}
