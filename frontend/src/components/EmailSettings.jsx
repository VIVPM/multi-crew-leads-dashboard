import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

// Self-contained (fetches its own settings) unlike CompanyProfile, since
// nothing else in the app needs this state — sending is opt-in per lead,
// not required to use the rest of the product.
export default function EmailSettings() {
  const [expanded, setExpanded] = useState(false)
  const [loaded, setLoaded] = useState(false)
  const [configured, setConfigured] = useState(false)
  const [fromAddress, setFromAddress] = useState('')
  const [smtpHost, setSmtpHost] = useState('smtp.gmail.com')
  const [smtpPort, setSmtpPort] = useState(587)
  const [smtpPassword, setSmtpPassword] = useState('')
  const [status, setStatus] = useState(null) // 'saving' | 'saved' | 'error' | null
  const [error, setError] = useState(null)

  useEffect(() => {
    api('GET', '/account/email-settings')
      .then(data => {
        setFromAddress(data.from_address || '')
        setSmtpHost(data.smtp_host || 'smtp.gmail.com')
        setSmtpPort(data.smtp_port || 587)
        setConfigured(!!data.configured)
      })
      .catch(() => { /* falls back to defaults; save flow surfaces its own errors */ })
      .finally(() => setLoaded(true))
  }, [])

  async function handleSave() {
    setStatus('saving')
    setError(null)
    try {
      await api('PUT', '/account/email-settings', {
        smtp_host: smtpHost,
        smtp_port: Number(smtpPort),
        from_address: fromAddress,
        smtp_password: smtpPassword,
      })
      setStatus('saved')
      setConfigured(true)
      setSmtpPassword('') // never keep the password in the form after saving
      setExpanded(false)
      setTimeout(() => setStatus(null), 2000)
    } catch (e) {
      setStatus('error')
      setError(friendlyError(e))
    }
  }

  return (
    <div className="card company-profile-card">
      <div className="company-profile-header" onClick={() => setExpanded(!expanded)}>
        <div>
          <h3 className="card-title" style={{ marginBottom: 0 }}>Email sending</h3>
          {!expanded && (
            <p className="muted company-profile-summary">
              {configured ? `Sending as ${fromAddress}` : 'Not set up — drafts can be edited but not sent.'}
            </p>
          )}
        </div>
        <span className="lead-card-chevron">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <>
          <p className="muted" style={{ marginTop: '0.5rem' }}>
            Connect your own email account to actually send drafted outreach,
            not just view it. Works with any SMTP-capable provider — Gmail's
            defaults are pre-filled, change the host/port for anything else.
            For Gmail: enable 2-Step Verification, then create an{' '}
            <a href="https://myaccount.google.com/apppasswords" target="_blank" rel="noreferrer">
              App Password
            </a>{' '}
            (not your normal password) and paste it below.
          </p>

          <div className="form-group" style={{ marginTop: '0.75rem' }}>
            <label>From address</label>
            <input
              type="email"
              value={fromAddress}
              onChange={e => setFromAddress(e.target.value)}
              placeholder="you@example.com"
              disabled={!loaded}
            />
          </div>
          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <div className="form-group" style={{ flex: 2 }}>
              <label>SMTP host</label>
              <input
                type="text"
                value={smtpHost}
                onChange={e => setSmtpHost(e.target.value)}
                placeholder="smtp.gmail.com"
                disabled={!loaded}
              />
            </div>
            <div className="form-group" style={{ flex: 1 }}>
              <label>Port</label>
              <input
                type="number"
                value={smtpPort}
                onChange={e => setSmtpPort(e.target.value)}
                disabled={!loaded}
              />
            </div>
          </div>
          <div className="form-group">
            <label>App password {configured && <span className="muted">(leave blank to keep current)</span>}</label>
            <input
              type="password"
              value={smtpPassword}
              onChange={e => setSmtpPassword(e.target.value)}
              placeholder={configured ? '••••••••••••••••' : '16-character app password'}
              disabled={!loaded}
            />
          </div>

          {error && <div className="alert alert-error">{error}</div>}
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={status === 'saving' || !loaded || !fromAddress.trim() || !smtpHost.trim() || (!configured && !smtpPassword)}
          >
            {status === 'saving' ? 'Saving…' : status === 'saved' ? 'Saved ✓' : 'Save email settings'}
          </button>
        </>
      )}
    </div>
  )
}
