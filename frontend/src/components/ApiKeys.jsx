// Lets users save their own Gemini and Tavily keys, which override the operator's.
import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

const PROVIDERS = [
  { id: 'gemini', label: 'Gemini API key', link: 'https://aistudio.google.com/app/apikey', placeholder: 'AIza…' },
  { id: 'tavily', label: 'Tavily API key', link: 'https://app.tavily.com', placeholder: 'tvly-…' },
]

// Describes which keys are in use and whether daily credits apply.
function summary(saved) {
  if (saved.gemini?.saved && saved.tavily?.saved) return 'Using your own Gemini and Tavily keys — no daily credit limit.'
  if (saved.gemini?.saved || saved.tavily?.saved) return 'Using one of your own keys — add both to remove the daily credit limit.'
  return 'Using the app\'s keys — daily credits apply. Add both of yours to remove the limit.'
}

export default function ApiKeys({ onMessage, onChanged }) {
  const [expanded, setExpanded] = useState(false)
  const [saved, setSaved] = useState({})
  const [values, setValues] = useState({ gemini: '', tavily: '' })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  function load() {
    return api('GET', '/account/api-keys').then(setSaved).catch(() => {})
  }

  useEffect(() => { load() }, [])

  async function handleSave() {
    setBusy(true)
    setError(null)
    try {
      await api('PUT', '/account/api-keys', {
        gemini_api_key: values.gemini || null,
        tavily_api_key: values.tavily || null,
      })
      setValues({ gemini: '', tavily: '' })
      await load()
      onChanged?.()
      onMessage?.('API keys saved.')
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setBusy(false)
    }
  }

  async function handleRemove(id) {
    setBusy(true)
    setError(null)
    try {
      await api('DELETE', `/account/api-keys/${id}`)
      await load()
      onChanged?.()
    } catch (e) {
      setError(friendlyError(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card company-profile-card">
      <div className="company-profile-header" onClick={() => setExpanded(!expanded)}>
        <div>
          <h3 className="card-title" style={{ marginBottom: 0 }}>API keys (optional)</h3>
          {!expanded && <p className="muted company-profile-summary">{summary(saved)}</p>}
        </div>
        <span className="lead-card-chevron">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <>
          <p className="muted" style={{ marginTop: '0.5rem' }}>
            By default leads are processed with the app's keys and limited by daily credits.
            Save your own keys to use them instead — with both saved, the daily limit goes away
            since the usage is billed to you. Keys are stored encrypted and never shown again.
          </p>

          {PROVIDERS.map(p => (
            <div className="form-group" key={p.id} style={{ marginTop: '0.75rem' }}>
              <label>
                {p.label}{' '}
                <a href={p.link} target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>get one</a>
              </label>
              {saved[p.id]?.saved && (
                <div className="muted" style={{ fontSize: '0.8rem', marginBottom: '0.35rem' }}>
                  Saved key ending in ••••{saved[p.id].last4}{' '}
                  <button className="btn btn-sm btn-outline" onClick={() => handleRemove(p.id)} disabled={busy}>
                    Remove
                  </button>
                </div>
              )}
              <input
                type="password"
                autoComplete="off"
                value={values[p.id]}
                onChange={e => setValues(v => ({ ...v, [p.id]: e.target.value }))}
                placeholder={saved[p.id]?.saved ? 'Enter a new key to replace it' : p.placeholder}
                disabled={busy}
              />
            </div>
          ))}

          {error && <div className="alert alert-error">{error}</div>}
          <button
            className="btn btn-primary"
            onClick={handleSave}
            disabled={busy || (!values.gemini.trim() && !values.tavily.trim())}
          >
            {busy ? 'Saving…' : 'Save API keys'}
          </button>
        </>
      )}
    </div>
  )
}
