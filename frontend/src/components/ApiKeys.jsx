// Lets users pick a model provider, save its credentials and a Tavily key, and so pay for their own usage.
import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

const MODEL_FIELDS = {
  GEMINI: [
    { id: 'gemini', label: 'Gemini API key', link: 'https://aistudio.google.com/app/apikey', placeholder: 'AIza…', secret: true },
  ],
  CLOUDFLARE: [
    { id: 'cloudflare_account_id', label: 'Cloudflare account ID', link: 'https://dash.cloudflare.com', placeholder: '32-character account ID', secret: false },
    { id: 'cloudflare_api_token', label: 'Cloudflare API token (Workers AI)', link: 'https://dash.cloudflare.com/profile/api-tokens', placeholder: 'Token with Workers AI access', secret: true },
  ],
}
const TAVILY_FIELD = { id: 'tavily', label: 'Tavily API key', link: 'https://app.tavily.com', placeholder: 'tvly-…', secret: true }
const PROVIDER_NAMES = { GEMINI: 'Gemini', CLOUDFLARE: 'Cloudflare' }
const EMPTY_VALUES = { gemini: '', cloudflare_account_id: '', cloudflare_api_token: '', tavily: '' }

// Names the fields still missing before the user's own keys lift the daily limit.
function missingFields(saved, provider) {
  return [...(MODEL_FIELDS[provider] || []), TAVILY_FIELD].filter(f => !saved[f.id]?.saved).map(f => f.label)
}

// Describes which model runs the leads and whether daily credits apply.
function summary(saved, provider) {
  if (saved.unlimited) return `Using your own ${PROVIDER_NAMES[saved.active_provider]} and Tavily keys — no daily credit limit.`
  if (!MODEL_FIELDS[provider]) return "Using the app's model — daily credits apply. Choose a model and add your keys to remove the limit."
  return `Daily credits apply until you add: ${missingFields(saved, provider).join(', ')}.`
}

export default function ApiKeys({ onMessage, onChanged }) {
  const [expanded, setExpanded] = useState(false)
  const [saved, setSaved] = useState({})
  const [provider, setProvider] = useState('')
  const [values, setValues] = useState(EMPTY_VALUES)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  function load() {
    return api('GET', '/account/api-keys').then(data => {
      setSaved(data)
      setProvider(data.provider || 'APP')
    }).catch(() => {})
  }

  useEffect(() => { load() }, [])

  async function handleSave() {
    setBusy(true)
    setError(null)
    try {
      await api('PUT', '/account/api-keys', {
        llm_provider: provider,
        gemini_api_key: values.gemini || null,
        cloudflare_account_id: values.cloudflare_account_id || null,
        cloudflare_api_token: values.cloudflare_api_token || null,
        tavily_api_key: values.tavily || null,
      })
      setValues(EMPTY_VALUES)
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

  function keyField(f) {
    return (
      <div className="form-group" key={f.id} style={{ marginTop: '0.75rem' }}>
        <label htmlFor={`${f.id}-api-key`}>
          {f.label}{' '}
          <a href={f.link} target="_blank" rel="noreferrer" style={{ fontSize: '0.8rem' }}>get one</a>
        </label>
        {saved[f.id]?.saved && (
          <div className="muted" style={{ fontSize: '0.8rem', marginBottom: '0.35rem' }}>
            Saved, ending in ••••{saved[f.id].last4}{' '}
            <button className="btn btn-sm btn-outline" onClick={() => handleRemove(f.id)} disabled={busy}>
              Remove
            </button>
          </div>
        )}
        <input
          id={`${f.id}-api-key`}
          type={f.secret ? 'password' : 'text'}
          autoComplete="off"
          value={values[f.id]}
          onChange={e => setValues(v => ({ ...v, [f.id]: e.target.value }))}
          placeholder={saved[f.id]?.saved ? 'Enter a new value to replace it' : f.placeholder}
          disabled={busy}
        />
      </div>
    )
  }

  const changed = provider !== (saved.provider || 'APP') || Object.values(values).some(v => v.trim())

  return (
    <div className="card company-profile-card">
      <div className="company-profile-header" role="button" tabIndex={0} aria-expanded={expanded}
        onClick={() => setExpanded(!expanded)}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpanded(!expanded) }
        }}>
        <div>
          <h3 className="card-title" style={{ marginBottom: 0 }}>API keys (optional)</h3>
          {!expanded && <p className="muted company-profile-summary">{summary(saved, saved.provider)}</p>}
        </div>
        <span className="lead-card-chevron">{expanded ? '▲' : '▼'}</span>
      </div>

      {expanded && (
        <>
          <p className="muted" style={{ marginTop: '0.5rem' }}>
            Without your own keys, leads run on the app's model with daily credits. Choose a model, add its
            credentials and a Tavily key, and your leads run on your accounts with no daily limit.
            {' '}Keys are stored encrypted and never shown again.
          </p>

          <div className="form-group" style={{ marginTop: '0.75rem' }}>
            <label htmlFor="llm-provider">Model provider</label>
            <select id="llm-provider" value={provider} onChange={e => setProvider(e.target.value)} disabled={busy}>
              <option value="APP">App's model (daily credits)</option>
              <option value="GEMINI">Gemini</option>
              <option value="CLOUDFLARE">Cloudflare Workers AI</option>
            </select>
          </div>

          {(MODEL_FIELDS[provider] || []).map(keyField)}
          {keyField(TAVILY_FIELD)}

          {MODEL_FIELDS[provider] && !saved.unlimited && (
            <p className="muted" style={{ fontSize: '0.85rem' }}>{summary(saved, provider)}</p>
          )}
          {error && <div className="alert alert-error">{error}</div>}
          <button className="btn btn-primary" onClick={handleSave} disabled={busy || !changed}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </>
      )}
    </div>
  )
}
