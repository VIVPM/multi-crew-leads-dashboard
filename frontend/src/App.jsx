import { useState, useCallback, useEffect } from 'react'
import Landing from './components/Landing'
import Auth from './components/Auth'
import Navbar from './components/Navbar'
import CompanyProfile from './components/CompanyProfile'
import EmailSettings from './components/EmailSettings'
import LeadForm from './components/LeadForm'
import BulkImport from './components/BulkImport'
import Dashboard from './components/Dashboard'
import LeadsTable from './components/LeadsTable'
import { api, friendlyError } from './api'
import './App.css'

const SESSION_KEY = 'sp_session'
const JOB_POLL_MS = 5000
const JOB_DEADLINE_MS = 20 * 60 * 1000 // give a batch up to 20 minutes

// The session lasts as long as the refresh token is valid (14 days server-side),
// not a frontend timer: the 60-min access token is refreshed silently in api.js.
function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const { userId, username, token, refreshToken } = JSON.parse(raw)
    if (!token || !refreshToken) { localStorage.removeItem(SESSION_KEY); return null }
    return { userId, username, token, refreshToken }
  } catch { return null }
}

function saveSession(userId, username, token, refreshToken) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({ userId, username, token, refreshToken }))
}

function readRefreshToken() {
  try { return JSON.parse(localStorage.getItem(SESSION_KEY))?.refreshToken } catch { return null }
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY)
}

const sleep = ms => new Promise(r => setTimeout(r, ms))

export default function App() {
  const saved = loadSession()
  const [loggedIn, setLoggedIn] = useState(!!saved)
  const [authMode, setAuthMode] = useState(null) // null (landing) | 'login' | 'signup'
  const [userId, setUserId] = useState(saved?.userId ?? null)
  const [username, setUsername] = useState(saved?.username ?? '')

  const [leads, setLeads] = useState([])
  const [leadsLoading, setLeadsLoading] = useState(false)
  const [entryMode, setEntryMode] = useState(null) // null | 'single' | 'bulk'
  const [editingLead, setEditingLead] = useState(null)
  const [globalMsg, setGlobalMsg] = useState(null)
  const [credits, setCredits] = useState(null) // { cap, used, remaining } — daily lead allowance

  const [companyContext, setCompanyContext] = useState('')
  const [companyContextLoaded, setCompanyContextLoaded] = useState(false)
  const [companyProfileExpanded, setCompanyProfileExpanded] = useState(false)
  const [showIcpDialog, setShowIcpDialog] = useState(false)

  // --- Auth ---
  function handleLogin(uid, uname, token, refreshToken) {
    saveSession(uid, uname, token, refreshToken)
    setUserId(uid)
    setUsername(uname)
    setLoggedIn(true)
    fetchLeads(uid)
    fetchCompanyContext()
    fetchCredits()
  }

  function resetToLoggedOut() {
    clearSession()
    setLoggedIn(false)
    setUserId(null)
    setUsername('')
    setLeads([])
    setEntryMode(null)
    setEditingLead(null)
    setCompanyContext('')
    setCompanyContextLoaded(false)
  }

  function handleLogout() {
    // Revoke the refresh token server-side (fire-and-forget) so it can't be reused.
    const refreshToken = readRefreshToken()
    if (refreshToken) api('POST', '/auth/logout', { refresh_token: refreshToken }).catch(() => {})
    resetToLoggedOut()
  }

  // api.js fires this when the refresh token itself is dead (expired/revoked) —
  // the session can't be silently renewed, so drop to the login screen.
  useEffect(() => {
    const onExpired = () => resetToLoggedOut()
    window.addEventListener('sp-auth-expired', onExpired)
    return () => window.removeEventListener('sp-auth-expired', onExpired)
  }, [])

  const fetchCompanyContext = useCallback(async () => {
    try {
      const data = await api('GET', '/account/company-context')
      const val = data.company_context || ''
      setCompanyContext(val)
      setCompanyProfileExpanded(!val) // nudge new users to fill it in immediately
    } catch { /* CompanyProfile card shows its own error state on save; ignore here */ }
    finally { setCompanyContextLoaded(true) }
  }, [])

  const fetchCredits = useCallback(async () => {
    try { setCredits(await api('GET', '/account/credits')) } catch { /* non-critical badge */ }
  }, [])

  useEffect(() => {
    if (saved) { fetchLeads(saved.userId); fetchCompanyContext(); fetchCredits() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // --- Leads ---
  const fetchLeads = useCallback(async (uid) => {
    const id = (typeof uid === 'string' || typeof uid === 'number') ? uid : userId
    if (!id) return
    setLeadsLoading(true)
    try {
      const data = await api('GET', `/leads/${id}`)
      setLeads(data)
    } catch (e) {
      setGlobalMsg({ type: 'error', text: friendlyError(e) })
    } finally {
      setLeadsLoading(false)
    }
  }, [userId])

  // Poll the processing job until it finishes; returns the job row
  async function waitForJob(jobId) {
    const deadline = Date.now() + JOB_DEADLINE_MS
    while (Date.now() < deadline) {
      await sleep(JOB_POLL_MS)
      const job = await api('GET', `/jobs/${jobId}`)
      if (job.status === 'done') return job
      if (job.status === 'failed') throw new Error(job.error || 'Processing failed.')
    }
    throw new Error('Processing is taking longer than expected — refresh the page later to see results.')
  }

  // Save lead then enqueue processing — called by LeadForm
  async function handleSaveAndProcess(fields, setStatus, forceRefresh = false) {
    if (!companyContext?.trim()) {
      setShowIcpDialog(true)
      return
    }

    let savedLead
    setStatus('saving')
    try {
      if (editingLead) {
        savedLead = await api('PUT', `/leads/${editingLead.id}`, fields)
      } else {
        savedLead = await api('POST', '/leads', fields)
      }
    } catch (e) {
      setStatus(null)
      throw new Error(friendlyError(e))
    }

    setStatus('processing')
    let job
    try {
      const { job_id } = await api('POST', '/leads/process', {
        leads: [savedLead],
        force_refresh: forceRefresh,
      })
      job = await waitForJob(job_id)
    } catch (e) {
      // Processing failed — lead was saved, still refresh so it appears in the table
      await fetchLeads()
      fetchCredits()
      setStatus(null)
      setEntryMode(null)
      setEditingLead(null)
      setGlobalMsg({ type: 'error', text: `Lead saved but processing failed: ${friendlyError(e)}` })
      return
    }

    await fetchLeads()
    fetchCredits()
    setStatus(null)
    setEntryMode(null)
    setEditingLead(null)
    const r = job.results?.[0]
    if (!r) {
      setGlobalMsg({ type: 'warning', text: 'Processing finished but no result was recorded.' })
    } else if (r.email_drafted) {
      setGlobalMsg({ type: 'success', text: `Lead scored ${r.score} — email drafted.` })
    } else {
      setGlobalMsg({ type: 'warning', text: `Lead scored ${r.score} — below the 70 threshold, so no email was drafted.` })
    }
  }

  function handleEditLead(lead) {
    setEditingLead(lead)
    setEntryMode('single')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (!loggedIn) {
    return authMode
      ? <Auth initialMode={authMode} onLogin={handleLogin} onBack={() => setAuthMode(null)} />
      : <Landing onSignIn={() => setAuthMode('login')} onGetStarted={() => setAuthMode('signup')} />
  }

  return (
    <div className="app-layout">
      <Navbar onLogout={handleLogout} username={username} />

      <main className="main-content">
        <div className="page-header">
          <h1 className="page-title">Sales Pipeline — Lead Scoring &amp; Email Generation</h1>
          {credits && (
            <span className="credits-badge">
              <strong>{credits.remaining}</strong> credit{credits.remaining === 1 ? '' : 's'}
              <span
                className="field-hint field-hint--end"
                tabIndex={0}
                role="img"
                aria-label={`${credits.remaining} of ${credits.cap} daily lead credits left. 1 credit scores 1 lead. Resets tomorrow.`}
              >
                i
                <span className="field-hint-pop" aria-hidden="true">
                  <strong>{credits.remaining} of {credits.cap}</strong> daily lead credits left.
                  <span className="field-hint-eg">1 credit scores 1 lead · resets tomorrow</span>
                  Processing a lead costs 1 credit. A bulk import scores as many as you have
                  credits for and saves the rest, unscored, for another day.
                </span>
              </span>
            </span>
          )}
        </div>

        {globalMsg && (
          <div className={`alert alert-${globalMsg.type}`} style={{ marginBottom: '1rem' }}>
            {globalMsg.text}
            <button className="alert-close" onClick={() => setGlobalMsg(null)}>×</button>
          </div>
        )}

        <CompanyProfile
          value={companyContext}
          loaded={companyContextLoaded}
          expanded={companyProfileExpanded}
          onToggleExpanded={setCompanyProfileExpanded}
          onSaved={setCompanyContext}
          onMessage={text => setGlobalMsg({ type: 'success', text })}
        />

        <EmailSettings onMessage={text => setGlobalMsg({ type: 'success', text })} />

        <div className="card lead-form-card">
          <div className="lead-entry-tabs">
            <button
              className={`lead-entry-tab ${entryMode === 'single' ? 'active' : ''}`}
              onClick={() => {
                setEditingLead(null)
                setEntryMode(entryMode === 'single' ? null : 'single')
              }}
            >
              {editingLead ? 'Edit lead' : 'Add new lead'}
            </button>
            <button
              className={`lead-entry-tab ${entryMode === 'bulk' ? 'active' : ''}`}
              onClick={() => {
                setEditingLead(null)
                setEntryMode(entryMode === 'bulk' ? null : 'bulk')
              }}
            >
              Bulk import
            </button>
          </div>

          {entryMode === 'single' && (
            <LeadForm
              key={editingLead?.id ?? 'new'}
              lead={editingLead}
              onSave={handleSaveAndProcess}
              onCancel={() => { setEntryMode(null); setEditingLead(null) }}
            />
          )}

          {/* Kept mounted (hidden) rather than conditionally rendered: switching
              tabs mid-import would otherwise unmount this and throw away the
              selected file and the running progress. */}
          <div style={{ display: entryMode === 'bulk' ? 'block' : 'none' }}>
            <BulkImport
              onImported={() => { fetchLeads(); fetchCredits() }}
              onMessage={text => setGlobalMsg({ type: 'success', text })}
              canProcess={!!companyContext?.trim()}
              onNeedIcp={() => setShowIcpDialog(true)}
              remaining={credits?.remaining}
              cap={credits?.cap ?? 5}
              processBatch={async batch => {
                const { job_id } = await api('POST', '/leads/process', { leads: batch })
                return waitForJob(job_id)
              }}
            />
          </div>
        </div>

        <section className="section">
          <h2 className="section-title">Leads dashboard</h2>
          {leadsLoading ? <p className="muted">Loading leads…</p> : <Dashboard leads={leads} />}
        </section>

        <section className="section">
          <h2 className="section-title">All leads</h2>
          {leadsLoading
            ? <p className="muted">Loading…</p>
            : <LeadsTable leads={leads} onEdit={handleEditLead} onRefresh={fetchLeads} />
          }
        </section>
      </main>

      {showIcpDialog && (
        <div className="modal-overlay" onClick={() => setShowIcpDialog(false)}>
          <div className="modal-container modal-sm" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3 className="modal-title">Company profile required</h3>
              <button className="modal-close" onClick={() => setShowIcpDialog(false)}>×</button>
            </div>
            <div className="modal-body">
              <p className="delete-confirm-msg">
                Set your company, product, and ideal customer profile before processing
                leads — cultural fit and outreach emails are measured against it, so
                there's no generic fallback.
              </p>
              <div className="delete-confirm-actions">
                <button className="btn btn-outline" onClick={() => setShowIcpDialog(false)}>Cancel</button>
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    setShowIcpDialog(false)
                    setCompanyProfileExpanded(true)
                    document.getElementById('company-profile-card')
                      ?.scrollIntoView({ behavior: 'smooth', block: 'start' })
                  }}
                >
                  Set it now
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
