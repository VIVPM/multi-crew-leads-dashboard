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

// Session length is server-side (14-day refresh token cookie), not a
// frontend timer. What's stored here is just a UI hint — {userId, username}
// — never a token; the cookies (httpOnly, set by backend.py) are the actual
// auth. A stale hint with dead cookies just means the next api() call 401s
// and sp-auth-expired clears it, same as if this were empty.
function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const { userId, username } = JSON.parse(raw)
    if (!userId) { localStorage.removeItem(SESSION_KEY); return null }
    return { userId, username }
  } catch { return null }
}

// csrfToken comes from the login/signup/refresh response body, not a cookie
// — see api.js's doFetch comment for why the cookie itself isn't readable
// here in production (frontend and backend are different origins).
function saveSession(userId, username, csrfToken) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({ userId, username, csrfToken }))
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY)
}

const sleep = ms => new Promise(r => setTimeout(r, ms))

const PAGE_TITLES = {
  'add-lead': 'Add a lead',
  dashboard: 'Dashboard',
  leads: 'Lead details',
  settings: 'Settings',
}

export default function App() {
  const saved = loadSession()
  const [loggedIn, setLoggedIn] = useState(!!saved)
  const [authMode, setAuthMode] = useState(null) // null (landing) | 'login' | 'signup'
  const [userId, setUserId] = useState(saved?.userId ?? null)
  const [username, setUsername] = useState(saved?.username ?? '')

  const [page, setPage] = useState('add-lead') // 'add-lead' | 'dashboard' | 'leads' | 'settings'
  const [leads, setLeads] = useState([])
  const [leadsLoading, setLeadsLoading] = useState(false)
  const [entryMode, setEntryMode] = useState('single') // 'single' | 'bulk'
  const [editingLead, setEditingLead] = useState(null)
  // Bumped after a process or cancel to remount LeadForm with a clean slate
  const [formResetKey, setFormResetKey] = useState(0)
  const [globalMsg, setGlobalMsg] = useState(null)
  const [credits, setCredits] = useState(null) // { cap, used, remaining } — daily lead allowance

  const [companyContext, setCompanyContext] = useState('')
  const [companyContextLoaded, setCompanyContextLoaded] = useState(false)
  const [companyProfileExpanded, setCompanyProfileExpanded] = useState(false)
  const [showIcpDialog, setShowIcpDialog] = useState(false)

  // --- Auth ---
  function handleLogin(uid, uname, csrfToken) {
    saveSession(uid, uname, csrfToken)
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
    setPage('add-lead')
    setEntryMode('single')
    setEditingLead(null)
    setCompanyContext('')
    setCompanyContextLoaded(false)
  }

  function handleLogout() {
    // Revoke the refresh token server-side (fire-and-forget, cookie carries
    // it) and clear the auth cookies so it can't be reused.
    api('POST', '/auth/logout').catch(() => {})
    resetToLoggedOut()
  }

  // Fired by api.js when the refresh token is dead and can't be renewed
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

  // Poll a job until it finishes. onProgress receives its {stage: state} map each poll.
  async function waitForJob(jobId, onProgress) {
    const deadline = Date.now() + JOB_DEADLINE_MS
    while (Date.now() < deadline) {
      await sleep(JOB_POLL_MS)
      const job = await api('GET', `/jobs/${jobId}`)
      onProgress?.(job.progress)
      if (job.status === 'done') return job
      if (job.status === 'failed') throw new Error(job.error || 'Processing failed.')
    }
    throw new Error('Processing is taking longer than expected — refresh the page later to see results.')
  }

  // Save the lead, then enqueue processing — called by LeadForm
  async function handleSaveAndProcess(fields, setStatus, forceRefresh = false, setSteps) {
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
    setSteps?.(null)
    let job
    try {
      const { job_id } = await api('POST', '/leads/process', {
        leads: [savedLead],
        force_refresh: forceRefresh,
      })
      job = await waitForJob(job_id, setSteps)
    } catch (e) {
      // Processing failed — lead was saved, still refresh so it appears in the table
      await fetchLeads()
      fetchCredits()
      setStatus(null)
      setEntryMode('single')
      setEditingLead(null)
      setFormResetKey(k => k + 1)
      setGlobalMsg({ type: 'error', text: `Lead saved but processing failed: ${friendlyError(e)}` })
      return
    }

    await fetchLeads()
    fetchCredits()
    setStatus(null)
    setEntryMode('single')
    setEditingLead(null)
    setFormResetKey(k => k + 1)
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
    setPage('add-lead')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (!loggedIn) {
    return authMode
      ? <Auth initialMode={authMode} onLogin={handleLogin} onBack={() => setAuthMode(null)} />
      : <Landing onSignIn={() => setAuthMode('login')} onGetStarted={() => setAuthMode('signup')} />
  }

  return (
    <div className="app-layout">
      <Navbar
        username={username}
        page={page}
        onNavigate={setPage}
        onLogout={handleLogout}
        credits={credits}
      />

      <main className="main-content">
        <div className="page-header">
          <h1 className="page-title">{PAGE_TITLES[page]}</h1>
        </div>

        {globalMsg && (
          <div className={`alert alert-${globalMsg.type}`} style={{ marginBottom: '1rem' }}>
            {globalMsg.text}
            <button className="alert-close" onClick={() => setGlobalMsg(null)}>×</button>
          </div>
        )}

        {/* ---- Add a lead ---- */}
        {page === 'add-lead' && (
          <div className="card lead-form-card">
            <div className="lead-entry-tabs">
              <button
                className={`lead-entry-tab ${entryMode === 'single' ? 'active' : ''}`}
                onClick={() => { setEditingLead(null); setFormResetKey(k => k + 1); setEntryMode('single') }}
              >
                {editingLead ? 'Edit lead' : 'Add new lead'}
              </button>
              <button
                className={`lead-entry-tab ${entryMode === 'bulk' ? 'active' : ''}`}
                onClick={() => { setEditingLead(null); setEntryMode('bulk') }}
              >
                Bulk import
              </button>
            </div>

            {entryMode === 'single' && (
              <LeadForm
                key={`${editingLead?.id ?? 'new'}-${formResetKey}`}
                lead={editingLead}
                onSave={handleSaveAndProcess}
                onCancel={() => { setEditingLead(null); setFormResetKey(k => k + 1) }}
              />
            )}

            {/* Kept mounted (hidden) so switching tabs mid-import doesn't throw
                away the selected file and the running progress. */}
            <div style={{ display: entryMode === 'bulk' ? 'block' : 'none' }}>
              <BulkImport
                onImported={() => { fetchLeads(); fetchCredits() }}
                onMessage={text => setGlobalMsg({ type: 'success', text })}
                canProcess={!!companyContext?.trim()}
                onNeedIcp={() => setShowIcpDialog(true)}
                remaining={credits?.remaining}
                cap={credits?.cap}
                processLead={async lead => {
                  const { job_id } = await api('POST', '/leads/process', { leads: [lead] })
                  return waitForJob(job_id)
                }}
              />
            </div>
          </div>
        )}

        {/* ---- Dashboard ---- */}
        {page === 'dashboard' && (
          leadsLoading ? <p className="muted">Loading leads…</p> : <Dashboard leads={leads} />
        )}

        {/* ---- Processed leads ---- */}
        {page === 'leads' && (
          leadsLoading
            ? <p className="muted">Loading…</p>
            : <LeadsTable leads={leads} onEdit={handleEditLead} onRefresh={fetchLeads} />
        )}

        {/* ---- Settings ---- */}
        {page === 'settings' && (
          <>
            <CompanyProfile
              value={companyContext}
              loaded={companyContextLoaded}
              expanded={companyProfileExpanded}
              onToggleExpanded={setCompanyProfileExpanded}
              onSaved={setCompanyContext}
              onMessage={text => setGlobalMsg({ type: 'success', text })}
            />
            <div id="email-settings">
              <EmailSettings onMessage={text => setGlobalMsg({ type: 'success', text })} />
            </div>
          </>
        )}
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
                    setPage('settings')
                    setCompanyProfileExpanded(true)
                    // let the Settings page render before scrolling to the card
                    setTimeout(() => document.getElementById('company-profile-card')
                      ?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 60)
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
