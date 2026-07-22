import { useState, useCallback, useEffect } from 'react'
import Landing from './components/Landing'
import Auth from './components/Auth'
import Navbar from './components/Navbar'
import CompanyProfile from './components/CompanyProfile'
import EmailSettings from './components/EmailSettings'
import LeadForm from './components/LeadForm'
import Dashboard from './components/Dashboard'
import LeadsTable from './components/LeadsTable'
import { api, friendlyError } from './api'
import './App.css'

const SESSION_KEY = 'sp_session'
const SESSION_TTL = 60 * 60 * 1000 // 1 hour in ms
const JOB_POLL_MS = 5000
const JOB_DEADLINE_MS = 20 * 60 * 1000 // give a batch up to 20 minutes

function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const { userId, username, token, expiresAt } = JSON.parse(raw)
    if (!token || Date.now() > expiresAt) { localStorage.removeItem(SESSION_KEY); return null }
    return { userId, username, token }
  } catch { return null }
}

function saveSession(userId, username, token) {
  localStorage.setItem(SESSION_KEY, JSON.stringify({
    userId, username, token, expiresAt: Date.now() + SESSION_TTL,
  }))
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
  const [addingLead, setAddingLead] = useState(false)
  const [editingLead, setEditingLead] = useState(null)
  const [globalMsg, setGlobalMsg] = useState(null)

  const [companyContext, setCompanyContext] = useState('')
  const [companyContextLoaded, setCompanyContextLoaded] = useState(false)
  const [companyProfileExpanded, setCompanyProfileExpanded] = useState(false)
  const [showIcpDialog, setShowIcpDialog] = useState(false)

  // --- Auth ---
  function handleLogin(uid, uname, token) {
    saveSession(uid, uname, token)
    setUserId(uid)
    setUsername(uname)
    setLoggedIn(true)
    fetchLeads(uid)
    fetchCompanyContext()
  }

  function handleLogout() {
    clearSession()
    setLoggedIn(false)
    setUserId(null)
    setUsername('')
    setLeads([])
    setAddingLead(false)
    setEditingLead(null)
    setCompanyContext('')
    setCompanyContextLoaded(false)
  }

  const fetchCompanyContext = useCallback(async () => {
    try {
      const data = await api('GET', '/account/company-context')
      const val = data.company_context || ''
      setCompanyContext(val)
      setCompanyProfileExpanded(!val) // nudge new users to fill it in immediately
    } catch { /* CompanyProfile card shows its own error state on save; ignore here */ }
    finally { setCompanyContextLoaded(true) }
  }, [])

  useEffect(() => {
    if (saved) { fetchLeads(saved.userId); fetchCompanyContext() }
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
      setStatus(null)
      setAddingLead(false)
      setEditingLead(null)
      setGlobalMsg({ type: 'error', text: `Lead saved but processing failed: ${friendlyError(e)}` })
      return
    }

    await fetchLeads()
    setStatus(null)
    setAddingLead(false)
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
    setAddingLead(true)
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
        />

        <EmailSettings />

        <div className="card lead-form-card">
          <div
            className="company-profile-header"
            onClick={() => {
              if (addingLead) { setAddingLead(false); setEditingLead(null) }
              else { setEditingLead(null); setAddingLead(true) }
            }}
          >
            <h3 className="card-title" style={{ marginBottom: 0 }}>
              {editingLead ? 'Edit lead' : 'Add new lead'}
            </h3>
            <span className="lead-card-chevron">{addingLead ? '▲' : '▼'}</span>
          </div>

          {addingLead && (
            <LeadForm
              key={editingLead?.id ?? 'new'}
              lead={editingLead}
              onSave={handleSaveAndProcess}
              onCancel={() => { setAddingLead(false); setEditingLead(null) }}
            />
          )}
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
