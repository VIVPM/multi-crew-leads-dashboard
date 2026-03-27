import { useState, useCallback, useEffect } from 'react'
import Auth from './components/Auth'
import Sidebar from './components/Sidebar'
import LeadForm from './components/LeadForm'
import Dashboard from './components/Dashboard'
import LeadsTable from './components/LeadsTable'
import { api, friendlyError } from './api'
import './App.css'

const SESSION_KEY = 'sp_session'
const SESSION_TTL = 60 * 60 * 1000 // 1 hour in ms

function loadSession() {
  try {
    const raw = localStorage.getItem(SESSION_KEY)
    if (!raw) return null
    const { userId, username, expiresAt, geminiKey, tavilyKey } = JSON.parse(raw)
    if (Date.now() > expiresAt) { localStorage.removeItem(SESSION_KEY); return null }
    return { userId, username, geminiKey: geminiKey || '', tavilyKey: tavilyKey || '' }
  } catch { return null }
}

function saveSession(userId, username, geminiKey = '', tavilyKey = '') {
  localStorage.setItem(SESSION_KEY, JSON.stringify({
    userId, username, geminiKey, tavilyKey, expiresAt: Date.now() + SESSION_TTL,
  }))
}

function clearSession() {
  localStorage.removeItem(SESSION_KEY)
}

export default function App() {
  const saved = loadSession()
  const [loggedIn, setLoggedIn] = useState(!!saved)
  const [userId, setUserId] = useState(saved?.userId ?? null)
  const [username, setUsername] = useState(saved?.username ?? '')

  const [geminiKey, setGeminiKey] = useState(saved?.geminiKey ?? '')
  const [tavilyKey, setTavilyKey] = useState(saved?.tavilyKey ?? '')

  const [leads, setLeads] = useState([])
  const [leadsLoading, setLeadsLoading] = useState(false)
  const [addingLead, setAddingLead] = useState(false)
  const [editingLead, setEditingLead] = useState(null)
  const [globalMsg, setGlobalMsg] = useState(null)

  // --- Auth ---
  function handleLogin(uid, uname) {
    saveSession(uid, uname)
    setUserId(uid)
    setUsername(uname)
    setLoggedIn(true)
    fetchLeads(uid)
  }

  function handleLogout() {
    clearSession()
    setLoggedIn(false)
    setUserId(null)
    setUsername('')
    setLeads([])
    setAddingLead(false)
    setEditingLead(null)
  }

  useEffect(() => {
    if (saved) fetchLeads(saved.userId)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!loggedIn || !userId) return
    saveSession(userId, username, geminiKey, tavilyKey)
  }, [geminiKey, tavilyKey]) // eslint-disable-line react-hooks/exhaustive-deps

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

  // Save lead then immediately process it — called by LeadForm
  async function handleSaveAndProcess(fields, setStatus) {
    let savedLead
    setStatus('saving')
    try {
      if (editingLead) {
        savedLead = await api('PUT', `/leads/${editingLead.id}`, fields)
      } else {
        savedLead = await api('POST', '/leads', { ...fields, user_id: userId })
      }
    } catch (e) {
      setStatus(null)
      throw new Error(friendlyError(e))
    }

    setStatus('processing')
    try {
      await api('POST', '/leads/process', {
        leads: [savedLead],
        gemini_api_key: geminiKey,
        tavily_api_key: tavilyKey,
      })
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
    setGlobalMsg({ type: 'success', text: '✅ Lead saved, scored, and email drafted!' })
  }

  function handleEditLead(lead) {
    setEditingLead(lead)
    setAddingLead(true)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  if (!loggedIn) return <Auth onLogin={handleLogin} />

  const keysReady = !!(geminiKey && tavilyKey)

  return (
    <div className="app-layout">
      <Sidebar
        geminiKey={geminiKey}
        setGeminiKey={setGeminiKey}
        tavilyKey={tavilyKey}
        setTavilyKey={setTavilyKey}
        onLogout={handleLogout}
        username={username}
      />

      <main className="main-content">
        <div className="page-header">
          <h1 className="page-title">🎯 Sales Pipeline — Lead Scoring &amp; Email Generation</h1>
        </div>

        {globalMsg && (
          <div className={`alert alert-${globalMsg.type}`} style={{ marginBottom: '1rem' }}>
            {globalMsg.text}
            <button className="alert-close" onClick={() => setGlobalMsg(null)}>×</button>
          </div>
        )}

        {addingLead ? (
          <LeadForm
            lead={editingLead}
            onSave={handleSaveAndProcess}
            onCancel={() => { setAddingLead(false); setEditingLead(null) }}
            keysReady={keysReady}
          />
        ) : (
          <div className="lead-controls">
            <button className="btn btn-primary" onClick={() => { setEditingLead(null); setAddingLead(true) }}>
              ➕ Add New Lead
            </button>
          </div>
        )}

        <section className="section">
          <h2 className="section-title">📊 Leads Dashboard</h2>
          {leadsLoading ? <p className="muted">Loading leads…</p> : <Dashboard leads={leads} />}
        </section>

        <section className="section">
          <h2 className="section-title">📋 Leads Data</h2>
          {leadsLoading
            ? <p className="muted">Loading…</p>
            : <LeadsTable leads={leads} onEdit={handleEditLead} onRefresh={fetchLeads} />
          }
        </section>
      </main>
    </div>
  )
}
