import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

const PAGE_SIZES = [10, 25, 50, 100]

// Repeat scoring of the same lead varies by ~+/-3.5 points, so a lead in this
// band could land either side of the cutoff on a different run.
const EMAIL_THRESHOLD = 70
const BORDERLINE_LOW = 65
const BORDERLINE_HIGH = 75
const BORDERLINE_HINT =
  `Scored close to the ${EMAIL_THRESHOLD} cutoff, and repeat runs vary by a few points — ` +
  `this lead could qualify or not depending on the run. Worth reviewing by hand.`

function flattenToText(obj) {
  if (obj == null) return ''
  if (typeof obj === 'object') return Object.values(obj).map(flattenToText).join(' ').toLowerCase()
  return String(obj).toLowerCase()
}

function exportCSV(leads) {
  const cols = ['Name', 'Job Title', 'Company', 'Email', 'Use Case', 'Industry', 'Location', 'Source', 'Score']
  const keys = ['name', 'job_title', 'company', 'email', 'use_case', 'industry', 'location', 'source', 'score']
  const sanitize = s => (/^[=+\-@]/.test(s) ? `'${s}` : s) // neutralize spreadsheet formula injection
  const rows = leads.map(l => keys.map(k => `"${sanitize((l[k] ?? '').toString()).replace(/"/g, '""')}"`).join(','))
  const csv = [cols.join(','), ...rows].join('\n')
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'leads_export.csv'; a.click()
  URL.revokeObjectURL(url)
}

function AnalysisModal({ leadId, onClose }) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState(null)

  useEffect(() => {
    api('GET', `/analysis/${leadId}`)
      .then(setData)
      .catch(e => setErr(friendlyError(e)))
      .finally(() => setLoading(false))
  }, [leadId])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">Analysis results</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          {loading && <p className="muted">Loading analysis…</p>}
          {err && <div className="alert alert-error">{err}</div>}
          {data && (
            <>
              <div className="analysis-metrics">
                <div className="metric-item">
                  <div className="metric-label">Total Cost</div>
                  <div className="metric-value">
                    {data.total_cost != null ? `$${data.total_cost.toFixed(4)}` : '—'}
                  </div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Total Tokens</div>
                  <div className="metric-value">
                    {data.total_tokens != null ? data.total_tokens.toLocaleString() : '—'}
                  </div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Duration</div>
                  <div className="metric-value">
                    {data.duration_seconds != null ? `${data.duration_seconds}s` : '—'}
                  </div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Success Rate</div>
                  <div className="metric-value">
                    {data.success_rate != null ? `${data.success_rate}%` : '—'}
                  </div>
                </div>
                <div className="metric-item">
                  <div className="metric-label">Agents Executed</div>
                  <div className="metric-value">
                    {data.agents_executed != null ? `${data.agents_executed}/${data.agents_executed}` : '—'}
                  </div>
                </div>
              </div>

              <h4 className="analysis-section-title">Agent Performance Breakdown</h4>
              <p className="muted" style={{ fontSize: '0.8rem' }}>
                Token usage is measured per agent, straight from each step of the
                pipeline — every row below is an exact figure, not a share of a
                larger total.
              </p>
              <table className="analysis-table">
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Status</th>
                    <th>Time</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.agents_data || []).map((agent, i) => (
                    <tr key={i}>
                      <td>{agent.agent}</td>
                      <td><span className={`badge ${agent.status === 'Skipped' ? 'badge-neutral' : 'badge-green'}`}>{agent.status}</span></td>
                      <td>{agent.time_seconds != null ? `${agent.time_seconds}s` : '—'}</td>
                      <td>{agent.tokens != null ? agent.tokens.toLocaleString() : '—'}</td>
                      <td>{agent.cost != null ? `$${agent.cost.toFixed(6)}` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function DeleteConfirmModal({ name, onConfirm, onCancel }) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-container modal-sm" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">Delete lead</h3>
          <button className="modal-close" onClick={onCancel}>×</button>
        </div>
        <div className="modal-body">
          <p className="delete-confirm-msg">
            Are you sure you want to delete <strong>{name}</strong>? This cannot be undone.
          </p>
          <div className="delete-confirm-actions">
            <button className="btn btn-outline" onClick={onCancel}>Cancel</button>
            <button className="btn btn-danger-solid" onClick={onConfirm}>Delete</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function SendConfirmModal({ name, onConfirm, onCancel }) {
  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-container modal-sm" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">Send email</h3>
          <button className="modal-close" onClick={onCancel}>×</button>
        </div>
        <div className="modal-body">
          <p className="delete-confirm-msg">
            Send this email to <strong>{name}</strong> now?
          </p>
          <div className="delete-confirm-actions">
            <button className="btn btn-outline" onClick={onCancel}>Cancel</button>
            <button className="btn btn-primary" onClick={onConfirm}>Send</button>
          </div>
        </div>
      </div>
    </div>
  )
}

function LeadCard({ lead, onEdit, onDelete, onRefresh }) {
  const [open, setOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [err, setErr] = useState(null)
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false)
  const [editingEmail, setEditingEmail] = useState(false)
  const [emailDraft, setEmailDraft] = useState('')
  const [savingEmail, setSavingEmail] = useState(false)
  const [sending, setSending] = useState(false)
  const [showSendConfirm, setShowSendConfirm] = useState(false)
  const [sentMsg, setSentMsg] = useState(null)
  // Heavy fields aren't in the list response; fetched when the card expands
  const [detail, setDetail] = useState(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const score = lead.score != null ? ` • Score: ${lead.score}` : ''
  const isBorderline =
    lead.score != null && lead.score >= BORDERLINE_LOW && lead.score <= BORDERLINE_HIGH

  // detailLoading must not be a dependency — it would re-run this effect and
  // cancel the in-flight request, leaving the card stuck on "Loading…".
  useEffect(() => {
    if (!open || detail) return
    let cancelled = false
    setDetailLoading(true)
    api('GET', `/leads/${lead.id}/detail`)
      .then(d => {
        if (cancelled) return
        setDetail(d)
        setEmailDraft(d.email_draft || '')
      })
      .catch(e => { if (!cancelled) setErr(friendlyError(e)) })
      .finally(() => { if (!cancelled) setDetailLoading(false) })
    return () => { cancelled = true }
  }, [open, detail, lead.id])

  async function handleDelete() {
    setShowDeleteConfirm(false)
    setDeleting(true)
    try {
      await api('DELETE', `/leads/${lead.id}`)
      onDelete()
    } catch (e) {
      setErr(friendlyError(e))
    } finally {
      setDeleting(false)
    }
  }

  async function handleSaveEmail() {
    setSavingEmail(true)
    setErr(null)
    try {
      await api('PUT', `/leads/${lead.id}`, { email_draft: emailDraft })
      setEditingEmail(false)
      // The list response has no email_draft, so keep the fetched detail in sync
      setDetail(d => ({ ...(d || {}), email_draft: emailDraft }))
      onRefresh()
    } catch (e) {
      setErr(friendlyError(e))
    } finally {
      setSavingEmail(false)
    }
  }

  async function handleSendEmail() {
    setShowSendConfirm(false)
    setSending(true)
    setErr(null)
    try {
      await api('POST', `/leads/${lead.id}/send-email`)
      onRefresh()
      setSentMsg(`Email sent successfully on ${new Date().toLocaleString()}`)
      setTimeout(() => setSentMsg(null), 5000)
    } catch (e) {
      setErr(friendlyError(e))
    } finally {
      setSending(false)
    }
  }

  return (
    <div className="lead-card">
      <div className="lead-card-header" onClick={() => setOpen(o => !o)}>
        <span className="lead-card-title">
          <span className="lead-name">{lead.name}</span>
          {lead.company && <span className="lead-company"> — {lead.company}</span>}
          {score && <span className="lead-score">{score}</span>}
          {isBorderline && (
            <span className="badge badge-amber" title={BORDERLINE_HINT}>Borderline</span>
          )}
        </span>
        <span className="lead-card-chevron">{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="lead-card-body">
          {err && <div className="alert alert-error">{err}</div>}
          {isBorderline && (
            <div className="alert alert-warning">
              <strong>Borderline ({lead.score}).</strong> This sits within a few points of
              the {EMAIL_THRESHOLD} cutoff, and re-scoring the same lead moves the number by
              a few points — so whether an email gets drafted is partly luck of the run.
              {detail?.email_draft
                ? ' An email was drafted; read it before sending.'
                : ' No email was drafted — if this lead looks worth it, use Edit → Save and process, or write to them directly.'}
            </div>
          )}
          {sentMsg && (
            <div className="alert alert-success">
              {sentMsg}
              <button className="alert-close" onClick={() => setSentMsg(null)}>×</button>
            </div>
          )}

          <div className="lead-details-grid">
            {[
              ['Job Title', lead.job_title],
              ['Email', lead.email],
              ['Use Case', lead.use_case],
              ['Industry', lead.industry],
              ['Location', lead.location],
              ['Source', lead.source],
            ].map(([label, val]) => val ? (
              <div key={label} className="lead-detail">
                <span className="detail-label">{label}:</span>
                <span className="detail-val">{val}</span>
              </div>
            ) : null)}
          </div>

          {detailLoading && <p className="muted">Loading scoring result and email draft…</p>}

          {detail?.scoring_result && (
            <div className="lead-section">
              <div className="lead-section-title">Scoring result</div>
              <pre className="lead-json">{JSON.stringify(detail.scoring_result, null, 2)}</pre>
            </div>
          )}

          {detail?.email_draft && (
            <div className="lead-section">
              <div className="lead-section-header">
                <div className="lead-section-title">Email draft</div>
                {!editingEmail && (
                  <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
                    {lead.email_sent_at ? (
                      <span className="badge badge-green">
                        Sent {new Date(lead.email_sent_at).toLocaleDateString()}
                      </span>
                    ) : (
                      <button className="btn btn-sm btn-outline" onClick={() => setShowSendConfirm(true)} disabled={sending}>
                        {sending ? 'Sending…' : 'Send'}
                      </button>
                    )}
                    <button className="btn btn-sm btn-outline" onClick={() => setEditingEmail(true)}>Edit</button>
                  </div>
                )}
              </div>
              {editingEmail ? (
                <>
                  <textarea
                    className="lead-email-edit"
                    value={emailDraft}
                    onChange={e => setEmailDraft(e.target.value)}
                    rows={10}
                    disabled={savingEmail}
                  />
                  <div className="lead-email-edit-actions">
                    <button className="btn btn-sm btn-primary" onClick={handleSaveEmail} disabled={savingEmail}>
                      {savingEmail ? 'Saving…' : 'Save'}
                    </button>
                    <button
                      className="btn btn-sm btn-outline"
                      onClick={() => { setEditingEmail(false); setEmailDraft(detail?.email_draft || '') }}
                      disabled={savingEmail}
                    >
                      Cancel
                    </button>
                  </div>
                </>
              ) : (
                <pre className="lead-email">{detail.email_draft}</pre>
              )}
            </div>
          )}

          <div className="lead-card-actions">
            {/* Edit stays available after processing: re-scoring a borderline
                lead can cross the cutoff on the next run (see the borderline
                note above, which tells the user to do exactly this). The
                "Processed" badge sits first, with Edit beside it. */}
            {lead.score != null && <span className="badge badge-green">Processed</span>}
            <button className="btn btn-sm btn-outline" onClick={() => onEdit(lead)}>Edit</button>
            <button className="btn btn-sm btn-danger" onClick={() => setShowDeleteConfirm(true)} disabled={deleting}>
              {deleting ? 'Deleting…' : 'Delete'}
            </button>
            <button className="btn btn-sm btn-outline" onClick={onRefresh}>Refresh</button>
            <button
              className="btn btn-sm btn-outline"
              onClick={() => setShowAnalysis(true)}
              disabled={lead.score == null}
              title={lead.score == null ? 'Process this lead first to view analysis' : 'View analysis results'}
            >
              Analysis
            </button>
          </div>
        </div>
      )}

      {showAnalysis && (
        <AnalysisModal leadId={lead.id} onClose={() => setShowAnalysis(false)} />
      )}
      {showSendConfirm && (
        <SendConfirmModal
          name={lead.name}
          onConfirm={handleSendEmail}
          onCancel={() => setShowSendConfirm(false)}
        />
      )}
      {showDeleteConfirm && (
        <DeleteConfirmModal
          name={lead.name}
          onConfirm={handleDelete}
          onCancel={() => setShowDeleteConfirm(false)}
        />
      )}
    </div>
  )
}

export default function LeadsTable({ leads, onEdit, onRefresh }) {
  const [search, setSearch] = useState('')
  const [pageSize, setPageSize] = useState(10)
  const [page, setPage] = useState(0)

  const tokens = search.split(' ').map(t => t.trim().toLowerCase()).filter(Boolean)
  const filtered = tokens.length
    ? leads.filter(l => tokens.every(t => flattenToText(l).includes(t)))
    : leads

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize))
  const safePage = Math.min(page, totalPages - 1)
  const start = safePage * pageSize
  const paginated = filtered.slice(start, start + pageSize)

  function handlePageSize(e) {
    setPageSize(Number(e.target.value))
    setPage(0)
  }

  return (
    <div>
      <div className="table-controls">
        <input
          className="search-input"
          placeholder="Search leads (name, company, email, score…)"
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0) }}
        />
        {filtered.length > 0 && (
          <button className="btn btn-outline" onClick={() => exportCSV(filtered)}>
            Export CSV
          </button>
        )}
      </div>

      <div className="table-meta">
        <span className="muted">
          Showing {start + 1}–{Math.min(start + pageSize, filtered.length)} of {filtered.length} lead(s) | Page {safePage + 1} of {totalPages}
        </span>
        <select value={pageSize} onChange={handlePageSize} className="page-size-select">
          {PAGE_SIZES.map(s => <option key={s} value={s}>{s} per page</option>)}
        </select>
      </div>

      <div className="leads-list">
        {paginated.length > 0
          ? paginated.map(lead => (
            <LeadCard
              key={lead.id}
              lead={lead}
              onEdit={onEdit}
              onDelete={() => onRefresh()}
              onRefresh={() => onRefresh()}
            />
          ))
          : <p className="muted">{search ? 'No leads match your search.' : 'No leads yet.'}</p>
        }
      </div>

      <div className="pagination-controls">
        <button className="btn btn-outline btn-sm" disabled={safePage === 0} onClick={() => setPage(p => p - 1)}>
          ← Previous
        </button>
        <button className="btn btn-outline btn-sm" disabled={safePage >= totalPages - 1} onClick={() => setPage(p => p + 1)}>
          Next →
        </button>
      </div>
    </div>
  )
}
