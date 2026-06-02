import { useState, useEffect } from 'react'
import { api, friendlyError } from '../api'

const PAGE_SIZES = [10, 25, 50, 100]

function flattenToText(obj) {
  if (obj == null) return ''
  if (typeof obj === 'object') return Object.values(obj).map(flattenToText).join(' ').toLowerCase()
  return String(obj).toLowerCase()
}

function exportCSV(leads) {
  const cols = ['Name', 'Job Title', 'Company', 'Email', 'Use Case', 'Industry', 'Location', 'Source', 'Score']
  const keys = ['name', 'job_title', 'company', 'email', 'use_case', 'industry', 'location', 'source', 'score']
  const rows = leads.map(l => keys.map(k => `"${(l[k] ?? '').toString().replace(/"/g, '""')}"`).join(','))
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
          <h3 className="modal-title">📊 Analysis Results</h3>
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
              <table className="analysis-table">
                <thead>
                  <tr>
                    <th>Agent</th>
                    <th>Status</th>
                    <th>Tokens</th>
                    <th>Cost</th>
                  </tr>
                </thead>
                <tbody>
                  {(data.agents_data || []).map((agent, i) => (
                    <tr key={i}>
                      <td>{agent.agent}</td>
                      <td><span className="badge badge-green">✅ {agent.status}</span></td>
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

function LeadCard({ lead, onEdit, onDelete, onRefresh }) {
  const [open, setOpen] = useState(false)
  const [deleting, setDeleting] = useState(false)
  const [err, setErr] = useState(null)
  const [showAnalysis, setShowAnalysis] = useState(false)
  const score = lead.score != null ? ` • Score: ${lead.score}` : ''

  async function handleDelete() {
    if (!confirm(`Delete ${lead.name}?`)) return
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

  return (
    <div className="lead-card">
      <div className="lead-card-header" onClick={() => setOpen(o => !o)}>
        <span className="lead-card-title">
          <span className="lead-name">{lead.name}</span>
          {lead.company && <span className="lead-company"> — {lead.company}</span>}
          {score && <span className="lead-score">{score}</span>}
        </span>
        <span className="lead-card-chevron">{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div className="lead-card-body">
          {err && <div className="alert alert-error">{err}</div>}

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

          {lead.scoring_result && (
            <div className="lead-section">
              <div className="lead-section-title">Scoring Result</div>
              <pre className="lead-json">{JSON.stringify(lead.scoring_result, null, 2)}</pre>
            </div>
          )}

          {lead.email_draft && (
            <div className="lead-section">
              <div className="lead-section-title">Generated Email Draft</div>
              <pre className="lead-email">{lead.email_draft}</pre>
            </div>
          )}

          <div className="lead-card-actions">
            {lead.score == null ? (
              <button className="btn btn-sm btn-outline" onClick={() => onEdit(lead)}>✏️ Edit</button>
            ) : (
              <span className="badge badge-green">✅ Processed</span>
            )}
            <button className="btn btn-sm btn-danger" onClick={handleDelete} disabled={deleting}>
              {deleting ? 'Deleting…' : '🗑️ Delete'}
            </button>
            <button className="btn btn-sm btn-outline" onClick={onRefresh}>🔄 Refresh</button>
            <button
              className="btn btn-sm btn-outline"
              onClick={() => setShowAnalysis(true)}
              disabled={lead.score == null}
              title={lead.score == null ? 'Process this lead first to view analysis' : 'View analysis results'}
            >
              📊 Analysis
            </button>
          </div>
        </div>
      )}

      {showAnalysis && (
        <AnalysisModal leadId={lead.id} onClose={() => setShowAnalysis(false)} />
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
          placeholder="🔍 Search leads (name, company, email, score…)"
          value={search}
          onChange={e => { setSearch(e.target.value); setPage(0) }}
        />
        {filtered.length > 0 && (
          <button className="btn btn-outline" onClick={() => exportCSV(filtered)}>
            📥 Export CSV
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
          ⬅️ Previous
        </button>
        <button className="btn btn-outline btn-sm" disabled={safePage >= totalPages - 1} onClick={() => setPage(p => p + 1)}>
          Next ➡️
        </button>
      </div>
    </div>
  )
}
