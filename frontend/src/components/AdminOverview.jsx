import { useState, useEffect } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'
import { api, friendlyError } from '../api'

const COLORS = ['#533afd', '#ea2261', '#f96bee', '#665efd', '#1c1e54', '#9b6829', '#b9b9f9', '#4434d4']
const STATUS_COLORS = { done: '#533afd', failed: '#ea2261', running: '#665efd', pending: '#b9b9f9' }

const sanitizeCell = s => (/^[=+\-@]/.test(s) ? `'${s}` : s) // neutralize spreadsheet formula injection

function ChartCard({ title, children }) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h4 className="chart-title">{title}</h4>
      </div>
      {children}
    </div>
  )
}

function jobsByDay(jobs) {
  const map = {}
  jobs.forEach(j => {
    const day = j.created_at.slice(0, 10) // YYYY-MM-DD
    map[day] = (map[day] || 0) + 1
  })
  return Object.entries(map)
    .sort(([a], [b]) => a.localeCompare(b))
    .slice(-30) // last 30 days that actually had activity
    .map(([name, count]) => ({ name, count }))
}

function jobsByStatus(jobs) {
  const map = {}
  jobs.forEach(j => { map[j.status] = (map[j.status] || 0) + 1 })
  return Object.entries(map).map(([name, value]) => ({ name, value }))
}

function exportUsersCSV(users) {
  const cols = ['Username', 'Leads', 'Qualified', 'Jobs Done', 'Jobs Failed', 'Jobs In Flight', 'Tokens', 'Cost', 'Last Activity']
  const keys = ['username', 'leads_total', 'leads_qualified', 'jobs_done', 'jobs_failed', 'jobs_in_flight', 'total_tokens', 'total_cost', 'last_activity']
  const rows = users.map(u => keys.map(k => `"${sanitizeCell((u[k] ?? '').toString()).replace(/"/g, '""')}"`).join(','))
  const csv = [cols.join(','), ...rows].join('\n')
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = 'admin_overview_export.csv'; a.click()
  URL.revokeObjectURL(url)
}

function SortableTh({ label, sortKey, currentKey, dir, onSort }) {
  const active = sortKey === currentKey
  return (
    <th onClick={() => onSort(sortKey)} style={{ cursor: 'pointer', userSelect: 'none' }}>
      {label}{active ? (dir === 'asc' ? ' ▲' : ' ▼') : ''}
    </th>
  )
}

function UserLeadsModal({ user, onClose }) {
  const [leads, setLeads] = useState(null)
  const [err, setErr] = useState(null)

  useEffect(() => {
    api('GET', `/admin/users/${user.user_id}/leads`)
      .then(setLeads)
      .catch(e => setErr(friendlyError(e)))
  }, [user.user_id])

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-container" onClick={e => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">{user.username}&rsquo;s leads</h3>
          <button className="modal-close" onClick={onClose}>×</button>
        </div>
        <div className="modal-body">
          {err && <div className="alert alert-error">{err}</div>}
          {!leads && !err && <p className="muted">Loading…</p>}
          {leads && (
            <table className="analysis-table">
              <thead>
                <tr>
                  <th>Name</th><th>Company</th><th>Job Title</th><th>Score</th><th>Created</th>
                </tr>
              </thead>
              <tbody>
                {leads.map(l => (
                  <tr key={l.id}>
                    <td>{l.name}</td>
                    <td>{l.company}</td>
                    <td>{l.job_title || '—'}</td>
                    <td>{l.score ?? '—'}</td>
                    <td>{new Date(l.created_at).toLocaleString()}</td>
                  </tr>
                ))}
                {!leads.length && <tr><td colSpan={5} className="muted">No leads yet.</td></tr>}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  )
}

export default function AdminOverview() {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState('last_activity')
  const [sortDir, setSortDir] = useState('desc')
  const [drillUser, setDrillUser] = useState(null)

  useEffect(() => {
    api('GET', '/admin/overview')
      .then(setData)
      .catch(e => setErr(friendlyError(e)))
  }, [])

  if (err) return <div className="alert alert-error">{err}</div>
  if (!data) return <p className="muted">Loading admin overview…</p>

  // PostgREST can serialize `numeric` SQL columns (total_cost) as JSON
  // strings rather than numbers, to avoid float precision loss — coerce
  // once here so every downstream .toFixed()/sort/sum call is safe.
  const users = (data.users || []).map(u => ({
    ...u, total_tokens: Number(u.total_tokens) || 0, total_cost: Number(u.total_cost) || 0,
  }))
  const jobs = data.jobs
  const totals = users.reduce((acc, u) => ({
    leads: acc.leads + u.leads_total,
    qualified: acc.qualified + u.leads_qualified,
    tokens: acc.tokens + u.total_tokens,
    cost: acc.cost + u.total_cost,
    failed: acc.failed + u.jobs_failed,
    inFlight: acc.inFlight + u.jobs_pending + u.jobs_running,
  }), { leads: 0, qualified: 0, tokens: 0, cost: 0, failed: 0, inFlight: 0 })

  const topUsers = [...users]
    .filter(u => u.leads_total > 0)
    .sort((a, b) => b.leads_total - a.leads_total)
    .slice(0, 10)
    .map(u => ({ name: u.username, value: u.leads_total }))

  const statusData = jobsByStatus(jobs)
  const trendData = jobsByDay(jobs)

  const withInFlight = users.map(u => ({ ...u, jobs_in_flight: u.jobs_pending + u.jobs_running }))
  const filtered = search.trim()
    ? withInFlight.filter(u => u.username.toLowerCase().includes(search.trim().toLowerCase()))
    : withInFlight
  const tableRows = [...filtered].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey]
    const cmp = typeof av === 'string' ? av.localeCompare(bv) : av - bv
    return sortDir === 'asc' ? cmp : -cmp
  })

  function handleSort(key) {
    if (sortKey === key) setSortDir(d => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <>
      <section className="section">
        <h2 className="section-title">Admin overview</h2>

        <div className="analysis-metrics">
          <div className="metric-item">
            <div className="metric-label">Users</div>
            <div className="metric-value">{users.length}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Leads processed</div>
            <div className="metric-value">{totals.leads}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Qualified</div>
            <div className="metric-value">{totals.qualified}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Jobs in flight</div>
            <div className="metric-value">{totals.inFlight}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Jobs failed</div>
            <div className="metric-value">{totals.failed}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Total tokens</div>
            <div className="metric-value">{totals.tokens.toLocaleString()}</div>
          </div>
          <div className="metric-item">
            <div className="metric-label">Total cost</div>
            <div className="metric-value">${totals.cost.toFixed(4)}</div>
          </div>
        </div>

        {totals.failed > 0 && (
          <div className="alert alert-warning" style={{ marginBottom: '1rem' }}>
            {totals.failed} job(s) have failed across all users — worth checking worker logs.
          </div>
        )}

        <div className="chart-grid">
          <ChartCard title="Jobs by Status (all users)">
            {statusData.length ? (
              <ResponsiveContainer width="100%" height={210}>
                <PieChart>
                  <Pie
                    data={statusData}
                    dataKey="value"
                    nameKey="name"
                    outerRadius={65}
                    label={({ value }) => `${Math.round((value / jobs.length) * 100)}%`}
                    labelLine={false}
                  >
                    {statusData.map((d, i) => (
                      <Cell key={i} fill={STATUS_COLORS[d.name] || COLORS[i % COLORS.length]} />
                    ))}
                  </Pie>
                  <Legend layout="horizontal" verticalAlign="bottom" wrapperStyle={{ fontSize: 11, lineHeight: '1.6' }} />
                </PieChart>
              </ResponsiveContainer>
            ) : <p className="no-data">No jobs yet</p>}
          </ChartCard>

          <ChartCard title="Leads Processed by User (Top 10)">
            {topUsers.length ? (
              <ResponsiveContainer width="100%" height={210}>
                <BarChart data={topUsers} layout="vertical" margin={{ left: 10, right: 20 }}>
                  <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={140} />
                  <Tooltip />
                  <Bar dataKey="value" fill="#533afd" radius={[0, 3, 3, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : <p className="no-data">No leads yet</p>}
          </ChartCard>

          <ChartCard title="Jobs per Day (last 30 active days)">
            {trendData.length ? (
              <ResponsiveContainer width="100%" height={230}>
                <LineChart data={trendData} margin={{ top: 5, left: 8, right: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 10 }}
                    interval={Math.max(0, Math.ceil(trendData.length / 10) - 1)}
                    angle={-45}
                    textAnchor="end"
                    height={55}
                  />
                  <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="count" stroke="#533afd" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : <p className="no-data">No jobs yet</p>}
          </ChartCard>
        </div>
      </section>

      <section className="section">
        <div className="chart-card-header" style={{ marginBottom: '0.75rem' }}>
          <h2 className="section-title" style={{ marginBottom: 0 }}>Per-user detail</h2>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <input
              type="text"
              placeholder="Search username…"
              value={search}
              onChange={e => setSearch(e.target.value)}
              style={{ fontSize: '0.8rem', padding: '0.4rem 0.6rem' }}
            />
            <button className="btn btn-outline" onClick={() => exportUsersCSV(tableRows)}>
              Export CSV
            </button>
          </div>
        </div>
        <div className="card">
          <table className="analysis-table">
            <thead>
              <tr>
                <SortableTh label="User" sortKey="username" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Leads" sortKey="leads_total" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Qualified" sortKey="leads_qualified" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Jobs done" sortKey="jobs_done" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Jobs failed" sortKey="jobs_failed" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Jobs in flight" sortKey="jobs_in_flight" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Tokens" sortKey="total_tokens" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Cost" sortKey="total_cost" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
                <SortableTh label="Last activity" sortKey="last_activity" currentKey={sortKey} dir={sortDir} onSort={handleSort} />
              </tr>
            </thead>
            <tbody>
              {tableRows.map(u => (
                <tr key={u.user_id} onClick={() => setDrillUser(u)} style={{ cursor: 'pointer' }}>
                  <td>{u.username}</td>
                  <td>{u.leads_total}</td>
                  <td>{u.leads_qualified}</td>
                  <td>{u.jobs_done}</td>
                  <td>{u.jobs_failed}</td>
                  <td>{u.jobs_in_flight}</td>
                  <td>{u.total_tokens.toLocaleString()}</td>
                  <td>${u.total_cost.toFixed(4)}</td>
                  <td>{new Date(u.last_activity).toLocaleString()}</td>
                </tr>
              ))}
              {!tableRows.length && (
                <tr><td colSpan={9} className="muted">{search ? 'No matching users.' : 'No users yet.'}</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {drillUser && <UserLeadsModal user={drillUser} onClose={() => setDrillUser(null)} />}
    </>
  )
}
