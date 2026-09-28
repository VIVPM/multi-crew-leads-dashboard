// Visualizes lead totals, scores, industries, sources, and activity.
import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'

const COLORS = ['#533afd', '#ea2261', '#f96bee', '#665efd', '#1c1e54', '#9b6829', '#b9b9f9', '#4434d4']
const MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

function countBy(arr, key) {
  const map = {}
  arr.forEach(item => {
    const val = item[key] || 'Unknown'
    map[val] = (map[val] || 0) + 1
  })
  return Object.entries(map).map(([name, value]) => ({ name, value }))
}

function scoreHistogram(leads) {
  const scores = leads.map(l => l.score).filter(s => s != null)
  if (!scores.length) return []
  const BIN = 5
  const minBin = Math.floor(Math.min(...scores) / BIN) * BIN
  const maxBin = Math.ceil(Math.max(...scores) / BIN) * BIN
  const count = Math.max(1, (maxBin - minBin) / BIN)
  const buckets = Array.from({ length: count }, (_, i) => ({
    name: `${minBin + i * BIN}–${minBin + (i + 1) * BIN}`,
    count: 0,
  }))
  scores.forEach(s => {
    const idx = Math.min(Math.floor((s - minBin) / BIN), count - 1)
    buckets[idx].count++
  })

  const merged = []
  for (const b of buckets) {
    const prev = merged[merged.length - 1]
    if (prev && (b.count === 0 || prev.count === 0)) {
      prev.name = `${prev.name.split('–')[0]}–${b.name.split('–')[1]}`
      prev.count += b.count
    } else {
      merged.push({ ...b })
    }
  }
  return merged
}

function avgScoreByIndustry(leads) {
  const map = {}
  leads.forEach(l => {
    if (l.score == null) return
    const ind = l.industry || 'Unknown'
    if (!map[ind]) map[ind] = { sum: 0, count: 0 }
    map[ind].sum += l.score
    map[ind].count++
  })
  return Object.entries(map)
    .map(([name, { sum, count }]) => ({ name, avg: +(sum / count).toFixed(1) }))
    .sort((a, b) => a.avg - b.avg)
}

// Returns years containing dated leads, newest first.
function availableYears(leads) {
  const years = new Set(
    leads.filter(l => l.created_at).map(l => new Date(l.created_at).getFullYear())
  )
  return [...years].sort((a, b) => b - a)
}

// Builds a fixed January-to-December series for the selected year.
function leadsByMonth(leads, year) {
  const counts = Array(12).fill(0)
  leads.forEach(l => {
    if (!l.created_at) return
    const d = new Date(l.created_at)
    if (d.getFullYear() === year) counts[d.getMonth()]++
  })
  return MONTH_LABELS.map((name, i) => ({ name, count: counts[i] }))
}

// Averages cost and tokens per processed lead for each month of the selected year.
function usageByMonth(leads, year) {
  const months = MONTH_LABELS.map(name => ({ name, cost: 0, tokens: 0, n: 0 }))
  leads.forEach(l => {
    if (!l.processed_at || l.total_cost == null) return
    const d = new Date(l.processed_at)
    if (d.getFullYear() !== year) return
    const m = months[d.getMonth()]
    m.cost += l.total_cost
    m.tokens += l.total_tokens || 0
    m.n++
  })
  return months.map(m => ({
    name: m.name,
    cost: m.n ? +(m.cost / m.n).toFixed(4) : null,
    tokens: m.n ? Math.round(m.tokens / m.n) : null,
  }))
}

// Counts scored leads above and at-or-below the email cutoff, plus the borderline band.
function scoreBands(leads) {
  const scores = leads.map(l => l.score).filter(s => s != null)
  return [
    { name: 'Above 70', value: scores.filter(s => s > 70).length, fill: '#533afd' },
    { name: '70 or below', value: scores.filter(s => s <= 70).length, fill: '#ea2261' },
    { name: 'Borderline 65–75', value: scores.filter(s => s >= 65 && s <= 75).length, fill: '#9b6829' },
  ]
}

// Formats a token count compactly, e.g. 1.2M or 46K.
function compact(n) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${Math.round(n / 1e3)}K`
  return String(n)
}

function KpiCard({ label, value, sub }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value tnum">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

function countByCountry(leads) {
  const map = {}
  leads.forEach(l => {
    const loc = l.location || ''
    const country = loc.includes(',') ? loc.split(',').pop().trim() : (loc.trim() || 'Unknown')
    map[country] = (map[country] || 0) + 1
  })
  return Object.entries(map).map(([name, value]) => ({ name, value }))
}

function ChartCard({ title, extra, children }) {
  return (
    <div className="chart-card">
      <div className="chart-card-header">
        <h4 className="chart-title">{title}</h4>
        {extra}
      </div>
      {children}
    </div>
  )
}

function NoData() {
  return <p className="no-data">No data yet</p>
}

// Renders a pie chart with an external legend for long labels.
function LegendPie({ data }) {
  const total = data.reduce((sum, d) => sum + d.value, 0)
  const pct = value => `${Math.round((value / total) * 100)}%`
  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie data={data} dataKey="value" nameKey="name" outerRadius={65}>
          {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
        </Pie>
        <Tooltip formatter={value => pct(value)} />
        <Legend
          layout="horizontal"
          verticalAlign="bottom"
          wrapperStyle={{ fontSize: 11, lineHeight: '1.6' }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

function YearSelect({ years, value, onChange }) {
  if (years.length < 2) return null
  return (
    <select className="chart-year-select" value={value} onChange={e => onChange(Number(e.target.value))}>
      {years.map(y => <option key={y} value={y}>{y}</option>)}
    </select>
  )
}

export default function Dashboard({ leads }) {
  const [leadsYear, setLeadsYear] = useState(null)
  const [costYear, setCostYear] = useState(null)
  const [tokensYear, setTokensYear] = useState(null)

  if (!leads.length) {
    return (
      <div className="card">
        <p className="muted">No leads yet — add some leads to see analytics.</p>
      </div>
    )
  }

  const industryData = countBy(leads, 'industry')
  const sourceData = countBy(leads, 'source')
  const scoreData = scoreHistogram(leads)
  const avgData = avgScoreByIndustry(leads)
  const countryData = countByCountry(leads)

  const years = availableYears(leads)
  const resolve = y => (years.includes(y) ? y : (years[0] ?? new Date().getFullYear()))
  const timeData = leadsByMonth(leads, resolve(leadsYear))
  const costData = usageByMonth(leads, resolve(costYear))
  const tokenData = usageByMonth(leads, resolve(tokensYear))
  const bandData = scoreBands(leads)

  const processed = leads.filter(l => l.score != null)
  const costed = leads.filter(l => l.total_cost != null)
  const totalCost = costed.reduce((s, l) => s + l.total_cost, 0)
  const totalTokens = costed.reduce((s, l) => s + (l.total_tokens || 0), 0)
  const emailsDrafted = processed.filter(l => l.score > 70).length

  return (
    <div className="dashboard">
      <div className="kpi-grid">
        <KpiCard label="Leads processed" value={processed.length} sub={`of ${leads.length} added`} />
        <KpiCard
          label="Total cost"
          value={`$${totalCost.toFixed(2)}`}
          sub={costed.length === processed.length ? 'all processed leads' : `${costed.length} leads with usage data`}
        />
        <KpiCard
          label="Avg cost per lead"
          value={costed.length ? `$${(totalCost / costed.length).toFixed(4)}` : '—'}
          sub="research, scoring and email"
        />
        <KpiCard label="Tokens used" value={compact(totalTokens)} sub={costed.length ? `~${compact(Math.round(totalTokens / costed.length))} per lead` : null} />
        <KpiCard
          label="Emails drafted"
          value={emailsDrafted}
          sub={processed.length ? `${Math.round((emailsDrafted / processed.length) * 100)}% scored above 70` : null}
        />
      </div>

      <div className="chart-grid">
        <ChartCard title="Leads Over Time" extra={<YearSelect years={years} value={resolve(leadsYear)} onChange={setLeadsYear} />}>
          <ResponsiveContainer width="100%" height={230}>
            <LineChart data={timeData} margin={{ top: 5, left: 8, right: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
              <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-90} textAnchor="end" height={45} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip />
              <Line type="monotone" dataKey="count" stroke="#533afd" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Avg Cost per Lead by Month" extra={<YearSelect years={years} value={resolve(costYear)} onChange={setCostYear} />}>
          {costData.some(m => m.cost != null) ? (
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={costData} margin={{ top: 5, left: 8, right: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-90} textAnchor="end" height={45} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={v => `$${v}`} width={60} />
                <Tooltip formatter={v => [`$${v.toFixed(4)}`, 'Avg cost per lead']} />
                <Line type="monotone" dataKey="cost" stroke="#ea2261" strokeWidth={2} dot={{ r: 3 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Avg Tokens per Lead by Month" extra={<YearSelect years={years} value={resolve(tokensYear)} onChange={setTokensYear} />}>
          {tokenData.some(m => m.tokens != null) ? (
            <ResponsiveContainer width="100%" height={230}>
              <LineChart data={tokenData} margin={{ top: 5, left: 8, right: 10, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ee" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} angle={-90} textAnchor="end" height={45} />
                <YAxis tick={{ fontSize: 11 }} tickFormatter={compact} />
                <Tooltip formatter={v => [v.toLocaleString(), 'Avg tokens per lead']} />
                <Line type="monotone" dataKey="tokens" stroke="#665efd" strokeWidth={2} dot={{ r: 3 }} connectNulls />
              </LineChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Score Bands">
          {processed.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={bandData} margin={{ top: 10, right: 10 }}>
                <XAxis dataKey="name" tick={{ fontSize: 11 }} interval={0} />
                <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
                <Tooltip formatter={v => [v, 'Leads']} />
                <Bar dataKey="value" radius={[3, 3, 0, 0]}>
                  {bandData.map(b => <Cell key={b.name} fill={b.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads by Industry (Top 6)">
          {industryData.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={[...industryData].sort((a,b) => b.value - a.value).slice(0,6)} layout="vertical" margin={{ left: 10, right: 20 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} allowDecimals={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
                <Tooltip />
                <Bar dataKey="value" fill="#533afd" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads by Source">
          {sourceData.length ? <LegendPie data={sourceData} /> : <NoData />}
        </ChartCard>

        <ChartCard title="Score Distribution">
          {scoreData.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={scoreData} margin={{ bottom: 50 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-40} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#665efd" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Avg Score by Industry (Top 6)">
          {avgData.length ? (
            <ResponsiveContainer width="100%" height={210}>
              <BarChart data={[...avgData].sort((a,b) => b.avg - a.avg).slice(0,6)} layout="vertical" margin={{ left: 10, right: 30 }}>
                <XAxis type="number" tick={{ fontSize: 11 }} domain={[0, 100]} allowDecimals={false} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11 }} width={150} />
                <Tooltip formatter={v => `${v}`} />
                <Bar dataKey="avg" fill="#ea2261" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads by Country">
          {countryData.length ? <LegendPie data={countryData} /> : <NoData />}
        </ChartCard>
      </div>
    </div>
  )
}
