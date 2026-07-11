import { useState } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'

// Stripe design system: chart colors come from the documented gradient stops only
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

  // Collapse consecutive empty buckets into one wide range so the chart stays
  // continuous (no gaps hidden) without a wall of empty 5-point bars.
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

// Years that actually have at least one dated lead, newest first.
function availableYears(leads) {
  const years = new Set(
    leads.filter(l => l.created_at).map(l => new Date(l.created_at).getFullYear())
  )
  return [...years].sort((a, b) => b - a)
}

// All 12 months always present (0 where there's no data), so the chart
// shape stays the same Jan-Dec regardless of which months have leads.
function leadsByMonth(leads, year) {
  const counts = Array(12).fill(0)
  leads.forEach(l => {
    if (!l.created_at) return
    const d = new Date(l.created_at)
    if (d.getFullYear() === year) counts[d.getMonth()]++
  })
  return MONTH_LABELS.map((name, i) => ({ name, count: counts[i] }))
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

// Legend below the pie instead of labels on the slices — slice labels
// clipped or overlapped for longer names (countries, sources); a legend
// stays readable no matter how long the name or how thin the slice.
function LegendPie({ data }) {
  const total = data.reduce((sum, d) => sum + d.value, 0)
  const pct = value => `${Math.round((value / total) * 100)}%`
  return (
    <ResponsiveContainer width="100%" height={220}>
      <PieChart>
        <Pie
          data={data}
          dataKey="value"
          nameKey="name"
          outerRadius={65}
          label={({ value }) => pct(value)}
          labelLine={false}
        >
          {data.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
        </Pie>
        <Legend
          layout="horizontal"
          verticalAlign="bottom"
          wrapperStyle={{ fontSize: 11, lineHeight: '1.6' }}
        />
      </PieChart>
    </ResponsiveContainer>
  )
}

export default function Dashboard({ leads }) {
  const [selectedYear, setSelectedYear] = useState(null)

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
  const activeYear = years.includes(selectedYear) ? selectedYear : (years[0] ?? new Date().getFullYear())
  const timeData = leadsByMonth(leads, activeYear)

  return (
    <div className="dashboard">
      <div className="chart-grid">
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

        <ChartCard
          title="Leads Over Time"
          extra={
            years.length > 1 && (
              <select
                className="chart-year-select"
                value={activeYear}
                onChange={e => setSelectedYear(Number(e.target.value))}
              >
                {years.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            )
          }
        >
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
