import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
  LineChart, Line, CartesianGrid,
} from 'recharts'

const COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#ec4899', '#14b8a6']

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
  return buckets
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

function leadsOverTime(leads) {
  const map = {}
  leads.forEach(l => {
    if (!l.created_at) return
    const day = l.created_at.slice(0, 10)
    map[day] = (map[day] || 0) + 1
  })
  return Object.entries(map).sort().map(([date, count]) => ({ date, count }))
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

function ChartCard({ title, children }) {
  return (
    <div className="chart-card">
      <h4 className="chart-title">{title}</h4>
      {children}
    </div>
  )
}

function NoData() {
  return <p className="no-data">No data yet</p>
}

export default function Dashboard({ leads }) {
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
  const timeData = leadsOverTime(leads)
  const avgData = avgScoreByIndustry(leads)
  const countryData = countByCountry(leads)

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
                <Bar dataKey="value" fill="#6366f1" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads by Source">
          {sourceData.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={sourceData} dataKey="value" nameKey="name" outerRadius={70} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false}>
                  {sourceData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Score Distribution">
          {scoreData.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={scoreData} margin={{ bottom: 50 }}>
                <XAxis dataKey="name" tick={{ fontSize: 10 }} angle={-40} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Bar dataKey="count" fill="#10b981" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads Over Time">
          {timeData.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={timeData} margin={{ bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} angle={-30} textAnchor="end" interval={0} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#6366f1" strokeWidth={2} dot={false} />
              </LineChart>
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
                <Bar dataKey="avg" fill="#f59e0b" radius={[0, 3, 3, 0]} />
              </BarChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>

        <ChartCard title="Leads by Country">
          {countryData.length ? (
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie data={countryData} dataKey="value" nameKey="name" outerRadius={70} label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`} labelLine={false}>
                  {countryData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          ) : <NoData />}
        </ChartCard>
      </div>
    </div>
  )
}
