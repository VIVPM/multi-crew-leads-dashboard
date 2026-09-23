// Parses and validates lead CSV imports.
export const MAX_ROWS = 200
const EMAIL_RE = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/

const HEADER_MAP = {
  'name': 'name',
  'job title': 'job_title', 'job_title': 'job_title',
  'company': 'company',
  'email': 'email',
  'use case': 'use_case', 'use_case': 'use_case',
  'industry': 'industry',
  'location': 'location',
  'source': 'source',
}

// Parses quoted fields, embedded commas or newlines, and escaped quotes.
export function parseCSV(text) {
  const rows = []
  let row = [], field = '', inQuotes = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (inQuotes) {
      if (c === '"') {
        if (text[i + 1] === '"') { field += '"'; i++ } else { inQuotes = false }
      } else field += c
    } else if (c === '"') inQuotes = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n') { row.push(field); rows.push(row); row = []; field = '' }
    else if (c !== '\r') field += c
  }
  if (field !== '' || row.length) { row.push(field); rows.push(row) }
  return rows.filter(r => r.some(cell => cell.trim() !== ''))
}

export function toLeads(text) {
  const rows = parseCSV(text)
  if (!rows.length) return { leads: [], errors: ['That file is empty.'] }

  const header = rows[0].map(h => h.trim().toLowerCase())
  const cols = header.map(h => HEADER_MAP[h] || null)
  for (const required of ['name', 'company', 'email']) {
    if (!cols.includes(required)) {
      return { leads: [], errors: [`Missing required column: "${required}". Expected headers: Name, Company, Email (plus optional Job Title, Use Case, Industry, Location, Source).`] }
    }
  }

  const leads = [], errors = []
  rows.slice(1).forEach((cells, i) => {
    const lead = {}
    cols.forEach((key, c) => { if (key) lead[key] = (cells[c] ?? '').trim() })
    const line = i + 2
    if (!lead.name || !lead.company || !lead.email) {
      errors.push(`Row ${line}: name, company and email are all required.`)
    } else if (!EMAIL_RE.test(lead.email)) {
      errors.push(`Row ${line}: "${lead.email}" is not a valid email.`)
    } else {
      if (!lead.source) lead.source = 'Website'
      leads.push(lead)
    }
  })
  if (leads.length > MAX_ROWS) errors.push(`Too many rows (${leads.length}). Import at most ${MAX_ROWS} at a time.`)
  return { leads, errors }
}
