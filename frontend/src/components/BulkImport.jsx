import { useState, useRef } from 'react'
import { api, friendlyError } from '../api'
import { toLeads, MAX_ROWS } from '../csv'

// Must match MAX_LEADS_PER_REQUEST in backend.py — processing is enqueued in
// chunks of this size, one job per chunk, run one after another so a big
// import can't stampede the shared Gemini/Tavily rate limits.
const BATCH_SIZE = 10

export default function BulkImport({ onImported, onMessage, processBatch, canProcess, onNeedIcp }) {
  const [leads, setLeads] = useState([])
  const [errors, setErrors] = useState([])
  const [fileName, setFileName] = useState('')
  const [running, setRunning] = useState(false)
  const [progress, setProgress] = useState(null) // { done, total, batch, batches }
  const fileRef = useRef(null)

  function handleFile(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setFileName(file.name)
    setProgress(null)
    const reader = new FileReader()
    reader.onload = () => {
      const { leads, errors } = toLeads(String(reader.result))
      setLeads(leads)
      setErrors(errors)
    }
    reader.onerror = () => { setLeads([]); setErrors(['Could not read that file.']) }
    reader.readAsText(file)
  }

  function reset() {
    setLeads([]); setErrors([]); setFileName(''); setProgress(null)
    if (fileRef.current) fileRef.current.value = ''
  }

  async function handleStart() {
    if (!canProcess) { onNeedIcp(); return }
    setRunning(true)
    setErrors([])
    try {
      // Create every row first, then process the created rows in batches —
      // /leads/process needs real lead ids and caps each call at BATCH_SIZE.
      const created = await api('POST', '/leads/bulk', { leads })
      onImported()
      const batches = []
      for (let i = 0; i < created.length; i += BATCH_SIZE) batches.push(created.slice(i, i + BATCH_SIZE))

      let done = 0
      let failed = 0
      for (let i = 0; i < batches.length; i++) {
        setProgress({ done, total: created.length, batch: i + 1, batches: batches.length })
        try {
          await processBatch(batches[i])
        } catch {
          failed += batches[i].length // keep going; the rest of the file still gets processed
        }
        done += batches[i].length
        setProgress({ done, total: created.length, batch: i + 1, batches: batches.length })
        onImported()
      }
      onMessage(
        failed
          ? `Imported ${created.length} lead(s); ${failed} could not be processed (they're saved — reprocess from the lead card).`
          : `Imported and processed ${created.length} lead(s).`
      )
      reset()
    } catch (e) {
      setErrors([friendlyError(e)])
    } finally {
      setRunning(false)
    }
  }

  const pct = progress ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <div className="lead-form-body">
      <p className="muted">
        Upload a CSV with the columns <strong>Name, Company, Email</strong> (required) and
        optionally Job Title, Use Case, Industry, Location, Source. A file exported
        from this app imports back as-is. Up to {MAX_ROWS} rows; they're processed{' '}
        {BATCH_SIZE} at a time.
      </p>

      <div className="form-group" style={{ marginTop: '0.75rem' }}>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFile}
          disabled={running}
        />
      </div>

      {errors.map((e, i) => (
        <div key={i} className="alert alert-error">{e}</div>
      ))}

      {fileName && !running && leads.length > 0 && (
        <div className="alert alert-success">
          {fileName}: {leads.length} valid lead(s) ready to import
          {errors.length ? ` (${errors.length} row(s) skipped)` : ''}.
        </div>
      )}

      {running && (
        <div className="alert alert-info">
          {progress
            ? `Processing batch ${progress.batch} of ${progress.batches} — ${progress.done} of ${progress.total} lead(s) done. This can take a few minutes per batch; keep this tab open.`
            : 'Creating leads…'}
          {progress && (
            <div className="bulk-progress-track">
              <div className="bulk-progress-bar" style={{ width: `${pct}%` }} />
            </div>
          )}
        </div>
      )}

      <div className="form-actions">
        <button
          className="btn btn-success"
          onClick={handleStart}
          disabled={running || leads.length === 0}
        >
          {running ? 'Importing…' : `Import & process${leads.length ? ` ${leads.length} lead(s)` : ''}`}
        </button>
        <button className="btn btn-outline" onClick={reset} disabled={running}>
          Clear
        </button>
      </div>
    </div>
  )
}
