import { useState, useRef } from 'react'
import { api, friendlyError } from '../api'
import { toLeads, MAX_ROWS } from '../csv'

// Matches MAX_LEADS_PER_REQUEST / DAILY_LEAD_CAP in backend.py — one job, run
// once, since a day's whole allowance fits in a single request now.
const BATCH_SIZE = 5

export default function BulkImport({ onImported, onMessage, processBatch, canProcess, onNeedIcp, remaining, cap = 5 }) {
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
      // Create every row first (creation is free — no LLM), then score only as
      // many as today's remaining credits allow. Creating rows we can't score
      // today isn't waste: they're saved and can be processed on later days.
      const { created, skipped } = await api('POST', '/leads/bulk', { leads })
      const dupeNote = skipped.length ? ` ${skipped.length} skipped as already imported.` : ''
      onImported()

      if (created.length === 0) {
        onMessage(`Nothing new to import — all ${skipped.length} row(s) are already in your leads.`)
        reset()
        return
      }

      // The server enforces the same cap, so a stale `remaining` just means the
      // one call fails safe (those rows stay saved, unscored).
      const budget = Number.isFinite(remaining) ? Math.max(0, remaining) : BATCH_SIZE
      const toProcess = created.slice(0, budget)
      const deferred = created.length - toProcess.length

      let scored = 0
      if (toProcess.length > 0) {
        setProgress({ done: 0, total: toProcess.length })
        try {
          await processBatch(toProcess)
          scored = toProcess.length
        } catch {
          /* cap race or processing error — leave them saved, unscored */
        }
        setProgress({ done: toProcess.length, total: toProcess.length })
        onImported()
      }

      const failedNow = toProcess.length - scored
      const parts = [`Imported ${created.length} lead(s).`]
      if (scored) parts.push(`Scored ${scored} today.`)
      if (deferred) parts.push(`${deferred} over today's ${cap}-credit limit — saved for tomorrow.`)
      if (failedNow) parts.push(`${failedNow} couldn't be processed — saved unscored.`)
      if (deferred || failedNow) parts.push('Retry with Edit → Save and process; re-importing skips them as duplicates.')
      onMessage(parts.join(' ') + dupeNote)
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
        from this app imports back as-is. Up to {MAX_ROWS} rows can be added; up to{' '}
        {cap} are scored per day (1 credit each), the rest saved for later.
      </p>

      {/* native file input is unstyleable across browsers — hide it and drive
          it from the label, which keeps the click/keyboard behaviour for free */}
      <label className={`file-picker ${running ? 'is-disabled' : ''}`}>
        <input
          ref={fileRef}
          type="file"
          accept=".csv,text/csv"
          onChange={handleFile}
          disabled={running}
        />
        <span className="file-picker-cta">Choose CSV file</span>
        <span className="file-picker-name">{fileName || 'No file selected'}</span>
      </label>

      {errors.map((e, i) => (
        <div key={i} className="alert alert-error">{e}</div>
      ))}

      {fileName && !running && leads.length > 0 && (
        <div className="alert alert-success">
          {fileName}: {leads.length} valid lead(s) ready to import
          {errors.length ? ` (${errors.length} row(s) skipped)` : ''}.
        </div>
      )}

      {/* Say upfront how many can actually be scored, rather than letting them
          click and find out afterwards. */}
      {fileName && !running && leads.length > 0 && Number.isFinite(remaining) && leads.length > remaining && (
        <div className="alert alert-info">
          {remaining === 0
            ? `No credits left today — all ${leads.length} will be saved unscored and can be processed tomorrow.`
            : `Only ${remaining} of these can be scored today (${remaining} credit${remaining === 1 ? '' : 's'} left); the other ${leads.length - remaining} will be saved unscored for tomorrow.`}
        </div>
      )}

      {running && (
        <div className="alert alert-info">
          {progress
            ? `Scoring ${progress.total} lead(s) — this can take a minute or two; keep this tab open.`
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
