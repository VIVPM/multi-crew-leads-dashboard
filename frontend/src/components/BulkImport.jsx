// Imports, validates, and processes batches of leads from CSV.
import { useState, useRef } from 'react'
import { api, friendlyError } from '../api'
import { toLeads, MAX_ROWS } from '../csv'

const STATUS_META = {
  pending:    { icon: '○', label: 'Queued' },
  processing: { icon: '',  label: 'Scoring…' },
  done:       { icon: '✓', label: 'Scored' },
  failed:     { icon: '✕', label: 'Failed' },
}

export default function BulkImport({ onImported, onMessage, processLead, canProcess, onNeedIcp, remaining, cap }) {
  const [leads, setLeads] = useState([])
  const [errors, setErrors] = useState([])
  const [fileName, setFileName] = useState('')
  const [running, setRunning] = useState(false)

  const [progress, setProgress] = useState(null)
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

  const overBudget = Number.isFinite(remaining) && leads.length > remaining

  async function handleStart() {
    if (!canProcess) { onNeedIcp(); return }
    if (overBudget) return
    setRunning(true)
    setErrors([])
    setProgress('creating')
    try {
      const { created, skipped } = await api('POST', '/leads/bulk', { leads })
      const dupeNote = skipped.length ? ` ${skipped.length} skipped as already imported.` : ''
      onImported()

      if (created.length === 0) {
        onMessage(`Nothing new to import — all ${skipped.length} row(s) are already in your leads.`)
        reset()
        return
      }

      const items = created.map(l => ({ name: l.name || l.company || 'Lead', status: 'pending' }))
      setProgress({ items: [...items], done: 0, total: items.length })

      let scored = 0, failed = 0
      for (let i = 0; i < created.length; i++) {
        items[i].status = 'processing'
        setProgress({ items: [...items], done: scored + failed, total: items.length })
        try {
          await processLead(created[i])
          items[i].status = 'done'
          scored++
        } catch {
          items[i].status = 'failed'
          failed++
        }
        setProgress({ items: [...items], done: scored + failed, total: items.length })
        onImported()
      }

      const parts = [`Imported ${created.length} lead(s).`]
      if (scored) parts.push(`Scored ${scored}.`)
      if (failed) parts.push(`${failed} couldn't be processed — saved unscored (retry with Edit → Save and process).`)
      onMessage(parts.join(' ') + dupeNote)
      reset()
    } catch (e) {
      setErrors([friendlyError(e)])
    } finally {
      setRunning(false)
    }
  }

  const active = progress && progress !== 'creating'
  const pct = active ? Math.round((progress.done / progress.total) * 100) : 0

  return (
    <div className="lead-form-body">
      <p className="muted">
        Upload a CSV with the columns <strong>Name, Company, Email</strong> (required) and
        optionally Job Title, Use Case, Industry, Location, Source. A file exported
        from this app imports back as-is. Up to {MAX_ROWS} rows parse, and an import
        must fit your daily credits{cap ? ` (${cap}/day, 1 credit scores 1 lead)` : ''}.
      </p>

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

      {fileName && !running && leads.length > 0 && !overBudget && (
        <div className="alert alert-success">
          {fileName}: {leads.length} valid lead(s) ready to import
          {errors.length ? ` (${errors.length} row(s) skipped)` : ''}.
        </div>
      )}

      {fileName && !running && leads.length > 0 && overBudget && (
        <div className="alert alert-error">
          {remaining === 0
            ? `You have no credits left today, but this file has ${leads.length} lead(s). Try again tomorrow, or process leads individually.`
            : `This file has ${leads.length} lead(s) but you have only ${remaining} credit${remaining === 1 ? '' : 's'} left today. Upload a file with at most ${remaining} lead${remaining === 1 ? '' : 's'} and process the rest tomorrow.`}
        </div>
      )}

      {progress === 'creating' && (
        <div className="alert alert-info processing-row">
          <span className="spinner" aria-hidden="true" />
          <span>Creating leads…</span>
        </div>
      )}

      {active && (
        <div className="alert alert-info bulk-progress">
          <div className="bulk-progress-head">
            Scoring {progress.done} of {progress.total} — keep this tab open.
          </div>
          <div className="bulk-progress-track">
            <div className="bulk-progress-bar" style={{ width: `${pct}%` }} />
          </div>
          <ul className="bulk-progress-list">
            {progress.items.map((it, i) => (
              <li key={i} className={`bulk-progress-item is-${it.status}`}>
                <span className="bulk-progress-icon">
                  {it.status === 'processing'
                    ? <span className="spinner spinner-sm" aria-hidden="true" />
                    : STATUS_META[it.status].icon}
                </span>
                <span className="bulk-progress-name">{it.name}</span>
                <span className="bulk-progress-state">{STATUS_META[it.status].label}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="form-actions">
        <button
          className="btn btn-success"
          onClick={handleStart}
          disabled={running || leads.length === 0 || overBudget}
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
