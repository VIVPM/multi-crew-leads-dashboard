import { useState } from 'react'
import { api, friendlyError } from '../api'

export default function Auth({ onLogin, onBack, initialMode = 'login' }) {
  const [mode, setMode] = useState(initialMode) // 'login' | 'signup'
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [message, setMessage] = useState(null) // { type: 'error'|'success', text }
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setMessage(null)
    if (!username.trim() || !password.trim()) {
      setMessage({ type: 'error', text: 'Please fill in all fields.' })
      return
    }
    if (mode === 'signup') {
      if (username.trim().length < 3) {
        setMessage({ type: 'error', text: 'Username must be at least 3 characters.' })
        return
      }
      if (password.length < 8) {
        setMessage({ type: 'error', text: 'Password must be at least 8 characters.' })
        return
      }
    }
    setLoading(true)
    try {
      if (mode === 'signup') {
        await api('POST', '/auth/signup', { username, password })
        setMessage({ type: 'success', text: 'Signup successful. Please login.' })
        setMode('login')
        setUsername('')
        setPassword('')
      } else {
        const data = await api('POST', '/auth/login', { username, password })
        onLogin(data.user_id, data.username, data.token, data.is_admin)
      }
    } catch (err) {
      setMessage({ type: 'error', text: friendlyError(err) })
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        {onBack && (
          <button className="auth-back" onClick={onBack}>← Back to overview</button>
        )}
        <div className="auth-logo">🎯</div>
        <h1 className="auth-title">Sales Pipeline</h1>
        <p className="auth-subtitle">Lead Scoring &amp; Email Generation</p>

        <div className="auth-tabs">
          <button
            className={`auth-tab ${mode === 'login' ? 'active' : ''}`}
            onClick={() => { setMode('login'); setMessage(null) }}
          >Login</button>
          <button
            className={`auth-tab ${mode === 'signup' ? 'active' : ''}`}
            onClick={() => { setMode('signup'); setMessage(null) }}
          >Sign Up</button>
        </div>

        {message && (
          <div className={`alert alert-${message.type}`}>{message.text}</div>
        )}

        <form onSubmit={handleSubmit} className="auth-form">
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={e => setUsername(e.target.value)}
              placeholder="Enter username"
              autoFocus
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <input
              type="password"
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="Enter password"
            />
          </div>
          <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
            {loading ? 'Please wait…' : mode === 'login' ? 'Login' : 'Sign Up'}
          </button>
        </form>
      </div>
    </div>
  )
}
