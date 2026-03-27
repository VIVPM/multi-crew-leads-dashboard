export default function Sidebar({ geminiKey, setGeminiKey, tavilyKey, setTavilyKey, onLogout, username }) {
  const missing = [
    !geminiKey && 'Gemini',
    !tavilyKey && 'Tavily',
  ].filter(Boolean)

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-logo">🎯</span>
        <div>
          <div className="sidebar-app-name">Sales Pipeline</div>
          <div className="sidebar-username">{username}</div>
        </div>
      </div>

      <div className="sidebar-section">
        <h3 className="sidebar-section-title">🔑 API Keys</h3>

        <div className="form-group">
          <label>Gemini API Key</label>
          <input
            type="password"
            value={geminiKey}
            onChange={e => setGeminiKey(e.target.value)}
            placeholder="AIza-…"
          />
          <a
            href="https://aistudio.google.com/app/apikey"
            target="_blank"
            rel="noreferrer"
            className="sidebar-link"
          >Get a Gemini API key →</a>
        </div>

        <div className="form-group">
          <label>Tavily API Key</label>
          <input
            type="password"
            value={tavilyKey}
            onChange={e => setTavilyKey(e.target.value)}
            placeholder="tvly-…"
          />
          <a
            href="https://app.tavily.com"
            target="_blank"
            rel="noreferrer"
            className="sidebar-link"
          >Get a Tavily API key →</a>
        </div>

        {missing.length > 0 && (
          <div className="alert alert-warning">
            Enter your {missing.join(' & ')} API key(s) to use the crew.
          </div>
        )}
      </div>

      <div className="sidebar-footer">
        <button className="btn btn-outline btn-full" onClick={onLogout}>
          🚪 Log Out
        </button>
      </div>
    </aside>
  )
}
