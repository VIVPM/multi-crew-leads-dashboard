export default function Sidebar({ onLogout, username }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <span className="sidebar-logo">🎯</span>
        <div>
          <div className="sidebar-app-name">Sales Pipeline</div>
          <div className="sidebar-username">{username}</div>
        </div>
      </div>

      <div className="sidebar-footer">
        <button className="btn btn-outline btn-full" onClick={onLogout}>
          Log out
        </button>
      </div>
    </aside>
  )
}
