export default function Navbar({ onLogout, username }) {
  return (
    <header className="navbar">
      <div className="navbar-brand">
        <span className="navbar-logo">🎯</span>
        <span className="navbar-app-name">Sales Pipeline</span>
      </div>
      <div className="navbar-right">
        <span className="navbar-username">{username}</span>
        <button className="btn btn-outline btn-sm" onClick={onLogout}>Log out</button>
      </div>
    </header>
  )
}
