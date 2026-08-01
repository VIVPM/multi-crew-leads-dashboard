import { useState, useRef, useEffect } from 'react'

// Each page is a distinct view in App.jsx, switched by the menu.
const PAGES = [
  { id: 'add-lead', label: 'Add new lead' },
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'leads', label: 'Lead details' },
  { id: 'settings', label: 'Settings' },
]

function CreditsBadge({ credits }) {
  return (
    <span className="credits-badge">
      <strong>{credits.remaining}</strong> credit{credits.remaining === 1 ? '' : 's'}
      <span
        className="field-hint field-hint--end"
        tabIndex={0}
        role="img"
        aria-label={`${credits.remaining} of ${credits.cap} daily lead credits left. 1 credit scores 1 lead. Resets tomorrow.`}
      >
        i
        <span className="field-hint-pop" aria-hidden="true">
          <strong>{credits.remaining} of {credits.cap}</strong> daily lead credits left.
          <span className="field-hint-eg">1 credit scores 1 lead · resets tomorrow</span>
          Processing a lead costs 1 credit. A bulk import has to fit within your
          remaining credits — nothing is imported unscored.
        </span>
      </span>
    </span>
  )
}

export default function Navbar({ username, page, onNavigate, onLogout, credits }) {
  const [open, setOpen] = useState(false)
  const menuRef = useRef(null)

  // Close on outside click or Escape — standard menu behaviour.
  useEffect(() => {
    if (!open) return
    const onDocClick = e => { if (menuRef.current && !menuRef.current.contains(e.target)) setOpen(false) }
    const onEsc = e => { if (e.key === 'Escape') setOpen(false) }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onEsc)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onEsc)
    }
  }, [open])

  function go(id) { onNavigate(id); setOpen(false) }

  return (
    <header className="navbar">
      <div className="navbar-brand">
        <span className="navbar-logo">🎯</span>
        <span className="navbar-app-name">Sales Pipeline</span>
      </div>

      <div className="navbar-right">
        {credits && <CreditsBadge credits={credits} />}

        <div className="navbar-menu" ref={menuRef}>
          <button
            className={`menu-button ${open ? 'is-open' : ''}`}
            onClick={() => setOpen(o => !o)}
            aria-label="Menu"
            aria-haspopup="true"
            aria-expanded={open}
          >
            <span className="menu-bars" aria-hidden="true"><i /><i /><i /></span>
          </button>

          {open && (
            <div className="menu-dropdown" role="menu">
              <div className="menu-account">
                <span className="menu-account-label">Signed in as</span>
                <span className="menu-account-email">{username}</span>
              </div>
              <div className="menu-divider" />
              {PAGES.map(p => (
                <button
                  key={p.id}
                  role="menuitem"
                  className={`menu-item ${page === p.id ? 'is-active' : ''}`}
                  onClick={() => go(p.id)}
                >
                  {p.label}
                </button>
              ))}
              <div className="menu-divider" />
              <button role="menuitem" className="menu-item menu-item-danger" onClick={() => { setOpen(false); onLogout() }}>
                Log out
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  )
}
