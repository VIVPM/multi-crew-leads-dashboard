import { useEffect, useRef, useState } from 'react'

const AGENTS = [
  { name: 'Lead Data Specialist', job: 'researches the person and company on the live web' },
  { name: 'Cultural Fit Analyst', job: 'checks alignment with your ideal customer profile' },
  { name: 'Lead Scorer & Validator', job: 'assigns a validated 0–100 score with reasoning' },
  { name: 'Email Content Writer', job: 'drafts a personalized outreach email' },
  { name: 'Engagement Strategist', job: 'sharpens the call to action' },
]

const STEPS = [
  {
    n: '1',
    title: 'Add a lead',
    body: 'Name, company, email, and what they came asking about. That form is the only manual work in the whole flow.',
  },
  {
    n: '2',
    title: 'The crew gets to work',
    body: 'Three agents research the lead on the live web, assess cultural fit, and settle on a validated 0–100 score with a full breakdown.',
  },
  {
    n: '3',
    title: 'Outreach, ready to send',
    body: 'Leads scoring above 70 get a personalized email drafted and optimized by two more agents — waiting in the dashboard.',
  },
]

function useCountUp(target, durationMs, delayMs) {
  // reduced motion: start (and stay) at the target — no animation
  const [value, setValue] = useState(() =>
    window.matchMedia('(prefers-reduced-motion: reduce)').matches ? target : 0,
  )
  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    let raf
    const timer = setTimeout(() => {
      const start = performance.now()
      const tick = now => {
        const t = Math.min((now - start) / durationMs, 1)
        setValue(Math.round(target * (1 - Math.pow(1 - t, 3))))
        if (t < 1) raf = requestAnimationFrame(tick)
      }
      raf = requestAnimationFrame(tick)
    }, delayMs)
    return () => { clearTimeout(timer); cancelAnimationFrame(raf) }
  }, [target, durationMs, delayMs])
  return value
}

function Reveal({ children, className = '' }) {
  const ref = useRef(null)
  useEffect(() => {
    const el = ref.current
    if (!el) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          el.classList.add('lp-in')
          io.disconnect()
        }
      },
      { threshold: 0.15 },
    )
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return (
    <div ref={ref} className={`lp-reveal ${className}`}>
      {children}
    </div>
  )
}

export default function Landing({ onSignIn, onGetStarted }) {
  const score = useCountUp(85, 1400, 2600)

  return (
    <div className="lp">
      {/* Nav */}
      <nav className="lp-nav">
        <span className="lp-wordmark">
          <span className="lp-logo">🎯</span> Sales Pipeline
        </span>
        <div className="lp-nav-actions">
          <button className="btn btn-outline" onClick={onSignIn}>Sign in</button>
          <button className="btn btn-primary" onClick={onGetStarted}>Get started</button>
        </div>
      </nav>

      {/* Hero on the gradient mesh */}
      <header className="lp-hero">
        <div className="lp-hero-copy">
          <p className="lp-eyebrow lp-fade" style={{ '--d': '0ms' }}>Multi-agent sales automation</p>
          <h1 className="lp-headline lp-fade" style={{ '--d': '120ms' }}>
            Every lead scored.<br />Every email written.
          </h1>
          <p className="lp-sub lp-fade" style={{ '--d': '240ms' }}>
            Five AI agents research each lead on the live web, assign a validated
            0–100 score, and draft personalized outreach for everyone worth your time.
          </p>
          <div className="lp-cta-row lp-fade" style={{ '--d': '360ms' }}>
            <button className="btn btn-primary" onClick={onGetStarted}>Get started</button>
            <a className="btn btn-outline" href="#how">See how it works</a>
          </div>
        </div>

        {/* Animated product composite — the crew running a real lead */}
        <div className="lp-composite lp-fade" style={{ '--d': '480ms' }}>
          <div className="lp-mock-header">
            <span className="lp-mock-lead">Jane Doe — Acme Software</span>
            <span className="lp-mock-score tnum">Score {score}</span>
          </div>
          <ul className="lp-agents">
            {AGENTS.map((a, i) => (
              <li key={a.name} className="lp-agent" style={{ '--i': i }}>
                <span className="lp-agent-dot" />
                <span className="lp-agent-name">{a.name}</span>
                <span className="lp-agent-job">{a.job}</span>
              </li>
            ))}
          </ul>
          <div className="lp-mock-email">
            <div className="lp-email-label">Email draft</div>
            <p className="lp-email-line" style={{ '--i': 0 }}>Hi Jane — saw Acme is scaling its support team.</p>
            <p className="lp-email-line" style={{ '--i': 1 }}>Teams like yours use agent orchestration to cut response times…</p>
            <p className="lp-email-line" style={{ '--i': 2 }}>Open to a 15-minute walkthrough this week?<span className="lp-caret" /></p>
          </div>
        </div>
      </header>

      {/* How it works — a real sequence, so the numbers mean something */}
      <section className="lp-section" id="how">
        <Reveal>
          <h2 className="lp-h2">From form to follow-up in three steps</h2>
        </Reveal>
        <div className="lp-steps">
          {STEPS.map((s, i) => (
            <Reveal key={s.n} className={`lp-stagger-${i}`}>
              <div className="lp-step">
                <div className="lp-step-n tnum">{s.n}</div>
                <h3 className="lp-h3">{s.title}</h3>
                <p className="lp-body">{s.body}</p>
              </div>
            </Reveal>
          ))}
        </div>
      </section>

      {/* Feature band */}
      <section className="lp-section lp-band-soft">
        <div className="lp-features">
          <Reveal className="lp-stagger-0">
            <div className="lp-feature">
              <h3 className="lp-h3">Research on the live web</h3>
              <p className="lp-body">
                Agents search and scrape current sources for every lead — company size,
                market presence, role relevance — instead of guessing from the form.
              </p>
            </div>
          </Reveal>
          <Reveal className="lp-stagger-1">
            <div className="lp-feature">
              <h3 className="lp-h3">Scores you can defend</h3>
              <p className="lp-body">
                Every score arrives with demographic, firmographic, and behavioral
                components plus validation notes — a breakdown, not a black box.
              </p>
            </div>
          </Reveal>
          <Reveal className="lp-stagger-2">
            <div className="lp-feature">
              <h3 className="lp-h3">Costs you can see</h3>
              <p className="lp-body">
                Tokens, duration, and cost per lead sit in the analysis panel.
                Bring your own Gemini and Tavily keys — your usage stays yours.
              </p>
            </div>
          </Reveal>
        </div>
      </section>

      {/* Cream interlude — the human artifact */}
      <section className="lp-section">
        <Reveal>
          <div className="lp-cream">
            <div>
              <h2 className="lp-h2">Cold numbers. Warm words.</h2>
              <p className="lp-body">
                Scoring decides who deserves attention; the email crew decides what to
                say. Drafts acknowledge the lead's role, company, and use case — short,
                specific, and ready for one final human read before sending.
              </p>
            </div>
            <div className="lp-cream-email">
              <p>Hi Priya — customer support automation at Freshworks's scale is exactly
              where agent orchestration earns its keep.</p>
              <p>Happy to show you how teams route triage to AI crews without losing
              the human tone. 15 minutes this week?</p>
            </div>
          </div>
        </Reveal>
      </section>

      {/* Dark CTA band */}
      <section className="lp-section lp-section-tight">
        <Reveal>
          <div className="lp-dark-cta">
            <h2 className="lp-h2-inverse">Process your first lead in minutes</h2>
            <p className="lp-body-inverse">
              Create an account, add your Gemini and Tavily keys, and let the crew work.
            </p>
            <button className="btn btn-primary" onClick={onGetStarted}>Get started</button>
          </div>
        </Reveal>
      </section>

      <footer className="lp-footer">
        <p>Sales Pipeline — built on CrewAI, Google Gemini, Tavily, and Supabase.</p>
        <p className="lp-copyright">© {new Date().getFullYear()} Sales Pipeline. All rights reserved.</p>
      </footer>
    </div>
  )
}
