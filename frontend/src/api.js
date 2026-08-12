// Required in production builds (no hardcoded fallback — same reasoning as
// the backend's ALLOWED_ORIGINS: the real deployed URL doesn't belong in
// source). Vite inlines env vars at build time, so this has to be set when
// the bundle is built, not when it's served — see the README setup section.
// Dev keeps a plain localhost fallback since that value isn't sensitive and
// every contributor needs it to just work.
if (!import.meta.env.DEV && !import.meta.env.VITE_BACKEND_URL) {
  throw new Error(
    "VITE_BACKEND_URL is not set. It must be set at build time (Vite inlines " +
    "env vars into the bundle) — check the build environment's config."
  );
}
const BACKEND = import.meta.env.VITE_BACKEND_URL || "http://localhost:8000";

const FETCH_TIMEOUT_MS = 60_000;
const SESSION_KEY = "sp_session";

// Access/refresh tokens live in httpOnly cookies now (backend.py sets them on
// login/signup/refresh) — this key stores {userId, username, csrfToken} as a
// UI hint, never a credential the httpOnly cookies protect, so there's
// nothing here for an XSS bug to steal beyond what CSRF already assumes is
// exposed.
function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
}

// Patches just the csrf token into the existing session (login/refresh may
// hand back a new one) without touching whatever else App.jsx put there.
function updateCsrfToken(csrfToken) {
  if (!csrfToken) return;
  const s = readSession();
  if (s) localStorage.setItem(SESSION_KEY, JSON.stringify({ ...s, csrfToken }));
}

async function doFetch(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  // Can't read the csrf_token cookie via document.cookie here — it belongs
  // to the backend's origin, not this page's, and cookies are only ever
  // JS-readable same-origin regardless of SameSite/Secure. The backend hands
  // the value back in the login/signup/refresh response body instead (the
  // one channel this page can actually read for its own fetches), and it's
  // stored below in the session blob.
  const csrf = readSession()?.csrfToken;
  if (csrf) headers["X-CSRF-Token"] = csrf;
  const opts = {
    method, headers, credentials: "include", // send/receive the auth cookies
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  };
  if (body) opts.body = JSON.stringify(body);
  return fetch(`${BACKEND}${path}`, opts);
}

// Refreshes the access_token cookie in place via the refresh_token cookie —
// no request body, no access/refresh token value ever touches JS. Returns
// whether it worked. The response does carry the csrf token though (see
// doFetch's comment), so a page reload that lost its in-memory/localStorage
// copy resyncs here instead of needing a fresh login.
async function refreshSession() {
  try {
    const res = await doFetch("POST", "/auth/refresh");
    if (!res.ok) return false;
    const data = await res.json().catch(() => null);
    updateCsrfToken(data?.csrf_token);
    return true;
  } catch {
    return false;
  }
}

export async function api(method, path, body) {
  let res = await doFetch(method, path, body);

  // Access token expired mid-session — refresh once and retry, so the user
  // isn't kicked to login every hour.
  if (res.status === 401 && path !== "/auth/refresh" && readSession()) {
    const refreshed = await refreshSession();
    if (refreshed) {
      res = await doFetch(method, path, body);
    } else {
      // Refresh token itself is dead — clear the session hint and show login.
      localStorage.removeItem(SESSION_KEY);
      window.dispatchEvent(new Event("sp-auth-expired"));
    }
  }

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") detail = d;
      else if (Array.isArray(d)) detail = d.map(x => x.msg || String(x)).join("; ");
    } catch { /* non-JSON error body — keep the HTTP status */ }
    throw new Error(detail);
  }
  return res.json();
}

export function friendlyError(err) {
  const msg = String(err?.message || err).toLowerCase();
  if (msg.includes("timeout") || msg.includes("timed out"))
    return "The request timed out. Please try again.";
  if (msg.includes("failed to fetch") || msg.includes("network"))
    return "Cannot reach the backend server. Make sure it is running.";
  // Checked before the generic token/session match below: a CSRF failure's
  // detail text ("Missing or invalid CSRF token.") also contains "token",
  // which used to make it show the same message as an actually-expired
  // session — a real bug, not a hypothetical one (see backend.py's
  // _set_auth_cookies comment for how it was found).
  if (msg.includes("csrf"))
    return "Your session needs a refresh — please reload the page and try again.";
  if (msg.includes("token") || msg.includes("unauthorized") || msg.includes("session expired"))
    return "Your session is invalid or expired. Please log in again.";
  if (msg.includes("429") || msg.includes("rate limit") || msg.includes("too many"))
    return "Too many requests. Please wait a moment and try again.";
  if (msg.includes("500") || msg.includes("internal server"))
    return "The server encountered an error. Please try again later.";
  return err?.message || String(err);
}
