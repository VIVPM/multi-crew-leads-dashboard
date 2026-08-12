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
// login/signup/refresh) — this key only stores {userId, username} as a UI
// hint, never a credential, so there's nothing here for an XSS bug to steal.
function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
}

// The CSRF cookie is deliberately not httpOnly — the frontend has to read it
// to echo it back as a header (double-submit pattern; see backend.py).
function getCsrfToken() {
  const m = document.cookie.match(/(?:^|;\s*)csrf_token=([^;]+)/);
  return m ? decodeURIComponent(m[1]) : null;
}

async function doFetch(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  const csrf = getCsrfToken();
  if (csrf) headers["X-CSRF-Token"] = csrf;
  const opts = {
    method, headers, credentials: "include", // send/receive the auth cookies
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  };
  if (body) opts.body = JSON.stringify(body);
  return fetch(`${BACKEND}${path}`, opts);
}

// Refreshes the access_token cookie in place via the refresh_token cookie —
// no request body, no token value ever touches JS. Returns whether it worked.
async function refreshSession() {
  try {
    const res = await doFetch("POST", "/auth/refresh");
    return res.ok;
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
  if (msg.includes("token") || msg.includes("unauthorized") || msg.includes("session expired"))
    return "Your session is invalid or expired. Please log in again.";
  if (msg.includes("429") || msg.includes("rate limit") || msg.includes("too many"))
    return "Too many requests. Please wait a moment and try again.";
  if (msg.includes("500") || msg.includes("internal server"))
    return "The server encountered an error. Please try again later.";
  if (msg.includes("csrf"))
    return "Your session needs a refresh — please reload the page.";
  return err?.message || String(err);
}
