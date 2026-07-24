const BACKEND =
  import.meta.env.VITE_BACKEND_URL ||
  (import.meta.env.DEV
    ? "http://localhost:8000"
    : "https://multi-crew-leads-dashboard.onrender.com");

const FETCH_TIMEOUT_MS = 60_000;
const SESSION_KEY = "sp_session";

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
}

function writeSession(s) {
  localStorage.setItem(SESSION_KEY, JSON.stringify(s));
}

async function doFetch(method, path, body, token) {
  const headers = { "Content-Type": "application/json" };
  if (token) headers.Authorization = `Bearer ${token}`;
  const opts = { method, headers, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) };
  if (body) opts.body = JSON.stringify(body);
  return fetch(`${BACKEND}${path}`, opts);
}

// Exchange the (long-lived) refresh token for a fresh access token and update
// the stored session. Returns the new access token, or null if it failed
// (refresh token expired/revoked → the user must log in again).
async function refreshAccessToken() {
  const s = readSession();
  if (!s?.refreshToken) return null;
  let res;
  try {
    res = await doFetch("POST", "/auth/refresh", { refresh_token: s.refreshToken });
  } catch {
    return null;
  }
  if (!res.ok) return null;
  const { token } = await res.json();
  const cur = readSession();
  if (cur) {
    cur.token = token;
    writeSession(cur);
  }
  return token;
}

export async function api(method, path, body) {
  let res = await doFetch(method, path, body, readSession()?.token);

  // Access token expired mid-session → silently refresh once and retry, so the
  // user isn't kicked to login every hour. Only when we actually have a refresh
  // token (i.e. logged in) and not on the refresh call itself.
  if (res.status === 401 && path !== "/auth/refresh" && readSession()?.refreshToken) {
    const newToken = await refreshAccessToken();
    if (newToken) {
      res = await doFetch(method, path, body, newToken);
    } else {
      // Refresh token itself is dead — clear the session and let the app show login.
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
  return err?.message || String(err);
}
