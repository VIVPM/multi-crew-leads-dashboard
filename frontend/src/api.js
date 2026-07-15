const BACKEND =
  import.meta.env.VITE_BACKEND_URL ||
  (import.meta.env.DEV
    ? "http://localhost:8000"
    : "https://multi-crew-leads-dashboard.onrender.com");

const FETCH_TIMEOUT_MS = 60_000;
const SESSION_KEY = "sp_session";
const SESSION_TTL = 60 * 60 * 1000; // 1 hour

function readSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY));
  } catch {
    return null;
  }
}

function touchSession() {
  const s = readSession();
  if (!s) return;
  s.expiresAt = Date.now() + SESSION_TTL;
  localStorage.setItem(SESSION_KEY, JSON.stringify(s));
}

export async function api(method, path, body) {
  const headers = { "Content-Type": "application/json" };
  const token = readSession()?.token;
  if (token) headers.Authorization = `Bearer ${token}`;
  const opts = { method, headers, signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${BACKEND}${path}`, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const d = (await res.json()).detail;
      if (typeof d === "string") detail = d;
      else if (Array.isArray(d)) detail = d.map(x => x.msg || String(x)).join("; ");
    } catch { /* non-JSON error body — keep the HTTP status */ }
    throw new Error(detail);
  }
  touchSession(); // sliding session: any successful call extends the TTL
  return res.json();
}

export function friendlyError(err) {
  const msg = String(err?.message || err).toLowerCase();
  if (msg.includes("timeout") || msg.includes("timed out"))
    return "The request timed out. Please try again.";
  if (msg.includes("failed to fetch") || msg.includes("network"))
    return "Cannot reach the backend server. Make sure it is running.";
  if (msg.includes("token") || msg.includes("unauthorized"))
    return "Your session is invalid or expired. Please log in again.";
  if (msg.includes("429") || msg.includes("rate limit") || msg.includes("too many"))
    return "Too many requests. Please wait a moment and try again.";
  if (msg.includes("500") || msg.includes("internal server"))
    return "The server encountered an error. Please try again later.";
  return err?.message || String(err);
}
