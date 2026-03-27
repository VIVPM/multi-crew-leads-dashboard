const BACKEND =
  import.meta.env.VITE_BACKEND_URL ||
  "https://multi-crew-leads-dashboard.onrender.com";
//  || "http://localhost:8000";

export async function api(method, path, body) {
  const opts = {
    method,
    headers: { "Content-Type": "application/json" },
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(`${BACKEND}${path}`, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      detail = (await res.json()).detail || detail;
    } catch {}
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
  if (msg.includes("401") || msg.includes("unauthorized"))
    return "Invalid credentials. Please check your API key.";
  if (msg.includes("429") || msg.includes("rate limit"))
    return "Rate limit exceeded. Please wait a moment and try again.";
  if (msg.includes("500") || msg.includes("internal server"))
    return "The server encountered an error. Please try again later.";
  return err?.message || String(err);
}
