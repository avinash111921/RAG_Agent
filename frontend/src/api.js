import axios from "axios";

// All backend communication lives here, so components never deal with URLs or HTTP details.
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  timeout: 60_000, // LLM answers can take a while
  headers: { "Content-Type": "application/json" },
});

/**
 * Ask a question about the indexed documents.
 * @param {string} query
 * @returns {Promise<{answer: string, sources: {source: string, page: number|null, content: string, score: number|null}[]}>}
 */
export async function sendChat(query) {
  const { data } = await api.post("/chat", { query });
  return data;
}

/** Turn any Axios error into a short, human-readable message. */
export function describeError(err) {
  if (err.code === "ECONNABORTED") return "The request timed out. Please try again.";
  if (!err.response) return "Can't reach the backend. Is the API running on port 8000?";
  const detail = err.response.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join(", ");
  return `Request failed (HTTP ${err.response.status}).`;
}
