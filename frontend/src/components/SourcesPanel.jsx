import { useId, useState } from "react";

/**
 * Collapsible list of the chunks the backend retrieved for this answer.
 * Collapsed: a one-line summary with the document names.
 * Expanded: each chunk as [Source N] with file, page, relevance and an excerpt.
 */
export default function SourcesPanel({ sources }) {
  const [open, setOpen] = useState(false);
  const panelId = useId();
  const docNames = [...new Set(sources.map((s) => s.source))];

  return (
    <div className="mt-3 rounded-xl border border-stone-200 bg-white">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-controls={panelId}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs text-stone-500 hover:text-stone-800"
      >
        <svg
          className={`h-3 w-3 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path d="M7 4l6 6-6 6V4z" />
        </svg>
        <span className="font-medium">Sources ({sources.length})</span>
        <span className="truncate text-stone-400">· {docNames.join(", ")}</span>
      </button>

      {open && (
        <ol id={panelId} className="space-y-2 border-t border-stone-100 px-3 py-3">
          {sources.map((s, i) => (
            <li key={`${s.source}-${s.page}-${i}`} className="rounded-lg bg-stone-50 p-2.5">
              <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                <span className="rounded bg-stone-200 px-1.5 py-0.5 font-mono text-[10px] text-stone-600">
                  Source {i + 1}
                </span>
                <span className="font-medium text-stone-700">📄 {s.source}</span>
                {s.page != null && <span className="text-stone-400">page {s.page}</span>}
                {s.score != null && (
                  <span className="ml-auto text-teal-700">{Math.round(s.score * 100)}% match</span>
                )}
              </div>
              <p className="mt-1.5 line-clamp-4 whitespace-pre-wrap text-xs leading-relaxed text-stone-500">
                {s.content}
              </p>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}
