import SourcesPanel from "./SourcesPanel";

/** One chat bubble: user question, assistant answer (+ sources), or error. */
export default function ChatMessage({ message }) {
  const { role, content, sources } = message;

  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-teal-600 px-4 py-2.5 text-sm text-white">
          {content}
        </div>
      </div>
    );
  }

  if (role === "error") {
    return (
      <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700" role="alert">
        {content}
      </div>
    );
  }

  return (
    <div className="flex gap-3">
      <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-stone-800 text-xs text-white">
        ✦
      </div>
      <div className="min-w-0 flex-1">
        <div className="whitespace-pre-wrap text-sm leading-relaxed">{content}</div>
        {sources?.length > 0 && <SourcesPanel sources={sources} />}
      </div>
    </div>
  );
}
