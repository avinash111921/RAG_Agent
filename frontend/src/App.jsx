import { useEffect, useRef, useState } from "react";
import { describeError, sendChat } from "./api";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";

const SUGGESTIONS = [
  "Summarize the key points of these documents",
  "What are the main risks mentioned?",
  "List any deadlines or important dates",
];

let nextId = 0;
const makeMessage = (role, content, sources = []) => ({ id: nextId++, role, content, sources });

export default function App() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  // Keep the newest message in view.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function ask(query) {
    const text = query.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, makeMessage("user", text)]);
    setLoading(true);
    try {
      const { answer, sources } = await sendChat(text);
      setMessages((prev) => [...prev, makeMessage("assistant", answer, sources)]);
    } catch (err) {
      setMessages((prev) => [...prev, makeMessage("error", describeError(err))]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex h-full flex-col font-sans text-stone-800">
      <header className="border-b border-stone-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-3xl items-center justify-between px-4 py-3">
          <div>
            <h1 className="text-base font-semibold tracking-tight">Document RAG Agent</h1>
            <p className="text-xs text-stone-500">Answers grounded in your PDFs, with sources</p>
          </div>
          {messages.length > 0 && (
            <button
              onClick={() => setMessages([])}
              disabled={loading}
              className="rounded-md px-2.5 py-1.5 text-xs font-medium text-stone-500 hover:bg-stone-100 hover:text-stone-800 disabled:opacity-40"
            >
              New chat
            </button>
          )}
        </div>
      </header>

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-6 px-4 py-8">
          {messages.length === 0 ? (
            <EmptyState onPick={ask} />
          ) : (
            messages.map((m) => <ChatMessage key={m.id} message={m} />)
          )}
          {loading && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-stone-200 bg-white">
        <div className="mx-auto max-w-3xl px-4 py-3">
          <ChatInput onSend={ask} disabled={loading} />
        </div>
      </footer>
    </div>
  );
}

function EmptyState({ onPick }) {
  return (
    <div className="pt-16 text-center">
      <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-600 text-xl text-white">
        ✦
      </div>
      <h2 className="text-lg font-semibold">Ask anything about your documents</h2>
      <p className="mx-auto mt-1 max-w-md text-sm text-stone-500">
        The agent searches your indexed PDFs, answers only from what it finds, and shows you exactly
        where each answer came from.
      </p>
      <div className="mt-6 flex flex-wrap justify-center gap-2">
        {SUGGESTIONS.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="rounded-full border border-stone-200 bg-white px-3.5 py-1.5 text-sm text-stone-600 transition hover:border-teal-300 hover:text-teal-700"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 pl-1" aria-label="Assistant is thinking">
      {[0, 150, 300].map((delay) => (
        <span
          key={delay}
          className="h-2 w-2 animate-bounce rounded-full bg-stone-400"
          style={{ animationDelay: `${delay}ms` }}
        />
      ))}
    </div>
  );
}
