# Context-Aware Document RAG Agent

Ask natural-language questions about a folder of PDFs. The agent retrieves the most relevant
passages from a local vector database and answers **only** from those passages, returning
the exact sources it used.

**Stack:** Python · FastAPI · LangChain · ChromaDB · pypdf · Gemini or OpenAI · React · Tailwind CSS · Axios

```
            ┌──────────── Phase 1: ingest.py (offline) ────────────┐
 data/*.pdf ─► pypdf text ─► clean ─► RecursiveCharacterTextSplitter ─► embeddings ─► ChromaDB (chroma_db/)
                                              1000 chars / 200 overlap                     │
            └──────────────────────────────────────────────────────┘                     │
                                                                                          │ top-3 similarity
            ┌──────────── Phase 2: main.py (FastAPI) ──────────────┐                     │
 React UI ──► POST /chat ─► embed query ─► retrieve ◄──────────────┼─────────────────────┘
    ▲                                         │
    │                                         ▼
    └──── { answer, sources[] } ◄── LLM (context-only prompt)
            Phase 3: frontend/
```

## Project structure

```
rag-agent/
├── backend/
│   ├── config.py          # Settings + provider factories (embeddings, LLM, vector store)
│   ├── ingest.py          # Phase 1: PDF → chunks → embeddings → ChromaDB
│   ├── main.py            # Phase 2: FastAPI RAG API (/chat, /health)
│   ├── requirements.txt
│   ├── .env.example
│   ├── data/              # Put your PDFs here
│   └── chroma_db/         # Created automatically (persisted vectors)
└── frontend/
    ├── src/
    │   ├── App.jsx                  # Chat layout, state, empty state
    │   ├── api.js                   # Axios client + error handling
    │   └── components/
    │       ├── ChatInput.jsx        # Enter to send, Shift+Enter for newline
    │       ├── ChatMessage.jsx      # User / assistant / error bubbles
    │       └── SourcesPanel.jsx     # Collapsible "Sources" under each answer
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Quick start

**Prerequisites:** Python 3.10+, Node.js 20+, and a Gemini API key
([Google AI Studio](https://aistudio.google.com/apikey)) or an OpenAI API key.

### 1. Backend + ingestion

```bash
cd backend
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # Windows: copy .env.example .env
# edit .env → set GOOGLE_API_KEY (or LLM_PROVIDER=openai + OPENAI_API_KEY)

# copy some PDFs into backend/data/, then:
python ingest.py                     # add --reset to rebuild from scratch
```

### 2. Start the API

```bash
uvicorn main:app --reload --port 8000
```

Open <http://localhost:8000/docs> to try `/chat` from the Swagger UI.

### 3. Start the frontend (new terminal)

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>.

## API

### `POST /chat`

```json
// request
{ "query": "What is the notice period for termination?" }

// response
{
  "answer": "The notice period is 30 days, given in writing [Source 1].",
  "sources": [
    {
      "source": "employment_contract.pdf",
      "page": 4,
      "content": "…either party may terminate this agreement with thirty (30) days' written notice…",
      "score": 0.812
    }
  ]
}
```

### `GET /health`

```json
{ "status": "ok", "provider": "gemini", "collection": "documents_gemini", "chunks": 412 }
```

## Configuration (`backend/.env`)

| Variable             | Default                    | Purpose                                             |
|----------------------|----------------------------|-----------------------------------------------------|
| `LLM_PROVIDER`       | `gemini`                   | `gemini` or `openai`, used for embeddings and chat |
| `GOOGLE_API_KEY`     | —                          | Required when provider is `gemini`                  |
| `OPENAI_API_KEY`     | —                          | Required when provider is `openai`                  |
| `GEMINI_EMBED_MODEL` | `gemini-embedding-001`     | Gemini embedding model                              |
| `GEMINI_CHAT_MODEL`  | `gemini-flash-latest`      | Auto-updating alias; pin a version for reproducibility |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small`   | OpenAI embedding model                              |
| `OPENAI_CHAT_MODEL`  | `gpt-5-mini`               | OpenAI chat model                                   |
| `CHUNK_SIZE`         | `1000`                     | Characters per chunk                                |
| `CHUNK_OVERLAP`      | `200`                      | Overlap between neighbouring chunks                 |
| `TOP_K`              | `3`                        | Chunks retrieved per question                       |
| `CORS_ORIGINS`       | `http://localhost:5173,…`  | Comma-separated allowed frontend origins            |

Frontend: set `VITE_API_URL` in `frontend/.env` if the API is not at `http://localhost:8000`.

## Design decisions

- **Single source of truth for providers.** `config.py` builds the embedding model, LLM and
  vector store for both ingestion and serving, so query vectors always live in the same space
  as stored vectors. Switching Gemini ↔ OpenAI is one line in `.env`.
- **One collection per provider.** Gemini and OpenAI embeddings have different dimensions;
  separate collections (`documents_gemini`, `documents_openai`) make mixing them impossible.
- **Page-level provenance.** Each PDF page becomes a `Document` with `source` and `page`
  metadata, which flows through chunking into the API response and the UI.
- **Idempotent, incremental ingestion.** Deterministic chunk IDs plus a stored per-file SHA-256:
  unchanged files are skipped (no repeat embedding cost), edited files are re-indexed cleanly,
  and deleted files are pruned from the index.
- **Grounded generation.** The system prompt restricts the model to the retrieved context,
  requires `[Source N]` citations, defines an exact "not found" reply, and tells the model to
  ignore instructions embedded in documents (basic prompt-injection hygiene).
- **Resilience.** Batched embedding with exponential-backoff retries; corrupt, encrypted and
  image-only PDFs are logged and skipped instead of crashing the run; API errors map to
  clear HTTP 502 messages that the UI displays.
- **Non-blocking API.** `/chat` is a sync endpoint, so FastAPI runs the blocking vector search
  and LLM call in a worker thread instead of stalling the event loop.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `GOOGLE_API_KEY is not set` | Create `backend/.env` from `.env.example` and add your key. |
| `no extractable text … needs OCR` | The PDF is a scanned image. Run it through OCR (e.g. `ocrmypdf`) first. |
| Answers always say "couldn't find that" | Run `python ingest.py` and check `GET /health` shows `chunks > 0`. Make sure the same `LLM_PROVIDER` is used for ingestion and the API. |
| UI says "Can't reach the backend" | Start `uvicorn` on port 8000, or set `VITE_API_URL`. |
| CORS error in the browser console | Add your frontend URL to `CORS_ORIGINS`. |
| 502 "language model request failed" | Check the key, quota and model name (`GEMINI_CHAT_MODEL` / `OPENAI_CHAT_MODEL`). The terminal running uvicorn shows the full error. |

## Possible extensions

- Conversation memory (rewrite follow-up questions into stand-alone queries before retrieval)
- Streaming answers over Server-Sent Events
- Hybrid search (BM25 + vectors) and a re-ranker
- Upload endpoint so users can add PDFs from the UI
- OCR fallback for scanned PDFs
