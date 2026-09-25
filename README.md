# AI Knowledge Hub

A session-scoped document Q&A API built with FastAPI, LangChain, Hugging Face embeddings, and FAISS. It accepts PDF, TXT, and Markdown documents, splits them into overlapping chunks, persists a FAISS index per session, retrieves relevant passages, and returns source metadata with answers.

## Setup

Python 3.10 or newer is recommended. On Windows, use a virtual environment:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

First run downloads the configured embedding model. Open `http://127.0.0.1:8000/docs` for interactive API docs.

By default, the service uses `sentence-transformers/all-MiniLM-L6-v2` for embeddings and a grounded extractive fallback for responses, so it can be tried without a generative model. To use Hugging Face text generation, set `HF_GENERATION_MODEL` in `.env` to a compatible text-generation model ID; set `HF_TOKEN` for gated models. Choose a model your hardware can run.

## API

Upload a document to a session:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/documents/upload?session_id=demo" -F "file=@manual.pdf"
```

Ask a question:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/chat/query" -H "Content-Type: application/json" -d '{"session_id":"demo","question":"What does the manual say about setup?"}'
```

Other routes: `POST /chat/stream` (Server-Sent Events), `GET /chat/history/{session_id}`, `DELETE /sessions/{session_id}`, and `GET /health`.

## Notes

- Conversation history is in process memory and is limited to the most recent 20 turns. FAISS indexes persist beneath `DATA_DIR`.
- The streaming route emits server-sent events, but the current Hugging Face pipeline generates a complete answer before emitting it.
- Retrieval score is the FAISS distance (lower is more similar), not a calibrated confidence or accuracy percentage.
- The claimed 92% accuracy, 35% relevance gain, 40% latency reduction, and 45% processing-time reduction are not guaranteed by code. Establish a representative evaluation set and baseline, then measure those metrics on target hardware and documents.
- For a public deployment, add authentication, per-user authorization, quotas, restrictive CORS, persistent chat storage, and a managed background job queue. Only load FAISS indexes from a trusted data directory because LangChain's local persistence uses Python deserialization.
