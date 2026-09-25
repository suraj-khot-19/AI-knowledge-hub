"""FastAPI entry point for document ingestion, retrieval, and session chat."""

import asyncio
import json
import logging
from collections import defaultdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.answer import AnswerGenerator
from app.config import get_settings
from app.documents import load_and_split
from app.schemas import HistoryMessage, QueryRequest, QueryResponse, Source, UploadResponse
from app.store import IndexStore

settings = get_settings()
# Keep each session's FAISS index on disk so it survives an application restart.
Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
store = IndexStore(settings.data_dir, settings.embedding_model)
generator = AnswerGenerator(settings.hf_generation_model, settings.hf_token)
histories: dict[str, list[dict]] = defaultdict(list)
# Protect shared in-process chat history while concurrent requests update it.
history_lock = asyncio.Lock()
app = FastAPI(title=settings.app_name, version="1.0.0", description="Session-scoped document Q&A backed by LangChain and FAISS.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log = logging.getLogger("knowledge_hub")


@app.get("/health")
async def health():
    return {"status": "ok", "embedding_model": settings.embedding_model, "generation_model": settings.hf_generation_model or None}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(session_id: str, file: UploadFile = File(...)):
    """Read a bounded upload, index its chunks, and report the number stored."""
    name = Path(file.filename or "upload").name
    payload = await file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    if len(payload) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    try:
        chunks = await asyncio.to_thread(load_and_split, name, payload, settings.chunk_size, settings.chunk_overlap)
        count = await asyncio.to_thread(store.add_documents, session_id, chunks)
    except ValueError as e:
        raise HTTPException(415, str(e)) from e
    except Exception as e:
        log.exception("Document ingestion failed")
        raise HTTPException(422, "Could not read or index this document") from e
    if not count:
        raise HTTPException(422, "No extractable text found in document")
    return UploadResponse(session_id=session_id, filename=name, chunks_indexed=count, message="Document indexed successfully")


async def _query(request: QueryRequest):
    """Shared retrieval and answer path used by both JSON and SSE endpoints."""
    k = request.k or settings.retrieval_k
    results = await asyncio.to_thread(store.search, request.session_id, request.question, k)
    docs = [doc for doc, _score in results]
    async with history_lock:
        prior = list(histories[request.session_id])
    answer = await asyncio.to_thread(generator.answer, request.question, docs, prior)
    sources = [Source(filename=d.metadata.get("filename", "unknown"), page=d.metadata.get("page"),
                      chunk_id=d.metadata.get("chunk_id", ""), score=float(score)) for d, score in results]
    async with history_lock:
        histories[request.session_id].extend([{"role": "user", "content": request.question}, {"role": "assistant", "content": answer}])
        histories[request.session_id] = histories[request.session_id][-40:]
    return QueryResponse(answer=answer, sources=sources, session_id=request.session_id)


@app.post("/chat/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return await _query(request)


@app.post("/chat/stream")
async def stream(request: QueryRequest):
    """Return answer chunks as Server-Sent Events for frontend clients."""
    async def events():
        try:
            response = await _query(request)
            # SSE event stream; emitted as one answer event for both configured and fallback models.
            yield f"event: sources\ndata: {json.dumps([s.model_dump() for s in response.sources])}\n\n"
            for start in range(0, len(response.answer), 80):
                yield f"event: token\ndata: {json.dumps(response.answer[start:start+80])}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/chat/history/{session_id}", response_model=list[HistoryMessage])
async def get_history(session_id: str):
    async with history_lock:
        return histories[session_id]


@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Delete the session index and its in-memory conversation history."""
    await asyncio.to_thread(store.clear, session_id)
    async with history_lock:
        histories.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}
