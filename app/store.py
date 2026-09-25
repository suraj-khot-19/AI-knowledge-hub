"""Manage session-isolated FAISS indexes and their embedding model."""

import re
from pathlib import Path
from threading import RLock

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class IndexStore:
    """Session-local FAISS indexes, persisted beneath DATA_DIR."""

    def __init__(self, root: str, embedding_model: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Normalized vectors make FAISS distance comparisons more consistent.
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, encode_kwargs={"normalize_embeddings": True})
        self._indexes: dict[str, FAISS] = {}
        self._locks: dict[str, RLock] = {}
        self._guard = RLock()

    @staticmethod
    def _safe(session_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:128]

    def _lock(self, session_id: str) -> RLock:
        with self._guard:
            return self._locks.setdefault(session_id, RLock())

    def _get(self, session_id: str) -> FAISS | None:
        """Return a cached index or restore it from the session's data directory."""
        if session_id in self._indexes:
            return self._indexes[session_id]
        path = self.root / self._safe(session_id)
        if (path / "index.faiss").exists():
            # Persisted indexes must only be loaded from this application's trusted data directory.
            index = FAISS.load_local(str(path), self.embeddings, allow_dangerous_deserialization=True)
            self._indexes[session_id] = index
            return index
        return None

    def add_documents(self, session_id: str, documents: list) -> int:
        """Create or extend one session index, then persist the updated index."""
        with self._lock(session_id):
            if not documents:
                return 0
            index = self._get(session_id)
            if index is None:
                index = FAISS.from_documents(documents, self.embeddings)
                self._indexes[session_id] = index
            else:
                index.add_documents(documents)
            index.save_local(str(self.root / self._safe(session_id)))
            return len(documents)

    def search(self, session_id: str, query: str, k: int):
        """Return the k closest passages with FAISS distances (lower is closer)."""
        with self._lock(session_id):
            index = self._get(session_id)
            return [] if index is None else index.similarity_search_with_score(query, k=k)

    def clear(self, session_id: str) -> None:
        """Remove an in-memory index and its persisted files for a session."""
        with self._lock(session_id):
            self._indexes.pop(session_id, None)
            path = self.root / self._safe(session_id)
            if path.exists():
                import shutil
                shutil.rmtree(path)
