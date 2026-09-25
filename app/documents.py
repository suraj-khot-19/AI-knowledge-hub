"""Load supported uploads and convert them into metadata-rich text chunks."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


SUPPORTED = {".pdf", ".txt", ".md"}


def load_and_split(filename: str, payload: bytes, chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Parse a PDF or text file, then split it into overlapping retrieval units."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise ValueError(f"Unsupported file type {suffix!r}. Supported types: PDF, TXT, MD.")
    with NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(payload)
        path = Path(temp.name)
    try:
        # Loaders expect a filesystem path; always remove the temporary upload afterward.
        if suffix == ".pdf":
            docs = PyPDFLoader(str(path)).load()
        else:
            docs = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    finally:
        path.unlink(missing_ok=True)

    for doc in docs:
        # Keep the original filename and one-based PDF page for source attribution.
        doc.metadata.update({"filename": Path(filename).name, "page": (doc.metadata.get("page", 0) + 1) if suffix == ".pdf" else None})
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        # Stable IDs make it easier to identify each returned passage in API responses.
        chunk.metadata["chunk_id"] = f"{Path(filename).stem}-{i}"
    return chunks
