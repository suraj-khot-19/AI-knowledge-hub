"""Validated request and response models exposed by the HTTP API."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)
    session_id: str = Field(min_length=1, max_length=128)
    k: int | None = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    filename: str
    page: int | None = None
    chunk_id: str
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    session_id: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    chunks_indexed: int
    message: str


class HistoryMessage(BaseModel):
    role: str
    content: str
