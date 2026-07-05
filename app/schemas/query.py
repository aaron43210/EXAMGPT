from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    citations: list[dict] = []
    evaluation: dict = {}
    retry_count: int = 0
    latency_ms: float = 0


class ChatHistoryItem(BaseModel):
    id: int
    query: str
    response: str
    citations_json: Optional[list[dict]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryItem]
    total: int
