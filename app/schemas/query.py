from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing import Optional
from datetime import datetime


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be blank")
        return v


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

    model_config = ConfigDict(from_attributes=True)


class ChatHistoryResponse(BaseModel):
    messages: list[ChatHistoryItem]
    total: int
