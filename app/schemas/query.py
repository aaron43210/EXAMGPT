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
    chat_id: int
    response: str
    intent: str
    confidence: float = 0
    citations: list[dict] = []
    evaluation: dict = {}
    retry_count: int = 0
    latency_ms: float = 0
    guardrail_flags: list[str] = []
    cached: bool = False
    trace: dict = {}


class FeedbackRequest(BaseModel):
    rating: int = Field(description="1 = helpful, -1 = not helpful")
    comment: Optional[str] = Field(default=None, max_length=1000)

    @field_validator("rating")
    @classmethod
    def rating_values(cls, v: int) -> int:
        if v not in (1, -1):
            raise ValueError("rating must be 1 or -1")
        return v


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
