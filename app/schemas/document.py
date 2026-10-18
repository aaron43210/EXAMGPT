from pydantic import BaseModel, ConfigDict
from typing import Optional
from datetime import datetime


class DocumentResponse(BaseModel):
    id: int
    title: str
    filename: str
    doc_type: str
    status: str
    course_id: int
    chunk_count: int = 0
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
