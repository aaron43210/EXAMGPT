from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DocumentUploadResponse(BaseModel):
    id: int
    filename: str
    doc_type: str
    status: str
    message: str

    class Config:
        from_attributes = True


class DocumentResponse(BaseModel):
    id: int
    title: str
    filename: str
    doc_type: str
    status: str
    course_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
