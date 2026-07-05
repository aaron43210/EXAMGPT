from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class StudyPlanRequest(BaseModel):
    days: int = 14
    hours_per_day: float = 4.0


class TopicPriority(BaseModel):
    topic: str
    priority: str  # high/medium/low
    estimated_hours: float
    pyq_frequency: int = 0
    rationale: str = ""


class StudyPlanResponse(BaseModel):
    id: int
    course_id: int
    plan_data: dict
    created_at: datetime

    class Config:
        from_attributes = True
