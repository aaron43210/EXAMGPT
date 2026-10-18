from pydantic import BaseModel, ConfigDict, Field
from datetime import datetime


class StudyPlanRequest(BaseModel):
    days: int = Field(default=14, ge=1, le=90)
    hours_per_day: float = Field(default=4.0, ge=0.5, le=16)


class StudyPlanResponse(BaseModel):
    id: int
    course_id: int
    plan_data: dict
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
