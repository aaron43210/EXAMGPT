from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class QuizRequest(BaseModel):
    topic: Optional[str] = Field(default=None, max_length=200)
    num_questions: int = Field(default=5, ge=1, le=10)
    difficulty: Literal["easy", "medium", "hard"] = "medium"

    @field_validator("topic")
    @classmethod
    def blank_topic(cls, v):
        return v.strip() or None if v else None


class GradeRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    answer: str = Field(min_length=1, max_length=8000)
    max_score: int = Field(default=10, ge=1, le=100)
