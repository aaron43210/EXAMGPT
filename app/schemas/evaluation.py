from pydantic import BaseModel

class EvaluationRequest(BaseModel):
    question_id: int
    answer: str

class EvaluationResponse(BaseModel):
    score: int
    feedback: str
