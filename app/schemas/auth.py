from pydantic import BaseModel, ConfigDict, Field


class GoogleLogin(BaseModel):
    credential: str = Field(min_length=1, max_length=4096)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    name: str
    role: str = "student"

    model_config = ConfigDict(from_attributes=True)


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
