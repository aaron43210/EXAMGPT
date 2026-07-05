"""
Centralized configuration via Pydantic Settings.
All secrets and connection strings are pulled from environment variables.
"""
import os
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── App ─────────────────────────────────────────────────────
    APP_NAME: str = "ExamGPT"
    DEBUG: bool = True
    SECRET_KEY: str = "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── PostgreSQL ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:examgpt@2026@db.pyjwdqvrhbmulscjapbr.supabase.co:5432/postgres"

    # ── Redis ───────────────────────────────────────────────────
    REDIS_URL: str = "redis://default:gQAAAAAAAia5AAIgcDEwN2EyY2I4YzY4MDU0MGZjYjllOWI5MGZiNmQ2ZGZlNw@simple-beetle-140985.upstash.io:6379"

    # ── Neo4j ───────────────────────────────────────────────────
    NEO4J_URI: str = "neo4j+s://9ec2908b.databases.neo4j.io"
    NEO4J_USER: str = "9ec2908b"
    NEO4J_PASSWORD: str = "g6jAqrD38VW46e5aAykME2WLx13WKZLXJj1zd4NrS_o"

    # ── Supabase Storage ────────────────────────────────────────
    SUPABASE_URL: str = "https://pyjwdqvrhbmulscjapbr.supabase.co"
    SUPABASE_KEY: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InB5andkcXZyaGJtdWxzY2phcGJyIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc4MzI2NzI0NywiZXhwIjoyMDk4ODQzMjQ3fQ.RZltZUUVUQeo-CL4Aqu75y6lMr6E0NwbIYztpx_vsQo"
    SUPABASE_BUCKET: str = "examgpt-docs"

    # ── Hugging Face Inference ──────────────────────────────────
    HUGGINGFACE_API_KEY: str = ""
    LLM_MODEL: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Upload ──────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 20
    UPLOAD_DIR: str = "/app/uploads"

    # ── Pipeline ────────────────────────────────────────────────
    MAX_EVAL_RETRIES: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
