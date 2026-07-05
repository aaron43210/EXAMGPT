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
    SECRET_KEY: str = "REDACTED_SECRET_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── PostgreSQL ──────────────────────────────────────────────
    DATABASE_URL: str = "postgresql://postgres:REDACTED_PASSWORD@db.pyjwdqvrhbmulscjapbr.supabase.co:5432/postgres"

    # ── Redis ───────────────────────────────────────────────────
    REDIS_URL: str = "redis://default:REDACTED_PASSWORD@simple-beetle-140985.upstash.io:6379"

    # ── Neo4j ───────────────────────────────────────────────────
    NEO4J_URI: str = "neo4j+s://9ec2908b.databases.neo4j.io"
    NEO4J_USER: str = "9ec2908b"
    NEO4J_PASSWORD: str = "g6jAqrD38VW46e5aAykME2WLx13WKZLXJj1zd4NrS_o"

    # ── Supabase Storage ────────────────────────────────────────
    SUPABASE_URL: str = "https://pyjwdqvrhbmulscjapbr.supabase.co"
    SUPABASE_KEY: str = "REDACTED_SUPABASE_KEY"
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
