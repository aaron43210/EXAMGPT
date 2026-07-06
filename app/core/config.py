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
    SECRET_KEY: str = "changeme-generate-a-real-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # ── PostgreSQL ──────────────────────────────────────────────
    DATABASE_URL: str = ""

    # ── Redis ───────────────────────────────────────────────────
    REDIS_URL: str = ""

    # ── Neo4j ───────────────────────────────────────────────────
    NEO4J_URI: str = ""
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""

    # ── Supabase Storage ────────────────────────────────────────
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    SUPABASE_BUCKET: str = "examgpt-docs"

    # ── Hugging Face Inference ──────────────────────────────────
    HUGGINGFACE_API_KEY: str = ""
    LLM_MODEL: str = "HuggingFaceH4/zephyr-7b-beta"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ── Upload ──────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 20
    UPLOAD_DIR: str = "/app/uploads"
    
    # ── CORS ────────────────────────────────────────────────────
    FRONTEND_URL: str = ""

    # ── Pipeline ────────────────────────────────────────────────
    MAX_EVAL_RETRIES: int = 3

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()
