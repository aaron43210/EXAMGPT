"""
Centralized configuration via Pydantic Settings.
All secrets and connection strings are pulled from environment variables.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_SECRET = "dev-insecure-secret-key-change-me-0000000000"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="ignore")

    # ── App ─────────────────────────────────────────────────────
    APP_NAME: str = "ExamGPT"
    ENVIRONMENT: str = "development"  # development | production | test
    SECRET_KEY: str = _DEFAULT_SECRET
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720
    ENABLE_DOCS: bool = False

    # ── Database (Supabase Postgres in production) ──────────────
    DATABASE_URL: str = "sqlite:///./examgpt.db"

    # ── Groq LLM ────────────────────────────────────────────────
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    LLM_TIMEOUT_SECONDS: int = 60

    # ── Embeddings (local, optional; BM25 is always available) ──
    EMBEDDINGS_ENABLED: bool = True
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"

    # ── Upload ──────────────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 20
    MAX_PDF_PAGES: int = 500
    MAX_DOCUMENTS_PER_COURSE: int = 30

    # ── CORS ────────────────────────────────────────────────────
    FRONTEND_URL: str = ""
    # Comma-separated extra origins, e.g. Netlify deploy previews
    CORS_EXTRA_ORIGINS: str = ""

    # Fallback when the primary model is rate-limited (separate Groq quota). Empty = none.
    GROQ_FALLBACK_MODEL: str = "llama-3.3-70b-versatile"
    # Client-side token budgets (tokens/minute) so we fail fast instead of hammering Groq
    GROQ_TPM_LIMIT: int = 6000
    GROQ_FALLBACK_TPM_LIMIT: int = 12000
    # Model used as LLM-as-judge (online + offline evals). Different from the answer model
    # to reduce self-preference bias.
    JUDGE_MODEL: str = "llama-3.3-70b-versatile"

    # ── Retrieval ───────────────────────────────────────────────
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    GRAPH_RAG_ENABLED: bool = True
    BM25_ENABLED: bool = True  # ablation switch for the eval harness; keep on in production

    # ── Semantic answer cache ───────────────────────────────────
    # Repeated / paraphrased questions are answered from cache without an LLM call.
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_THRESHOLD: float = 0.95  # cosine similarity between question embeddings
    SEMANTIC_CACHE_MIN_CONFIDENCE: float = 0.5  # only cache answers backed by good retrieval
    SEMANTIC_CACHE_TTL_HOURS: int = 168

    # ── Pipeline ────────────────────────────────────────────────
    EVAL_ENABLED: bool = False  # inline critic: costs a 2nd Groq call when confidence is low
    MAX_EVAL_RETRIES: int = 1
    LOW_CONFIDENCE_THRESHOLD: float = 0.35
    ROUTER_LLM_ENABLED: bool = False  # rules route most requests; LLM only for ambiguous ones
    # Fraction of answers scored for groundedness in the background (online eval)
    ONLINE_EVAL_SAMPLE_RATE: float = 0.1

    # ── Rate limits (requests per minute) ───────────────────────
    RATE_LIMIT_AUTH_PER_MIN: int = 10
    # X-Forwarded-For can be spoofed, so per-IP limits get a global backstop
    RATE_LIMIT_REGISTER_GLOBAL_PER_MIN: int = 30
    RATE_LIMIT_QUERY_PER_MIN: int = 10
    RATE_LIMIT_UPLOAD_PER_MIN: int = 5

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT.lower() == "production"

    @property
    def sqlalchemy_url(self) -> str:
        url = self.DATABASE_URL.strip()
        # Supabase / Heroku style URLs use the deprecated "postgres://" scheme
        if url.startswith("postgres://"):
            url = "postgresql://" + url[len("postgres://"):]
        return url

    @property
    def cors_origins(self) -> list[str]:
        origins = [o.strip().rstrip("/") for o in self.CORS_EXTRA_ORIGINS.split(",") if o.strip()]
        if self.FRONTEND_URL:
            origins.append(self.FRONTEND_URL.strip().rstrip("/"))
        if not self.is_production:
            origins += ["http://localhost:3000", "http://localhost:5173"]
        return origins

    def validate_for_production(self) -> None:
        """Refuse to boot with insecure settings in production."""
        if not self.is_production:
            return
        problems = []
        if self.SECRET_KEY == _DEFAULT_SECRET or len(self.SECRET_KEY) < 32:
            problems.append("SECRET_KEY must be set to a random value of at least 32 characters")
        if self.sqlalchemy_url.startswith("sqlite"):
            problems.append("DATABASE_URL must point to Postgres in production")
        if not self.GROQ_API_KEY:
            problems.append("GROQ_API_KEY is required")
        if not self.FRONTEND_URL:
            problems.append("FRONTEND_URL is required for CORS")
        if problems:
            raise RuntimeError("Invalid production configuration: " + "; ".join(problems))


@lru_cache()
def get_settings() -> Settings:
    return Settings()
