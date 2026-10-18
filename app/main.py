"""
ExamGPT Backend — FastAPI application entry point.
Registers all routers, middleware, error handlers and health endpoints.
"""
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from app.api.auth import router as auth_router
from app.api.courses import router as courses_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.study_plan import router as study_plan_router
from app.agents.llm_setup import LLMError
from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()
settings.validate_for_production()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ExamGPT API starting (env=%s, model=%s)", settings.ENVIRONMENT, settings.GROQ_MODEL)
    if settings.EMBEDDINGS_ENABLED:
        from app.ingestion.embedder import warmup
        # Load the embedding model off the request path
        threading.Thread(target=warmup, daemon=True).start()
    yield
    logger.info("ExamGPT API shut down.")


app = FastAPI(
    title="ExamGPT API",
    description="AI-Powered Study Intelligence Platform — Multi-Agent RAG System",
    version="4.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url=None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,  # bearer tokens, no cookies
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    if not request.url.path.startswith("/docs"):
        response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors https://huggingface.co"
        response.headers["Cache-Control"] = "no-store"
    return response


# ── Error handlers: never leak internals to clients ─────────────────────
@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.user_message},
                        headers={"Retry-After": "30"})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errors = []
    for e in exc.errors():
        field = ".".join(str(p) for p in e.get("loc", []) if p not in ("body", "query", "path", "form"))
        msg = str(e.get("msg", "Invalid value")).removeprefix("Value error, ")
        errors.append(f"{field}: {msg}" if field else msg)
    return JSONResponse(status_code=422, content={"detail": "; ".join(errors) or "Invalid request"})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", "-")
    logger.exception("Unhandled error [request_id=%s] %s %s", request_id, request.method, request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": f"Something went wrong on our side. Reference: {request_id}"})


# ── Register Routers ────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(courses_router, prefix="/api/v1/courses", tags=["Courses"])
app.include_router(documents_router, prefix="/api/v1/courses", tags=["Documents"])
app.include_router(query_router, prefix="/api/v1/courses", tags=["Query"])
app.include_router(study_plan_router, prefix="/api/v1/courses", tags=["Study Plan"])


# ── Health / Readiness ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to ExamGPT API", "version": app.version}


@app.get("/health")
@app.get("/healthz")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/readyz")
def readiness():
    """Readiness: the database is reachable. Ping this to keep Space + Supabase awake."""
    from app.db.database import engine
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("Readiness check failed: %s", e)
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ready", "llm_configured": bool(settings.GROQ_API_KEY)}
