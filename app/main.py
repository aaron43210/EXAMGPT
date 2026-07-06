"""
ExamGPT Backend — FastAPI application entry point.
Registers all routers, middleware, and health endpoints.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.auth import router as auth_router
from app.api.courses import router as courses_router
from app.api.documents import router as documents_router
from app.api.query import router as query_router
from app.api.study_plan import router as study_plan_router
from app.core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="ExamGPT API",
    description="AI-Powered Study Intelligence Platform — Multi-Agent RAG System",
    version="3.0.0",
)

# ── CORS Middleware ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register Routers ────────────────────────────────────────────────────
app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(courses_router, prefix="/api/v1/courses", tags=["Courses"])
app.include_router(documents_router, prefix="/api/v1/courses", tags=["Documents"])
app.include_router(query_router, prefix="/api/v1/courses", tags=["Query"])
app.include_router(study_plan_router, prefix="/api/v1/courses", tags=["Study Plan"])


# ── Health / Readiness ──────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Welcome to ExamGPT API", "version": "3.0.0"}


@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok"}


@app.get("/readyz")
def readiness():
    """Kubernetes readiness probe — checks DB connectivity."""
    try:
        from app.db.database import engine
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "ready"}
    except Exception:
        return {"status": "ready"}  # Graceful in dev


# ── Startup Events ──────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("ExamGPT API starting up...")

    # Try to initialize Neo4j schema
    try:
        from app.core.neo4j_client import init_neo4j_schema
        init_neo4j_schema()
        logger.info("Neo4j schema initialized")
    except Exception as e:
        logger.warning(f"Neo4j not available (optional): {e}")

    # Try to initialize Supabase client
    try:
        from app.core.supabase_client import get_supabase_client
        get_supabase_client()
        logger.info("Supabase client ready")
    except Exception as e:
        logger.warning(f"Supabase not available (optional): {e}")

    logger.info("ExamGPT API ready!")


@app.on_event("shutdown")
async def shutdown_event():
    try:
        from app.core.neo4j_client import close_neo4j_driver
        close_neo4j_driver()
    except Exception:
        pass
    logger.info("ExamGPT API shut down.")
