---
title: ExamGPT Backend
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# ExamGPT Backend API

FastAPI backend for ExamGPT, an AI-powered exam preparation platform.
It uses Groq (Llama 3.1) for answers and study plans, and course-scoped hybrid retrieval
(BM25 plus local embeddings) over chunks stored in Postgres.

Frontend: https://examgpt.netlify.app · Deployment guide: [DEPLOYMENT.md](DEPLOYMENT.md)

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/auth/register` | Create an account |
| POST | `/api/v1/auth/login` | Get a bearer token (form: `username`, `password`) |
| POST | `/api/v1/auth/logout` | Revoke all of the user's tokens |
| GET | `/api/v1/auth/me` | Current user |
| GET/POST | `/api/v1/courses/` | List / create courses |
| GET/DELETE | `/api/v1/courses/{id}` | Get / delete a course and everything in it |
| GET/POST | `/api/v1/courses/{id}/documents` | List / upload PDFs (indexed in the background) |
| DELETE | `/api/v1/courses/{id}/documents/{doc_id}` | Delete a document |
| POST | `/api/v1/courses/{id}/query` | Ask a question |
| GET/DELETE | `/api/v1/courses/{id}/chat-history` | Read / clear chat history |
| GET/POST | `/api/v1/courses/{id}/study-plan` | Read / generate a study plan |
| GET | `/health`, `/readyz` | Liveness / readiness (`/readyz` checks the database) |

## Local development

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # add your GROQ_API_KEY
alembic upgrade head
uvicorn app.main:app --reload --port 8000
pytest -q                     # 49 tests; TEST_DATABASE_URL=postgresql://... to run on Postgres
```
