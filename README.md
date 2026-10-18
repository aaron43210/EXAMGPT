<div align="center">

# 🎓 ExamGPT — Autonomous Multi-Agent GraphRAG & Adaptive Learning Platform

[![Live App](https://img.shields.io/badge/Live%20App-examgpt.netlify.app-00C7B7?style=for-the-badge&logo=netlify&logoColor=white)](https://examgpt.netlify.app)
[![API Status](https://img.shields.io/badge/API-HuggingFace%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/aaronr43210/examgpt-backend)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic%20Workflow-FF6F00?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![Supabase Postgres](https://img.shields.io/badge/Supabase-pgvector%20HNSW-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![Groq LPU](https://img.shields.io/badge/Groq-Llama%203.3%2070B-F55036?style=for-the-badge)](https://groq.com)

**An enterprise-grade, multi-agent AI study companion that turns academic syllabi, lecture notes, and past year question papers into an adaptive, hallucination-resistant exam preparation engine.**

[Live Application](https://examgpt.netlify.app) · [Architecture Guide](ARCHITECTURE.md) · [Deployment Guide](DEPLOYMENT.md) · [API Documentation](#-api-reference)

</div>

---

## 🌟 Key Highlights

* **🤖 Autonomous Multi-Agent Pipeline (LangGraph):** Intent-based dynamic routing across specialized agents: *Intent Router*, *Hybrid Retriever*, *Socratic Tutor*, *Quiz Master*, *Exam Analyst*, *ChitChat*, and an autonomous *Critic* self-correction loop.
* **🔍 Hybrid GraphRAG Engine:** Fuses sparse lexical search (**BM25** Lucene IDF) with dense semantic search (**`pgvector` HNSW** with `BAAI/bge-small-en-v1.5` embeddings) via **Reciprocal Rank Fusion (RRF)**, combined with an entity-relationship **Knowledge Graph** stored in PostgreSQL.
* **⚡ Ultra-Fast, Resilient LLM Ops:** Powered by **Groq LPUs** (`Llama-3.3-70B-versatile` and `Llama-3.1-8B-instant`) with dynamic model discovery, automatic rate-limit failover, and sliding-window token budgeting.
* **📅 Adaptive Exam Engine & Planner:** Dynamically generates day-by-day and phased study schedules calibrated by days-until-exam, student study hours, and historical topic weightage.
* **📊 Rigorous Offline Evaluation Harness:** Quantitative benchmarking suite assessing retrieval (`Hit@k`, `MRR`, `nDCG@k`) and generation quality via an automated **LLM-as-a-Judge** scoring faithfulness and answer relevance.
* **🛡️ Production Security & Safety:** Multi-layered defense featuring Aho-Corasick prompt injection filtering, strict course-level tenant isolation, and JWT session revocation.
* **🚀 100% Free-Tier Cloud Deployment:** React 19 SPA deployed on **Netlify**, backend containerized with **Docker** on **Hugging Face Spaces**, data stored in **Supabase**, and kept alive 24/7 via automated health probes.

---

## 🏛️ Architecture Overview

ExamGPT separates concerns across a modern decoupled architecture:

```
[ React 19 Vite SPA ] (Netlify)
         │  HTTPS / JWT
         ▼
[ FastAPI Gateway + Security Guard ] (Hugging Face Docker)
         │
         ├───▶ [ LangGraph Multi-Agent Workflow ]
         │         ├── Router ──▶ Cache Lookup
         │         ├── Hybrid Retriever (BM25 + pgvector + KG)
         │         ├── Specialists: Tutor | Quiz Master | Exam Analyst
         │         └── Self-Correction Critic Loop ──▶ Output Guard
         │
         ├───▶ [ Persistence Layer ] (Supabase PostgreSQL)
         │         ├── Relational Tables (Users, Courses, Documents)
         │         ├── pgvector HNSW Index (384d Dense Embeddings)
         │         ├── Relational Knowledge Graph (kg_nodes & kg_edges)
         │         └── Semantic Cache (answer_cache)
         │
         └───▶ [ Inference Gateway ] (Groq Cloud)
                   ├── Llama 3.3 70B (Primary Reasoning & Judge)
                   └── Llama 3.1 8B (Fallback & High-Speed Routing)
```

> 📖 *For detailed state machines, retrieval fusion formulas, and database schemas, see [ARCHITECTURE.md](ARCHITECTURE.md).*

---

## 🔬 Evaluation & Benchmarks

ExamGPT includes an automated evaluation harness in [`evals/`](evals/) to ensure zero regressions across releases:

```bash
# Run retrieval ablation benchmarking
python -m evals.run --suite retrieval --config all

# Run safety and prompt injection resistance suite
python -m evals.run --suite safety

# Enforce regression gates against baseline
python -m evals.run --suite retrieval --check
```

### Retrieval Ablation Results

| Pipeline Configuration | Hit@1 | Hit@4 | Recall@4 | MRR | nDCG@4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **BM25 Lexical Only** | 0.68 | 0.84 | 0.72 | 0.74 | 0.76 |
| **Dense Only (`pgvector`)** | 0.74 | 0.89 | 0.81 | 0.80 | 0.82 |
| **Hybrid (BM25 + Dense RRF)** | 0.86 | 0.95 | 0.89 | 0.89 | 0.91 |
| **Hybrid + Reranker + GraphRAG** | **0.92** | **0.98** | **0.94** | **0.93** | **0.95** |

---

## 🛠️ Tech Stack

| Domain | Technology |
| :--- | :--- |
| **Frontend** | React 19, Vite, React Router, React Bootstrap, React Markdown, Axios |
| **Backend Framework** | FastAPI, Pydantic v2, Uvicorn, Python 3.11 |
| **Agent Orchestration** | LangGraph, LangChain Core |
| **Database & Vector Store** | PostgreSQL 15, `pgvector` extension with HNSW cosine index, SQLAlchemy, Alembic |
| **Information Retrieval** | BAAI/bge-small-en-v1.5 (FastEmbed ONNX), BM25 (Lucene IDF), Reciprocal Rank Fusion |
| **Inference & LLMs** | Groq Cloud LPU (Meta Llama 3.3 70B Versatile, Llama 3.1 8B Instant) |
| **Deployment & Hosting** | Netlify (Frontend), Hugging Face Spaces Docker (Backend), Supabase (Database) |
| **Reliability & Monitoring** | UptimeRobot (24/7 `/readyz` probes), custom sliding-window rate limiters |

---

## 🚀 Local Development Setup

### Prerequisites
* Python 3.11+
* Node.js 18+
* PostgreSQL with `pgvector` enabled (or a free Supabase instance)
* Free Groq Cloud API Key ([console.groq.com](https://console.groq.com))

### 1. Backend Setup
```bash
# Clone the repository
git clone https://github.com/aaron43210/EXAMGPT.git
cd EXAMGPT

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your DATABASE_URL and GROQ_API_KEY

# Run database migrations
alembic upgrade head

# Start backend server
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd EXAMGPT-Frontend
npm install
npm run dev
```

The application will be accessible at `http://localhost:5173`.

---

## 📖 API Reference

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :---: |
| `POST` | `/api/v1/auth/register` | Register a new user | No |
| `POST` | `/api/v1/auth/login` | Authenticate and obtain JWT bearer token | No |
| `POST` | `/api/v1/auth/logout` | Invalidate all active user tokens | Yes |
| `GET` | `/api/v1/courses/` | List all enrolled courses | Yes |
| `POST` | `/api/v1/courses/` | Create a new course workspace | Yes |
| `POST` | `/api/v1/courses/{id}/documents` | Upload and asynchronously index course PDFs | Yes |
| `POST` | `/api/v1/courses/{id}/query` | Ask a question via the multi-agent pipeline | Yes |
| `POST` | `/api/v1/courses/{id}/study-plan` | Generate an adaptive, schedule-aware study plan | Yes |
| `GET` | `/api/v1/courses/{id}/practice/quiz` | Generate topic-specific practice quizzes | Yes |
| `GET` | `/health`, `/readyz` | System liveness and database/model readiness probes | No |

---

## 🛡️ Security & Privacy

* **Zero Data Retention for Training:** All inference is executed through enterprise Groq LPUs without data training retention.
* **Strict Tenant Isolation:** Data access is strictly compartmentalized by `user_id` and `course_id` with cascaded foreign key protection.
* **Aho-Corasick Input Sanitization:** Inspects uploads for prompt injection strings, canary markers, and jailbreak vectors.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

<div align="center">
Built with passion by <b>Aaron R.</b>
</div>
