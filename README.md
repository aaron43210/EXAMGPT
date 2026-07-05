---
title: ExamGPT Backend
emoji: 🎓
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# ExamGPT Backend API

FastAPI backend for ExamGPT - an AI-powered exam preparation platform.

## API Endpoints

- `POST /api/v1/auth/register` - Register a new user
- `POST /api/v1/auth/login` - Login
- `GET /api/v1/courses` - List courses
- `POST /api/v1/documents/upload/{course_id}` - Upload a document
- `POST /api/v1/query` - Ask the AI a question
