# Deploying ExamGPT

| Piece | Where | URL |
|---|---|---|
| Database | Supabase Postgres (free) | — |
| Backend | Hugging Face Space (Docker) | https://aaronr43210-examgpt-backend.hf.space |
| Frontend | Netlify | https://examgpt.netlify.app |
| LLM | Groq | — |

## 1. Database (Supabase)

1. Create a project at https://supabase.com, or reuse the existing one. Save the database password.
2. Go to **Project Settings → Database → Connection string → URI** and pick **Session pooler** (port 5432).
   It looks like `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   URL-encode any special characters in the password.
3. You don't need to create any tables. The backend runs `alembic upgrade head` on every start.

> Upgrading an existing database: the migration keeps users, courses and chat history.
> Documents uploaded before this version were only indexed in server memory, so they are
> marked **"Please re-upload this file"**. Ask users to re-upload them once.

## 2. Groq API key

Go to https://console.groq.com/keys, choose **Create API key**, and copy it (it starts with `gsk_`).
The free tier is rate-limited (about 6k tokens per minute on `llama-3.1-8b-instant`). When the
limit is hit, the app shows a friendly "please wait a minute" message.

## 3. Backend (Hugging Face Space)

### Secrets

In the Space, go to **Settings → Variables and secrets → New secret** and add:

| Name | Value |
|---|---|
| `SECRET_KEY` | output of `python -c "import secrets; print(secrets.token_urlsafe(48))"` |
| `DATABASE_URL` | the Supabase URI from step 1 |
| `GROQ_API_KEY` | `gsk_...` |
| `FRONTEND_URL` | `https://examgpt.netlify.app` |

Optional **variables** (not secret):

| Name | Default | Notes |
|---|---|---|
| `GROQ_MODEL` | `llama-3.1-8b-instant` | `llama-3.3-70b-versatile` gives better answers but has lower free limits |
| `EVAL_ENABLED` | `false` | `true` adds a self-review pass, which costs a second Groq call per question |
| `CORS_EXTRA_ORIGINS` | — | comma-separated, e.g. Netlify deploy-preview URLs |
| `ENABLE_DOCS` | `false` | `true` exposes `/docs` |

Remove the old `HUGGINGFACE_API_KEY`, `REDIS_URL`, `NEO4J_*` and `SUPABASE_KEY` secrets. They are no longer used.

The server **refuses to start** in production if `SECRET_KEY` is weak, `DATABASE_URL` is missing,
or `GROQ_API_KEY` or `FRONTEND_URL` is missing. The reason appears in the Space's **Logs** tab.

### Push the code

The Space is its own git repository. Either:

**Manual:**
```bash
cd EXAMGPT
git remote add space https://huggingface.co/spaces/aaronr43210/examgpt-backend   # once
git push space main            # username: aaronr43210, password: an HF *write* access token
```

**Automatic (recommended):** create a write token at https://huggingface.co/settings/tokens,
and add it to the GitHub repo `aaron43210/EXAMGPT` under **Settings → Secrets → Actions** as `HF_TOKEN`.
After that, every push to `main` runs the tests (SQLite and Postgres), builds the Docker image,
and then deploys to the Space (`.github/workflows/ci.yml`).

### Verify

```bash
curl https://aaronr43210-examgpt-backend.hf.space/health   # {"status":"ok"}
curl https://aaronr43210-examgpt-backend.hf.space/readyz   # {"status":"ready","llm_configured":true}
```

The first build takes about 5 minutes, because it downloads the embedding model into the image.

## 4. Frontend (Netlify)

1. Go to **Netlify → Site configuration → Environment variables** and add
   `VITE_API_URL = https://aaronr43210-examgpt-backend.hf.space` (no trailing slash).
   The build fails on purpose if this is missing.
2. Under **Build settings**, the base directory is the repo root. The build command and publish directory
   come from `netlify.toml` (`npm ci && npm run lint && npm test && npm run build`, publishing `dist`).
3. Push to `main` on `aaron43210/EXAMGPT-Frontend`. Netlify builds and deploys automatically.

If the backend URL ever changes, also update `connect-src` in `netlify.toml`, because the
Content-Security-Policy only allows the frontend to call that one origin.

## 5. Keeping it live

- **Free HF Spaces sleep** after about 48h without traffic, and **free Supabase projects pause** after 7 days.
  `/readyz` touches the database, so one ping keeps both awake:
  - `.github/workflows/keepalive.yml` pings it every 6 hours.
  - Also add a free **UptimeRobot** HTTP monitor on
    `https://aaronr43210-examgpt-backend.hf.space/readyz` every 5 minutes. It doubles as downtime email
    alerts, and GitHub disables scheduled workflows after 60 days with no commits.
- If the Space is asleep anyway, the frontend shows "Waking up the server…" and retries automatically.
- For guaranteed uptime with no cold starts, upgrade the Space hardware (paid CPU never sleeps).

## 6. Post-deploy checklist

- [ ] `/readyz` returns `ready` with `llm_configured: true`
- [ ] Register, log in, create a course
- [ ] Upload a PDF; its status goes `processing` → `ready`
- [ ] Ask a question; the answer cites the PDF
- [ ] Generate a study plan
- [ ] Restart the Space (Settings → Restart) and ask again; the answer still cites the PDF
- [ ] Log out; the old token no longer works
- [ ] UptimeRobot monitor is green

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Space shows "Runtime error" | Open **Logs**. `Invalid production configuration: ...` means a missing secret. `alembic` errors usually mean a bad `DATABASE_URL` or a password that isn't URL-encoded. |
| Frontend says "Can't reach the server" | Space is asleep or crashed. Check `/health`. Check `VITE_API_URL`. |
| Browser console shows a CORS error | `FRONTEND_URL` doesn't exactly match the site origin (`https://examgpt.netlify.app`, no trailing slash). |
| Browser console shows a CSP `connect-src` error | The backend URL differs from the one in `netlify.toml`. |
| "AI model is receiving too many requests" | Groq free-tier limit. Wait a minute, or use a paid Groq plan. |
| Document stuck in `failed` | The message explains why: scanned/image-only PDF, password-protected PDF, or more than 500 pages. |
