# AI Doc Q&A

Ask questions about any PDF in plain language. Answers are grounded in the document using RAG.

**Live:** frontend on Vercel · backend on Railway

## Stack

- **Frontend:** React + TypeScript + Vite
- **Backend:** FastAPI + sentence-transformers + Gemini 3 Flash

## How it works

1. Upload a PDF → text is extracted, chunked, and embedded locally
2. Ask a question → top-4 relevant chunks are retrieved by cosine similarity
3. Chunks + question are sent to Gemini → answer returned with sources

## Run locally

```bash
# Backend
cd backend
cp .env.example .env   # add your GEMINI_API_KEY
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev            # proxies /upload and /ask to localhost:8000
```

## Deploy

| Part | Service | Env var |
|---|---|---|
| Backend | Railway | `GEMINI_API_KEY` |
| Frontend | Vercel | `VITE_API_URL` = Railway URL |

See `CODE_GUIDE.md` for a full explanation of the codebase.
