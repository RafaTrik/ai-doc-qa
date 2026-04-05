# Code Guide — AI Doc Q&A

## What it does
Upload a PDF → ask questions in plain language → get answers grounded in the document.
Uses **RAG** (Retrieval-Augmented Generation): instead of sending the whole PDF to the AI, it finds only the relevant parts first.

---

## Architecture

```
User → React frontend (Vercel)
            ↓  HTTP
       FastAPI backend (Railway)
            ↓  local        ↓  API call
  sentence-transformers   Gemini (Google)
  (embeddings/search)     (answer generation)
```

---

## Backend (`backend/main.py`)

### Flow when a PDF is uploaded (`POST /upload`)
1. **Extract text** — `pdfplumber` reads each page and concatenates the text.
2. **Chunk** — the text is split into overlapping 350-word chunks so no context is lost at boundaries.
3. **Embed** — `sentence-transformers` converts every chunk into a vector (a list of numbers that captures its meaning). Runs locally, no API needed.
4. **Store** — chunks + vectors are saved in RAM under a UUID (`session_id`) returned to the frontend.

### Flow when a question is asked (`POST /ask`)
1. **Embed the question** — same model converts the question into a vector.
2. **Retrieve** — cosine similarity finds the 4 chunks most relevant to the question.
3. **Prompt Gemini** — the 4 chunks are injected into a prompt as context, then sent to Gemini 3 Flash.
4. **Return** — the answer + the source chunks are sent back to the frontend.

### Key constants
| Constant | Value | Why |
|---|---|---|
| `CHUNK_SIZE` | 350 words | Fits in context without losing meaning |
| `CHUNK_OVERLAP` | 60 words | Avoids cutting sentences at chunk boundaries |
| `TOP_K` | 4 | Enough context without overloading the prompt |

---

## Frontend (`frontend/src/App.tsx`)

Two screens controlled by the `doc` state:

- **`doc === null`** → upload screen (drag & drop or file picker)
- **`doc !== null`** → chat screen

### State variables
| Variable | Purpose |
|---|---|
| `doc` | Metadata of the uploaded PDF (session_id, filename, pages) |
| `messages` | Chat history array |
| `uploading / thinking` | Loading flags to show spinners |
| `error` | Error message shown as a toast |

### API calls
- `POST /upload` — sends the PDF as `multipart/form-data`, gets back `DocInfo`
- `POST /ask` — sends `{ session_id, question }`, gets back `{ answer, sources }`

The base URL is read from `VITE_API_URL` (set in Vercel). In local dev it's empty and Vite's proxy forwards requests to `localhost:8000`.

---

## Deploy

| Part | Service | Trigger |
|---|---|---|
| Frontend | Vercel | Push to `main` → auto build & deploy |
| Backend | Railway | Push to `main` → auto deploy |

### Environment variables
- **Railway**: `GEMINI_API_KEY`
- **Vercel**: `VITE_API_URL` = Railway backend URL (e.g. `https://ai-doc-qa.railway.app`)
