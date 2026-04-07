import json
import os
import uuid
from contextlib import asynccontextmanager
from io import BytesIO

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai
from google.genai import errors as genai_errors
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

# ── Startup: load models after the port is bound ─────────────────────────────
# Using lifespan so uvicorn binds to the port before the slow model download,
# avoiding Render's port-scan timeout.

embed_model: SentenceTransformer | None = None
gemini_client: genai.Client | None = None
GEMINI_MODEL = "gemini-3-flash-preview"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global embed_model, gemini_client
    print("Loading embedding model…")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # local, runs on CPU
    print("Embedding model ready.")
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    yield

app = FastAPI(title="AI Doc Q&A", lifespan=lifespan)

# Allow all origins so the Vercel frontend can reach this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session store ───────────────────────────────────────────────────
# Each uploaded PDF gets a session_id; data lives in RAM until the server restarts
sessions: dict[str, dict] = {}

CHUNK_SIZE    = 350   # words per chunk
CHUNK_OVERLAP = 60    # overlap between consecutive chunks
TOP_K         = 4     # number of chunks retrieved per query


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract plain text from all pages of a PDF."""
    pages = []
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        page_count = len(pdf.pages)
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())
    return "\n\n".join(pages), page_count


def chunk_text(text: str) -> list[str]:
    """Split text into overlapping word-based chunks for better retrieval."""
    words = text.split()
    chunks: list[str] = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def cosine_top_k(query_emb: np.ndarray, doc_embs: np.ndarray, k: int) -> list[int]:
    """Return indices of the k chunks most similar to the query (cosine similarity)."""
    q = query_emb / (np.linalg.norm(query_emb) + 1e-10)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-10)
    scores = (d @ q).flatten()
    return list(np.argsort(scores)[::-1][:k])


QUOTA_MSG = (
    "This free service has a usage limit shared across all users. "
    "Please try again in a few minutes. "
    "If you need priority access, contact the developer at rafatrik@gmail.com"
)

def stream_prompt(prompt: str):
    """Generator that yields SSE chunks from a Gemini streaming call."""
    try:
        for chunk in gemini_client.models.generate_content_stream(model=GEMINI_MODEL, contents=prompt):
            if chunk.text:
                yield f"data: {json.dumps({'type': 'chunk', 'text': chunk.text})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
    except genai_errors.ClientError as e:
        if e.status_code == 429:
            yield f"data: {json.dumps({'type': 'error', 'message': QUOTA_MSG})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'error', 'message': 'An error occurred. Please try again.'})}\n\n"


# ── Request models ────────────────────────────────────────────────────────────

class SessionRequest(BaseModel):
    session_id: str

class AskRequest(BaseModel):
    session_id: str
    question: str

class CompareRequest(BaseModel):
    session_id_a: str
    session_id_b: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/ai")
def health_ai():
    """Check Gemini availability with a minimal test call."""
    try:
        gemini_client.models.generate_content(model=GEMINI_MODEL, contents=".")
        return {"available": True}
    except genai_errors.ClientError as e:
        if e.status_code == 429:
            return {
                "available": False,
                "message": "The service has reached its usage limit shared across all users. Please try again in a few minutes.",
            }
        return {"available": False, "message": "AI service temporarily unavailable."}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Receive a PDF, extract + embed its text, and return a session_id."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    data = await file.read()
    text, page_count = extract_text(data)

    if not text.strip():
        raise HTTPException(422, "Could not extract text from this PDF. It may be scanned/image-only.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(422, "Document appears to be empty after extraction.")

    # Embed all chunks locally (no API call needed)
    embeddings = embed_model.encode(chunks, show_progress_bar=False, batch_size=32)

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "chunks": chunks,
        "embeddings": np.array(embeddings),
        "filename": file.filename,
        "page_count": page_count,
    }

    return {
        "session_id": session_id,
        "filename": file.filename,
        "page_count": page_count,
        "chunks": len(chunks),
    }


@app.post("/ask")
async def ask(req: AskRequest):
    """Stream the answer token-by-token using Server-Sent Events."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload the PDF again.")
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty.")

    # Embed the question and retrieve the top-k most similar chunks
    q_emb = embed_model.encode([req.question])[0]
    top_idx = cosine_top_k(q_emb, session["embeddings"], TOP_K)
    context_chunks = [session["chunks"][i] for i in top_idx]
    context = "\n\n---\n\n".join(context_chunks)

    # Build the RAG prompt: context + question
    prompt = f"""You are a precise assistant that answers questions strictly based on the provided document context.
If the answer is not contained in the context, say "I couldn't find that information in the document."
Do not make up information. Be concise and clear.

Document context:
{context}

Question: {req.question}

Answer:"""

    def generate():
        # Send source chunks first so the frontend can display them immediately
        yield f"data: {json.dumps({'type': 'sources', 'sources': context_chunks})}\n\n"
        yield from stream_prompt(prompt)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/suggest")
async def suggest(req: SessionRequest):
    """Generate 5 relevant questions based on the document content."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    # Sample from the first few chunks to understand the document
    sample = " ".join(session["chunks"][:5])

    prompt = f"""Based on this document excerpt, generate exactly 5 insightful questions a reader might want to ask.
Return ONLY a JSON array of strings, no explanation, no markdown fences.
Example: ["What is...?", "How does...?", "When was...?", "Why did...?", "What are...?"]

Document excerpt:
{sample}

Questions JSON:"""

    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        text = response.text.strip().strip("```json").strip("```").strip()
        questions = json.loads(text)
        return {"questions": questions[:5]}
    except genai_errors.ClientError as e:
        if e.status_code == 429:
            raise HTTPException(429, QUOTA_MSG)
        raise HTTPException(500, "Could not generate suggestions.")


@app.post("/summary")
async def summary(req: SessionRequest):
    """Generate a structured auto-summary of the document."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    # Use the first 20 chunks (~7000 words) for the summary
    text = " ".join(session["chunks"][:20])

    prompt = f"""Analyze this document and provide a structured summary with these sections:

**Document type** — what kind of document is this?
**Purpose** — what is it about? (1-2 sentences)
**Key entities** — people, organizations, dates, amounts mentioned
**Main points** — up to 5 bullet points
**Important dates or deadlines** — if any
**Conclusions or action items** — if any

Document:
{text}

Structured summary:"""

    return StreamingResponse(stream_prompt(prompt), media_type="text/event-stream")


@app.post("/compare")
async def compare(req: CompareRequest):
    """Compare two uploaded documents."""
    session_a = sessions.get(req.session_id_a)
    session_b = sessions.get(req.session_id_b)
    if not session_a:
        raise HTTPException(404, "First document session not found.")
    if not session_b:
        raise HTTPException(404, "Second document session not found.")

    text_a = " ".join(session_a["chunks"][:15])
    text_b = " ".join(session_b["chunks"][:15])

    prompt = f"""Compare these two documents thoroughly:

**DOCUMENT A** ({session_a["filename"]}):
{text_a}

**DOCUMENT B** ({session_b["filename"]}):
{text_b}

Provide a structured comparison covering:
- **Main topic / purpose** of each document
- **Key similarities**
- **Key differences**
- **Which is more detailed / complete** and why
- **Any conflicting information** between them

Comparison:"""

    return StreamingResponse(stream_prompt(prompt), media_type="text/event-stream")


@app.post("/contradictions")
async def contradictions(req: SessionRequest):
    """Detect contradictions, inconsistencies, and ambiguities in the document."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(404, "Session not found.")

    text = " ".join(session["chunks"][:20])

    prompt = f"""Carefully analyze this document and identify:

- **Contradictions** — statements that directly conflict with each other
- **Inconsistencies** — data or facts that don't align (dates, numbers, names)
- **Ambiguities** — unclear or vague statements that could be misinterpreted
- **Missing information** — important gaps that should be addressed

For each issue found, quote the relevant text and explain the problem.
If the document appears consistent and clear, state that explicitly.

Document:
{text}

Analysis:"""

    return StreamingResponse(stream_prompt(prompt), media_type="text/event-stream")
