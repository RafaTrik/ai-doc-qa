import os
import uuid
from io import BytesIO

import numpy as np
import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="AI Doc Q&A")

# Allow all origins so the Vercel frontend can reach this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models (loaded once at startup) ──────────────────────────────────────────

print("Loading embedding model…")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # local, runs on CPU
print("Embedding model ready.")

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
GEMINI_MODEL = "gemini-3-flash-preview"

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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


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


class AskRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    """Find the most relevant chunks and send them to Gemini to answer the question."""
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

    response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)

    return {
        "answer": response.text.strip(),
        "sources": context_chunks,
    }
