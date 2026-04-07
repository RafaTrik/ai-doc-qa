import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

// In production (Vercel), set VITE_API_URL to the Railway backend URL.
// In dev, leave it unset — Vite's proxy forwards /upload and /ask to localhost:8000.
const API = import.meta.env.VITE_API_URL ?? ''

// ── Types ─────────────────────────────────────────────────────────────────────

interface DocInfo {
  session_id: string
  filename: string
  page_count: number
  chunks: number
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: string[]
  showSources?: boolean
}

const SUGGESTIONS = [
  'What is this document about?',
  'Summarise the key points.',
  'What are the main conclusions?',
]

// ── Component ─────────────────────────────────────────────────────────────────

export default function App() {
  const [doc, setDoc]           = useState<DocInfo | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [uploading, setUploading] = useState(false)
  const [thinking, setThinking]   = useState(false)
  const [dragOver, setDragOver]   = useState(false)
  const [error, setError]         = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef       = useRef<HTMLTextAreaElement>(null)

  const scrollToBottom = () =>
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)

  // ── Upload ──────────────────────────────────────────────────────────────────

  const handleFile = async (file: File) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported.')
      return
    }
    setError(null)
    setUploading(true)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed.' }))
        throw new Error(err.detail ?? 'Upload failed.')
      }
      const data: DocInfo = await res.json()
      setDoc(data)
      setMessages([])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.')
    } finally {
      setUploading(false)
    }
  }

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
    e.target.value = ''
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  // ── Ask ─────────────────────────────────────────────────────────────────────

  const sendQuestion = async (q: string) => {
    if (!doc || !q.trim() || thinking) return
    setError(null)

    setMessages(prev => [...prev, { role: 'user', content: q }])
    setQuestion('')
    setThinking(true)
    scrollToBottom()

    try {
      const res = await fetch(`${API}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: doc.session_id, question: q }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Request failed.' }))
        throw new Error(err.detail ?? 'Request failed.')
      }

      // Add empty assistant message that will be filled progressively
      setMessages(prev => [...prev, { role: 'assistant', content: '', sources: [], showSources: false }])
      setThinking(false)

      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''  // keep incomplete last line in buffer

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const event = JSON.parse(line.slice(6))

          if (event.type === 'chunk') {
            // Append token to the last message
            setMessages(prev => prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, content: m.content + event.text } : m
            ))
            scrollToBottom()
          } else if (event.type === 'sources') {
            // Attach sources to the last message
            setMessages(prev => prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, sources: event.sources } : m
            ))
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Something went wrong.')
      setThinking(false)
    } finally {
      scrollToBottom()
    }
  }

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuestion(question)
    }
  }

  const toggleSources = (idx: number) => {
    setMessages(prev =>
      prev.map((m, i) => (i === idx ? { ...m, showSources: !m.showSources } : m))
    )
  }

  const reset = () => {
    setDoc(null)
    setMessages([])
    setError(null)
  }

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="app">
      {/* Nav */}
      <nav className="nav">
        <a href="https://rafatrik.com" className="nav-logo">rpp<span>._</span></a>
        <span className="nav-sep">/</span>
        <span className="nav-title">AI Doc Q&amp;A</span>
        <div className="nav-links">
          <a href="https://rafatrik.com/#projects-section">Projects</a>
          <a href="https://github.com/rafatrik" target="_blank" rel="noopener">GitHub</a>
          <a href="https://linkedin.com/in/rafatrik" target="_blank" rel="noopener">LinkedIn</a>
        </div>
      </nav>

      {/* Error */}
      {error && <div className="error-toast">{error}</div>}

      {/* Upload screen */}
      {!doc && (
        <div className="upload-screen">
          <div className="upload-label">AI Document Q&amp;A</div>
          <h1>Ask anything about<br />your PDF</h1>
          <p>Upload a PDF and ask questions in plain language. Powered by RAG — answers are grounded in your document.</p>

          {uploading ? (
            <div className="upload-progress">
              <div className="spinner" />
              Processing document…
            </div>
          ) : (
            <div
              className={`dropzone${dragOver ? ' drag-over' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
            >
              <input type="file" accept=".pdf" onChange={onFileInput} />
              <span className="dz-icon">📄</span>
              <span className="dz-text"><strong>Drop a PDF here</strong> or click to browse</span>
              <span className="dz-sub">Max recommended: ~50 pages</span>
            </div>
          )}
        </div>
      )}

      {/* Chat screen */}
      {doc && (
        <div className="chat-screen">
          {/* Doc bar */}
          <div className="doc-bar">
            <span className="doc-icon">📄</span>
            <span className="doc-name">{doc.filename}</span>
            <span className="doc-meta">{doc.page_count} pages · {doc.chunks} chunks</span>
            <button className="doc-reset" onClick={reset}>× New document</button>
          </div>

          {/* Messages */}
          <div className="messages">
            {messages.length === 0 && (
              <div className="empty-state">
                <span className="hint-icon">💬</span>
                <span>Ask anything about <strong style={{ color: 'var(--text)' }}>{doc.filename}</strong></span>
                <div className="suggestions">
                  {SUGGESTIONS.map(s => (
                    <button key={s} className="suggestion-btn" onClick={() => sendQuestion(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <span className="msg-role">{m.role === 'user' ? 'You' : 'AI'}</span>
                <div className="msg-bubble"><ReactMarkdown>{m.content}</ReactMarkdown></div>
                {m.role === 'assistant' && m.sources && m.sources.length > 0 && (
                  <div className="msg-sources">
                    <button className="sources-toggle" onClick={() => toggleSources(i)}>
                      {m.showSources ? 'Hide sources' : `Show ${m.sources.length} source chunks`}
                    </button>
                    {m.showSources && (
                      <div className="sources-list">
                        {m.sources.map((s, j) => (
                          <div key={j} className="source-chip">
                            <strong style={{ color: 'var(--accent)', marginRight: 6 }}>[{j + 1}]</strong>
                            {s.slice(0, 280)}{s.length > 280 ? '…' : ''}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {thinking && (
              <div className="msg assistant">
                <span className="msg-role">AI</span>
                <div className="thinking">
                  <div className="spinner" /> Thinking…
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="input-bar">
            <textarea
              ref={inputRef}
              className="question-input"
              placeholder="Ask a question… (Enter to send)"
              value={question}
              onChange={e => setQuestion(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={thinking}
            />
            <button
              className="send-btn"
              onClick={() => sendQuestion(question)}
              disabled={!question.trim() || thinking}
            >
              Ask →
            </button>
          </div>
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <span>Built by <a href="https://rafatrik.com">Rafael Pérez</a></span>
        <span style={{ color: 'var(--border)' }}>·</span>
        <span style={{ color: 'var(--muted)' }}>Groq · sentence-transformers · FastAPI · React</span>
        <div className="footer-right">
          <a href="https://github.com/rafatrik" target="_blank" rel="noopener">GitHub</a>
          <a href="https://linkedin.com/in/rafatrik" target="_blank" rel="noopener">LinkedIn</a>
          <a href="mailto:rafatrik@gmail.com">Contact</a>
        </div>
      </footer>
    </div>
  )
}
