import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'

// In production (Vercel), set VITE_API_URL to the Railway/Render backend URL.
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

type Tool = 'chat' | 'summary' | 'compare' | 'contradictions'

// ── Component ─────────────────────────────────────────────────────────────────

export default function App() {
  // Core state
  const [doc, setDoc]           = useState<DocInfo | null>(null)
  const [tool, setTool]         = useState<Tool>('chat')
  const [error, setError]       = useState<string | null>(null)
  const [uploading, setUploading]   = useState(false)
  const [dragOver, setDragOver]     = useState(false)

  // Chat state
  const [messages, setMessages]     = useState<Message[]>([])
  const [question, setQuestion]     = useState('')
  const [thinking, setThinking]     = useState(false)
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [listening, setListening]   = useState(false)

  // Tool states
  const [summary, setSummary]               = useState('')
  const [summaryLoading, setSummaryLoading] = useState(false)
  const [docB, setDocB]                     = useState<DocInfo | null>(null)
  const [uploadingB, setUploadingB]         = useState(false)
  const [compareResult, setCompareResult]   = useState('')
  const [compareLoading, setCompareLoading] = useState(false)
  const [contradictions, setContradictions]         = useState('')
  const [contradictionsLoading, setContradictionsLoading] = useState(false)

  const [aiWarning, setAiWarning] = useState<string | null>(null)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef       = useRef<HTMLTextAreaElement>(null)

  // Poll Gemini availability every 2 minutes; check once on mount
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API}/health/ai`)
        if (res.ok) {
          const data = await res.json()
          setAiWarning(data.available ? null : data.message)
        }
      } catch { /* ignore network errors */ }
    }
    check()
    const id = setInterval(check, 2 * 60 * 1000)
    return () => clearInterval(id)
  }, [])

  const scrollToBottom = () =>
    setTimeout(() => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)

  // ── Streaming helper ──────────────────────────────────────────────────────────
  // Reads a text/event-stream response and appends chunks to a state setter.

  const streamTo = async (
    url: string,
    body: object,
    setter: React.Dispatch<React.SetStateAction<string>>
  ) => {
    setter('')
    const res = await fetch(`${API}${url}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed.' }))
      throw new Error(err.detail ?? 'Request failed.')
    }
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const event = JSON.parse(line.slice(6))
        if (event.type === 'chunk') setter(prev => prev + event.text)
        if (event.type === 'error') throw new Error(event.message)
      }
    }
  }

  // ── Upload ────────────────────────────────────────────────────────────────────

  const handleFile = async (file: File, isB = false) => {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files are supported.')
      return
    }
    setError(null)
    isB ? setUploadingB(true) : setUploading(true)

    const form = new FormData()
    form.append('file', file)

    try {
      const res = await fetch(`${API}/upload`, { method: 'POST', body: form })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed.' }))
        throw new Error(err.detail ?? 'Upload failed.')
      }
      const data: DocInfo = await res.json()

      if (isB) {
        setDocB(data)
        setCompareResult('')
      } else {
        setDoc(data)
        setMessages([])
        setSuggestions([])
        setSummary('')
        setContradictions('')
        setCompareResult('')
        setDocB(null)
        setTool('chat')
        fetchSuggestions(data.session_id)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Upload failed.')
    } finally {
      isB ? setUploadingB(false) : setUploading(false)
    }
  }

  const fetchSuggestions = async (sessionId: string) => {
    try {
      const res = await fetch(`${API}/suggest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      })
      if (res.ok) {
        const data = await res.json()
        setSuggestions(data.questions)
      }
    } catch { /* silently ignore */ }
  }

  const onFileInput = (e: React.ChangeEvent<HTMLInputElement>, isB = false) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file, isB)
    e.target.value = ''
  }

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  // ── Chat ──────────────────────────────────────────────────────────────────────

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

      // Add empty assistant message that will fill progressively
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
        buffer = lines.pop() ?? ''
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const event = JSON.parse(line.slice(6))
          if (event.type === 'chunk') {
            setMessages(prev => prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, content: m.content + event.text } : m
            ))
            scrollToBottom()
          } else if (event.type === 'sources') {
            setMessages(prev => prev.map((m, i) =>
              i === prev.length - 1 ? { ...m, sources: event.sources } : m
            ))
          } else if (event.type === 'error') {
            throw new Error(event.message)
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

  // ── Voice input ───────────────────────────────────────────────────────────────

  const startVoice = () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) { setError('Voice input is not supported in this browser. Use Chrome or Edge.'); return }
    const rec = new SR()
    rec.lang = 'en-US'
    rec.interimResults = false
    rec.onstart  = () => setListening(true)
    rec.onend    = () => setListening(false)
    rec.onerror  = () => setListening(false)
    rec.onresult = (e: any) => setQuestion(e.results[0][0].transcript)
    rec.start()
  }

  // ── Tool actions ──────────────────────────────────────────────────────────────

  const runSummary = async () => {
    if (!doc || summaryLoading) return
    setSummaryLoading(true)
    setError(null)
    try { await streamTo('/summary', { session_id: doc.session_id }, setSummary) }
    catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.') }
    finally { setSummaryLoading(false) }
  }

  const runCompare = async () => {
    if (!doc || !docB || compareLoading) return
    setCompareLoading(true)
    setError(null)
    try { await streamTo('/compare', { session_id_a: doc.session_id, session_id_b: docB.session_id }, setCompareResult) }
    catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.') }
    finally { setCompareLoading(false) }
  }

  const runContradictions = async () => {
    if (!doc || contradictionsLoading) return
    setContradictionsLoading(true)
    setError(null)
    try { await streamTo('/contradictions', { session_id: doc.session_id }, setContradictions) }
    catch (e) { setError(e instanceof Error ? e.message : 'Something went wrong.') }
    finally { setContradictionsLoading(false) }
  }

  const reset = () => {
    setDoc(null); setDocB(null); setMessages([]); setError(null)
    setSuggestions([]); setSummary(''); setContradictions(''); setCompareResult('')
    setTool('chat')
  }

  // ── Tabs config ───────────────────────────────────────────────────────────────

  const TOOLS: { id: Tool; icon: string; label: string }[] = [
    { id: 'chat',           icon: '💬', label: 'Chat'            },
    { id: 'summary',        icon: '📊', label: 'Auto Summary'    },
    { id: 'compare',        icon: '🔄', label: 'Compare'         },
    { id: 'contradictions', icon: '⚠️', label: 'Contradictions'  },
  ]

  // ── Render ────────────────────────────────────────────────────────────────────

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

      {/* AI availability warning */}
      {aiWarning && <div className="ai-warning-toast">⚠️ {aiWarning}</div>}

      {/* Upload screen */}
      {!doc && (
        <div className="upload-screen">
          <div className="upload-label">AI Document Q&amp;A</div>
          <h1>Ask anything about<br />your PDF</h1>
          <p>Upload a PDF and unlock a full AI toolkit. Powered by RAG + Gemini.</p>

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
              <input type="file" accept=".pdf" onChange={e => onFileInput(e)} />
              <span className="dz-icon">📄</span>
              <span className="dz-text"><strong>Drop a PDF here</strong> or click to browse</span>
              <span className="dz-sub">Max recommended: ~50 pages</span>
            </div>
          )}

          <div className="feature-pills">
            <span className="pill">💬 Chat</span>
            <span className="pill">📊 Auto Summary</span>
            <span className="pill">🔄 Compare docs</span>
            <span className="pill">⚠️ Find contradictions</span>
            <span className="pill">🎤 Voice input</span>
          </div>
        </div>
      )}

      {/* Main screen */}
      {doc && (
        <div className="chat-screen">
          {/* Doc bar */}
          <div className="doc-bar">
            <span className="doc-icon">📄</span>
            <span className="doc-name">{doc.filename}</span>
            <span className="doc-meta">{doc.page_count} pages · {doc.chunks} chunks</span>
            <button className="doc-reset" onClick={reset}>× New document</button>
          </div>

          {/* Tool tabs */}
          <div className="tool-tabs">
            {TOOLS.map(t => (
              <button
                key={t.id}
                className={`tool-tab${tool === t.id ? ' active' : ''}`}
                onClick={() => setTool(t.id)}
              >
                {t.icon} {t.label}
              </button>
            ))}
          </div>

          {/* ── Chat ── */}
          {tool === 'chat' && (<>
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty-state">
                  <span className="hint-icon">💬</span>
                  <span>Ask anything about <strong style={{ color: 'var(--text)' }}>{doc.filename}</strong></span>
                  <div className="suggestions">
                    {suggestions.length > 0
                      ? suggestions.map(s => (
                          <button key={s} className="suggestion-btn" onClick={() => sendQuestion(s)}>{s}</button>
                        ))
                      : <div className="spinner" style={{ margin: '8px auto' }} />
                    }
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
                  <div className="thinking"><div className="spinner" /> Thinking…</div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="input-bar">
              <button
                className={`voice-btn${listening ? ' active' : ''}`}
                onClick={startVoice}
                title="Voice input (Chrome/Edge)"
              >🎤</button>
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
              >Ask →</button>
            </div>
          </>)}

          {/* ── Auto Summary ── */}
          {tool === 'summary' && (
            <div className="tool-panel">
              <div className="tool-panel-header">
                <div>
                  <div className="tool-panel-title">📊 Auto Summary</div>
                  <div className="tool-panel-sub">Gemini analyzes the document and extracts key facts, entities, and conclusions automatically.</div>
                </div>
                {!summary && (
                  <button className="run-btn" onClick={runSummary} disabled={summaryLoading}>
                    {summaryLoading ? <><div className="spinner-sm" /> Analyzing…</> : 'Analyze document'}
                  </button>
                )}
              </div>
              {summary
                ? <div className="tool-result"><ReactMarkdown>{summary}</ReactMarkdown></div>
                : !summaryLoading && <div className="tool-empty">Click "Analyze document" to generate a structured summary.</div>
              }
            </div>
          )}

          {/* ── Compare ── */}
          {tool === 'compare' && (
            <div className="tool-panel">
              <div className="tool-panel-header">
                <div>
                  <div className="tool-panel-title">🔄 Document Comparison</div>
                  <div className="tool-panel-sub">Upload a second PDF and compare both documents side by side.</div>
                </div>
                {docB && !compareResult && (
                  <button className="run-btn" onClick={runCompare} disabled={compareLoading}>
                    {compareLoading ? <><div className="spinner-sm" /> Comparing…</> : 'Compare documents'}
                  </button>
                )}
              </div>

              <div className="compare-docs">
                <div className="compare-doc">
                  <div className="compare-doc-label">Document A</div>
                  <div className="compare-doc-name">📄 {doc.filename}</div>
                </div>
                <div className="compare-vs">vs</div>
                <div className="compare-doc">
                  <div className="compare-doc-label">Document B</div>
                  {docB ? (
                    <div className="compare-doc-name">📄 {docB.filename}</div>
                  ) : uploadingB ? (
                    <div className="compare-doc-name"><div className="spinner-sm" /> Uploading…</div>
                  ) : (
                    <label className="compare-upload-btn">
                      <input type="file" accept=".pdf" onChange={e => onFileInput(e, true)} style={{ display: 'none' }} />
                      + Upload second PDF
                    </label>
                  )}
                </div>
              </div>

              {compareResult && <div className="tool-result"><ReactMarkdown>{compareResult}</ReactMarkdown></div>}
            </div>
          )}

          {/* ── Contradictions ── */}
          {tool === 'contradictions' && (
            <div className="tool-panel">
              <div className="tool-panel-header">
                <div>
                  <div className="tool-panel-title">⚠️ Contradiction Finder</div>
                  <div className="tool-panel-sub">Detect conflicting statements, inconsistencies, and ambiguities in the document.</div>
                </div>
                {!contradictions && (
                  <button className="run-btn" onClick={runContradictions} disabled={contradictionsLoading}>
                    {contradictionsLoading ? <><div className="spinner-sm" /> Scanning…</> : 'Scan document'}
                  </button>
                )}
              </div>
              {contradictions
                ? <div className="tool-result"><ReactMarkdown>{contradictions}</ReactMarkdown></div>
                : !contradictionsLoading && <div className="tool-empty">Click "Scan document" to find contradictions and inconsistencies.</div>
              }
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="footer">
        <span>Built by <a href="https://rafatrik.com">Rafael Pérez</a></span>
        <span style={{ color: 'var(--border)' }}>·</span>
        <span style={{ color: 'var(--muted)' }}>Gemini · sentence-transformers · FastAPI · React</span>
        <div className="footer-right">
          <a href="https://github.com/rafatrik" target="_blank" rel="noopener">GitHub</a>
          <a href="https://linkedin.com/in/rafatrik" target="_blank" rel="noopener">LinkedIn</a>
          <a href="mailto:rafatrik@gmail.com">Contact</a>
        </div>
      </footer>
    </div>
  )
}
