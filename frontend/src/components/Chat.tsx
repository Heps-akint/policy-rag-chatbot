import React, { useState, useRef } from 'react'

type ApiResponse = {
  answer: string
  sources: string[]
}

export default function Chat() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<string | null>(null)
  const [sources, setSources] = useState<string[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const API_BASE = import.meta.env.VITE_API_URL ?? '/api' // dev proxy fallback

  async function ask(e?: React.FormEvent) {
    e?.preventDefault()
    const q = question.trim()
    if (!q) return
    setLoading(true); setError(null); setAnswer(null); setSources([])
    try {
      const res = await fetch(`${API_BASE}/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q }),
      })
      if (!res.ok) {
        const text = await res.text()
        let msg = `HTTP ${res.status}`
        try { const j = JSON.parse(text); msg = j.error || j.detail || msg } catch {}
        throw new Error(msg)
      }
      const data = (await res.json()) as ApiResponse
      setAnswer(data.answer)
      setSources(Array.isArray(data.sources) ? data.sources : [])
    } catch (err: any) {
      setError(err.message || 'Request failed')
    } finally {
      setLoading(false)
    }
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      ask()
    }
  }

  // ✨ Client-side belt-and-braces: basename + de-duplicate
  const uniqueSources = Array.from(
    new Set(
      (sources || []).map(s => {
        const parts = String(s).split(/[\\/]/)
        return parts[parts.length - 1] || String(s)
      })
    )
  )

  return (
    <div className="mx-auto max-w-3xl p-6">
      <h1 className="text-2xl font-semibold tracking-tight">Policy RAG Chat</h1>
      <p className="text-sm text-gray-500 mt-1">
        Ask a question about the ingested policies. Answers are short and cite sources.
      </p>

      <form onSubmit={ask} className="mt-4 space-y-3">
        <label htmlFor="q" className="sr-only">Question</label>
        <textarea
          id="q"
          ref={textareaRef}
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="e.g., What is the dress code?"
          rows={3}
          className="w-full rounded-2xl border border-gray-300 p-3 outline-none focus:ring-2 focus:ring-black/10"
        />
        <div className="flex items-center gap-3">
          <button
            type="submit"
            disabled={loading}
            className="rounded-2xl px-4 py-2 font-medium shadow-sm border border-gray-300 disabled:opacity-50"
          >
            {loading ? 'Thinking…' : 'Ask'}
          </button>
          <button
            type="button"
            onClick={() => { setQuestion(''); textareaRef.current?.focus() }}
            className="text-sm underline underline-offset-2"
          >
            Clear
          </button>
        </div>
      </form>

      {error && (
        <div className="mt-4 rounded-2xl border border-red-300 bg-red-50 p-3 text-red-700">
          {error}
        </div>
      )}

      {answer && (
        <div className="mt-6 space-y-3">
          <div className="rounded-2xl border p-4">
            <h2 className="text-sm font-semibold text-gray-600 mb-2">Answer</h2>
            <p className="leading-7">{answer}</p>
          </div>

          <details className="rounded-2xl border p-4">
            <summary className="cursor-pointer text-sm font-semibold text-gray-600">
              Sources ({uniqueSources.length})
            </summary>
            <ul className="mt-3 list-disc pl-6 space-y-1">
              {uniqueSources.map((s, i) => (
                <li key={i} className="text-sm text-gray-700 break-words">{s}</li>
              ))}
            </ul>
          </details>
        </div>
      )}
    </div>
  )
}
