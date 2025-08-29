# Policy RAG Chatbot — Phases 2–8

## Corporate-Policy Sample Corpus (Phase 2)

Include 15 publicly available **HR, IT-security, supplier and ethics policies** drawn from technology, finance, retail, healthcare, energy, consulting and food-&-beverage firms.
Because every organisation—regardless of sector—must publish and enforce policies such as *Codes of Conduct* 🔗 \[1], *Remote-Work rules* 🔗 \[2] and *Supplier Codes of Conduct* 🔗 \[3], a RAG chatbot trained on this diverse but common genre remains **sector-agnostic and reusable across clients**.
The raw PDFs live in `/data/raw/`, are text-selectable, free for internal redistribution, and named `company_topic_YYYY.pdf` for traceability.

---

## Phase 3 — Document → Chunk → Embedding Pipeline

Run **`make ingest`** whenever you add or update PDFs in `/data/raw/`.
The target calls `python ingest.py --pdf_dir data/raw`, which

1. crawls every PDF,
2. splits text into ≈500-token chunks with 50-token overlap,
3. generates dense *intfloat/e5-large* embeddings, and
4. writes vectors + metadata into an embedded Qdrant collection.

The persisted vector store lives in `.storage/` (or `.qdrant/`)—both are git-ignored and can be rebuilt deterministically at any time.

---

## Phase 4 — LLM Setup (Mistral-7B-Instruct, 4-bit GPTQ)

> *Goal:* get a local, quantised Mistral-7B that loads in seconds and is ready for the RAG chain.

### 0) Prerequisites

* **Python 3.11** (`conda activate ragbot` or `source venv/bin/activate`).
* GPU with **CUDA 12.1** driver (CPU fallback works but is slow).
* No compiler required—pre-built wheels include all CUDA kernels.
* `requirements.txt` now pins:

  ```text
  torch==2.5.1+cu121
  torchvision==0.20.1+cu121
  torchaudio==2.5.1+cu121
  optimum[gptq]>=1.24.0
  gptqmodel>=2.2.0
  huggingface_hub[cli]>=0.31.0
  ```

  Run `pip install -r requirements.txt` after you activate the env.

### 1) Install quantisation tooling (Task 10)

```bash
pip install --upgrade "optimum[gptq]" gptqmodel
```

This pulls GPU wheels with bundled **ExLlama-V2** & **TorchQuantLinear** kernels—no source build on Windows/Linux/macOS.

### 2) Download a ready-made 4-bit checkpoint (Task 11)

```bash
huggingface-cli login
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --local-dir models/mistral/gptq \
  --local-dir-use-symlinks False
```

* Size ≈ 4.2 GB, group-size 128, Act-Order True.
* Folder **`models/` is git-ignored**—weights stay out of the repo.

### 3) Smoke-test & latency benchmark (Task 12)

```bash
python smoke_mistral.py models/mistral/gptq -- --bench
```

* Cold load ≈ 3 s on an RTX 30-series; hot reload ≈ 2 s.
* Throughput ≈ 20-30 tok / s with ExLlama-V2 kernels.
* If you see only 3-5 tok / s you’re on CPU—re-install the *cu121* wheels.

> **Notes (unchanged content):** Phase 4 is now complete; proceed to **Phase 5 — RAG chain wiring**.

---

## Phase 5 — RAG Chain Wiring & Source-Grounded QA

> *Goal:* stitch the vector store, retriever, local Mistral-7B-GPTQ, and a robust prompt into an end-to-end chatbot that answers with **one concise paragraph and inline citations**.

### 0) File layout

```
rag_chain.py          # main CLI entry-point
models/mistral/gptq/  # 4-bit weights (Phase 4)
qdrant_db/            # persisted vectors (Phase 3)
```

### 1) Run a smoke test

```bash
py -3.11 rag_chain.py --question "What is the dress code?"
```

* First call streams the model to GPU (≈ 3 s); subsequent calls complete in < 1 s.
* Output example:

```
Smart-casual attire is required for on-site work. [Dress_Code_Guide_2025]
```

### 2) Implementation notes

* `VectorStoreIndex.from_vector_store()` re-attaches the persisted Qdrant collection.
* Embeddings reuse *intfloat/e5-large* to guarantee retriever/ingest parity.
* Retrieval: `index.as_retriever(similarity_top_k=2)`—tune *top\_k* for recall ↔ noise.
* Prompt enforces grounded answers and an “I don’t know” refusal when context is missing.
* `index.as_query_engine(response_mode="compact")` builds a full `RetrieverQueryEngine` in one line.
* Retrieval: `index.as_query_engine(..., similarity_top_k=4)` — tune top_k for recall vs. noise.
* The LLM is loaded via `transformers.pipeline` wrapped by `HuggingFacePipeline` (with `trust_remote_code=True` for ExLlama‑V2 kernels).
* Quantised weights follow the GPTQ spec shipped in **optimum\[gptq]**.
* API returns JSON with an `answer` string and a `sources` list (filenames) for the UI; guardrails may also add a `[Sources]` section in the text.

### 3) Troubleshooting cheatsheet

| Symptom                             | Likely cause                      | Fix                                                                  |
| ----------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| Answer is “I don’t know”            | PDF missing or chunk too large    | Add PDF and re-run `make ingest`, or lower `CHUNK_SIZE`.             |
| Output includes entire FAQ sections | Retriever pulled multi-Q chunks   | Lower `similarity_top_k`, shrink `CHUNK_SIZE`, or refine the prompt. |
| CUDA OOM on 4-GB GPU                | Not enough vRAM for 4-bit weights | Use CPU fallback (`device_map="cpu"`) or a 2-bit GPTQ checkpoint.    |

> **Update:** to preserve VRAM for the 7B model, embeddings are executed on **CPU** during retrieval; ingest remains deterministic.

---

## Phase 6 — FastAPI Micro-service & API Tests  🚀

> **Goal:** wrap the Phase-5 `rag_chain` in an HTTP service, add a latency-logging middleware, expose interactive docs, and lock everything down with a pytest smoke-test.

### 0) Key deliverables

| File                | Purpose                                                                       |
| ------------------- | ----------------------------------------------------------------------------- |
| `main.py`           | FastAPI app exposing **POST /ask** and **GET /** (health)                     |
| `tests/` (optional) | add pytest-based smoke tests as needed; `pytest.ini` is preconfigured        |
| `pytest.ini`        | isolates collection to `tests/`, ignores 3rd-party suites, runs in quiet mode |
| `requirements.txt`  | pins `fastapi 0.111.*`, `uvicorn[standard] 0.30.*`, `python-multipart`        |

### 1) Run the service locally

```bash
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

* **`GET /`** → `{"status":"ok","docs":"/docs","ask":"/ask (POST)"}`
* **`GET /docs`** → Swagger UI (auto-generated)
* **`POST /ask`** with `{"question":"What is the dress code?"}` returns `{"answer": "...", "sources": [...]}` and an `X-Process-Time` header.

### 2) Implementation notes

1. **CORS** — `allow_origins=["*"]` for dev; tighten before prod.
2. **Latency middleware** — adds `X-Process-Time` **and** logs the path + ms.
3. **Defensive error wrapper** — all uncaught exceptions become JSON 500s.
4. **`rag_chain` import hygiene** — CLI guarded under `if __name__ == "__main__":`.
5. **Model & DB startup** — hot singletons; first request incurs load, subsequent are fast.

> **Update:** responses now **normalize inline citations** (strip absolute paths, format pages as `(p. N)` / `(pp. X–Y)`) and **de-duplicate sources** to filename-only.

---

## Phase 7 — Safety, Guardrails & Evaluation ✅

> **Goal:** add multi-layer safety (Guardrails-AI SafeText, NeMo Guardrails flows), enforce citation discipline, instrument the stack with TruLens-Eval, and pass strict linters.

### 0) Highlights

| Area                   | What changed                                                                           |
| ---------------------- | -------------------------------------------------------------------------------------- |
| **Prompt hardening**   | New `TEMPLATE` in `rag_chain.py` adds explicit refusal & citation rules.               |
| **SafeText validator** | `main.py` runs Guardrails-AI SafeText *before* NeMo guardrails to scrub unsafe output. |
| **NeMo flows**         | `rails/flows/force_citations.cl` & `refuse_medical.cl` enforce citations/refusals.     |
| **LLM guard**          | Legal/medical/financial requests → fixed refusal message.                              |
| **Instrumentation**    | `TruLlama` logs every query/response; SQLite by default.                               |
| **Lint/CI**            | `ruff`/`black` clean.                                                                  |

### 1) New / Modified files

| File / Dir                       | Purpose                                                          |
| -------------------------------- | ---------------------------------------------------------------- |
| `main.py`                        | SafeText → NeMo → response; richer logging; citation cleanup.    |
| `rag_chain.py`                   | Hardened prompt; retrieval config; CPU embeddings for stability. |
| `rails/config.yml`               | Custom engine; disables stock extensions.                        |
| `rails/flows/force_citations.cl` | Ensures `[Sources]` + ≥1 citation.                               |
| `rails/flows/refuse_medical.cl`  | Refuses medical/legal advice.                                    |
| `.pre-commit-config.yaml`        | `ruff`, `black`, `ruff-format`.                                  |

---

## Phase 8 — Front-End (React + Vite + Tailwind) 🎨

> **Goal:** deliver a simple, fast web UI that calls **POST `/ask`**, shows a concise answer, and lists sources.

### 0) Stack

* **Vite + React + TypeScript** (`frontend/`)
* **Tailwind** via `@tailwindcss/vite`
* Dev proxy: **`/api → http://localhost:8000`** to avoid CORS

### 1) Scaffold & install

```bash
# from repo root
npm create vite@latest frontend -- --template react-ts
cd frontend
npm i
npm i -D tailwindcss @tailwindcss/vite
```

### 2) Vite config (proxy + plugins)

`frontend/vite.config.ts`

```ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: p => p.replace(/^\/api/, ''),
      },
    },
  },
})
```

### 3) Tailwind wiring

`frontend/src/index.css`

```css
@import "tailwindcss";
```

### 4) Chat UI

* `frontend/src/components/Chat.tsx` — textarea + “Ask/Clear”, answer card, collapsible sources.
* `frontend/src/App.tsx` — renders `<Chat />`.
* Fetches `POST /api/ask` (or `VITE_API_URL`).

### 5) Run it

```bash
# backend
python -m uvicorn main:app --reload --port 8000

# frontend
cd frontend
npm run dev
# open http://localhost:5173
```

### 6) Screenshot

* Added at **`docs/assets/ui.jpg`**.

---

## Screenshots

* **UI (Phase 8):** `docs/assets/ui.jpg`

---

## Changelog

* **2025-08-15** — Phase 8 front-end (React/Vite/Tailwind) added; Vite dev proxy; error surfacing in UI; backend now normalizes inline citations and de-duplicates sources; screenshot saved to `docs/assets/ui.jpg`.
* **2025-06-12** — Phase 7 section (safety layers, prompt, guardrails, TruLens).
* **2025-05-14** — Phase 6 (FastAPI service) documentation.
* Older changes: see commit history.
