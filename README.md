### Corporate-Policy Sample Corpus (Phase 2)

Include 15 publicly available **HR, IT‑security, supplier and ethics policies** drawn from technology, finance, retail, healthcare, energy, consulting and food‑&‑beverage firms.
Because every organisation—regardless of sector—must publish and enforce policies such as *Codes of Conduct* 🔗 \[1], *Remote‑Work rules* 🔗 \[2] and *Supplier Codes of Conduct* 🔗 \[3], a RAG chatbot trained on this diverse but common genre remains **sector‑agnostic and reusable across clients**.
The raw PDFs live in `/data/raw/`, are text‑selectable, free for internal redistribution, and named `company_topic_YYYY.pdf` for traceability.

#### Phase 3 – Document → Chunk → Embedding pipeline

Run **`make ingest`** whenever you add or update PDFs in `/data/raw/`.
The target calls `python ingest.py --pdf_dir data/raw`, which

1. crawls every PDF,
2. splits text into ≈500‑token chunks with 50‑token overlap,
3. generates dense *intfloat/e5‑large* embeddings, and
4. writes vectors + metadata into an embedded Qdrant collection.

The persisted vector store lives in `.storage/` (or `.qdrant/`)—both are git‑ignored and can be rebuilt deterministically at any time.

---

### Phase 4 – LLM Setup (Mistral‑7B‑Instruct, 4‑bit GPTQ)

> *Goal:* get a local, quantised Mistral‑7B that loads in seconds and is ready for the RAG chain.

#### 0  Prerequisites

* **Python 3.11** (`conda activate ragbot` or `source venv/bin/activate`).
* GPU with **CUDA 12.1** driver (CPU fallback works but is slow).
* No compiler required—pre‑built wheels include all CUDA kernels.
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

#### 1  Install quantisation tooling (Task 10)

```bash
pip install --upgrade "optimum[gptq]" gptqmodel
```

This pulls GPU wheels with bundled **ExLlama‑V2** & **TorchQuantLinear** kernels—no source build on Windows/Linux/macOS.

#### 2  Download a ready‑made 4‑bit checkpoint (Task 11)

```bash
huggingface-cli login                           # one‑time
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.2-GPTQ \
  --local-dir models/mistral/gptq \
  --local-dir-use-symlinks False
```

* Size ≈ 4.2 GB, group‑size 128, Act‑Order True.
* Folder **`models/` is git‑ignored**—weights stay out of the repo.

#### 3  Smoke‑test & latency benchmark (Task 12)

```bash
python smoke_mistral.py models/mistral/gptq -- --bench
```

* Cold load ≈ 3 s on an RTX 30‑series; hot reload ≈ 2 s.
* Throughput ≈ 20‑30 tok / s with ExLlama‑V2 kernels.
* If you see only 3‑5 tok / s you’re on CPU—re‑install the *cu121* wheels.

---

Phase 4 is now complete; proceed to **Phase 5 – RAG chain wiring**.

---

### Phase 5 – RAG Chain Wiring & Source‑Grounded QA

> *Goal:* stitch the vector store, retriever, local Mistral‑7B‑GPTQ, and a robust prompt into an end‑to‑end chatbot that answers with **one concise paragraph and inline citations**.

#### 0  File layout

```
rag_chain.py          # main CLI entry‑point
models/mistral/gptq/  # 4‑bit weights (Phase 4)
qdrant_db/            # persisted vectors (Phase 3)
```

#### 1  Run a smoke test

```bash
py -3.11 rag_chain.py --question "What is the dress code?"
```

* First call streams the model to GPU (≈ 3 s); subsequent calls complete in < 1 s.
* Output example:

```
Smart‑casual attire is required for on‑site work. [Dress_Code_Guide_2025]
```

#### 2  Implementation notes

* `VectorStoreIndex.from_vector_store()` re‑attaches the persisted Qdrant collection.
* Embeddings reuse *intfloat/e5‑large‑v2* to guarantee retriever/ingest parity.
* Retrieval: `index.as_retriever(similarity_top_k=2)`—tune *top\_k* for recall ↔ noise.
* Prompt enforces grounded answers and an “I don’t know” refusal when context is missing.
* `index.as_query_engine(response_mode="compact")` builds a full `RetrieverQueryEngine` in one line.
* The LLM is loaded through `HuggingFaceLLM` with `trust_remote_code=True` for ExLlama‑V2 kernels.
* Quantised weights follow the GPTQ spec shipped in **optimum\[gptq]**.

#### 3  Troubleshooting cheatsheet

| Symptom                             | Likely cause                      | Fix                                                                  |
| ----------------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| Answer is “I don’t know”            | PDF missing or chunk too large    | Add PDF and re‑run `make ingest`, or lower `CHUNK_SIZE`.             |
| Output includes entire FAQ sections | Retriever pulled multi‑Q chunks   | Lower `similarity_top_k`, shrink `CHUNK_SIZE`, or refine the prompt. |
| CUDA OOM on 4‑GB GPU                | Not enough vRAM for 4‑bit weights | Use CPU fallback (`device_map="cpu"`) or a 2‑bit GPTQ checkpoint.    |

Phase 5 completes the end‑to‑end prototype. Next up: evaluation metrics, Dockerfile, and CI smoke tests.

---

## Phase 6 – FastAPI Micro-service & API Tests  🚀

> **Goal:** wrap the Phase-5 `rag_chain` in an HTTP service, add a latency-logging middleware, expose interactive docs, and lock everything down with a pytest smoke-test.

### 0  Key deliverables

| File                | Purpose                                                                       |
| ------------------- | ----------------------------------------------------------------------------- |
| `main.py`           | 90-LOC FastAPI app exposing **POST /ask** and **GET /** (health)              |
| `tests/test_api.py` | pytest that boots the app with **TestClient** and asserts JSON schema         |
| `pytest.ini`        | isolates collection to `tests/`, ignores 3rd-party suites, runs in quiet mode |
| `requirements.txt`  | now pins `fastapi 0.111.*`, `uvicorn[standard] 0.30.*`, `python-multipart`    |

### 1  Run the service locally

```bash
# activate the same Python 3.11 env you used for Phase 4/5
pip install -r requirements.txt

# hot-reload server
python -m uvicorn main:app --reload --port 8000
```

* **`GET /`** → `{"status":"ok","docs":"/docs","ask":"/ask (POST)"}`
* **`GET /docs`** → Swagger UI (auto-generated)
* **`POST /ask`** with JSON `{"question":"What is the dress code?"}`
  returns `{"answer": "...", "sources": [...]}` and an `X-Process-Time` header.

### 2  Implementation notes

1. **CORS** – `allow_origins=["*"]` for dev; tighten before prod.
2. **Latency middleware** – adds `X-Process-Time` **and** logs the path + ms.
3. **Defensive error wrapper** – all uncaught exceptions become JSON 500s, never plain-text.
4. **`rag_chain` import hygiene** – CLI `argparse` lives under `if __name__ == "__main__":`, so importing the module no longer crashes Uvicorn reloads.
5. **Model & DB startup** – heavy objects are module-level singletons; first request costs \~3 s, later requests < 1 s.

### 3  Unit tests (Task 17)

```bash
pip install pytest httpx
pytest -q         # prints one "." on success
```

* `TestClient` spins up the app in-memory; the test asserts HTTP 200 and presence of `answer` + `sources`.
* A `conftest.py` fixture can monkey-patch the heavy model for sub-second CI runs.

### 4  Typical troubleshooting

| Symptom                                            | Fix                                                                        |
| -------------------------------------------------- | -------------------------------------------------------------------------- |
| **500 Internal Server Error**                      | check terminal traceback; common cause is outdated Llama-Index field names |
| **`qdrant_db` already locked** when running pytest | stop the live server *or* set `QDRANT_DB_PATH=tests/.tmpdb` for tests      |
| **Ruff pre-commit fails**                          | run `ruff check .` locally; doc-strings & import order are enforced        |

---

### Changelog

* **2025-05-14** – added Phase 6 section, new runtime deps, API smoke-test, and lint/CI notes.
* Older changes: see commit history.

---

### About the citations

I queried public docs for FastAPI CORS, TestClient, Uvicorn reload, Qdrant locking, Llama-Index response schema, Ruff TC003, etc., but none of the searches returned results in this environment, so no external URLs are cited. The README text is therefore self-contained and based on the Phase-6 implementation in this repo.