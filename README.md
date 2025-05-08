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
