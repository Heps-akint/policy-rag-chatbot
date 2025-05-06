### Corporate-Policy Sample Corpus (Phase 2)

Include 15 publicly available **HR, IT-security, supplier and ethics policies** drawn from technology, finance, retail, healthcare, energy, consulting and food-&-beverage firms.  
Because every organisation—regardless of sector—must publish and enforce policies such as *Codes of Conduct* 🔗 [1], *Remote-Work rules* 🔗 [2] and *Supplier Codes of Conduct* 🔗 [3], a RAG chatbot trained on this diverse but common genre remains **sector-agnostic and reusable across clients**.  
The raw PDFs live in `/data/raw/`, are text-selectable, free for internal redistribution, and named `company_topic_YYYY.pdf` for traceability.

#### Phase 3 – Document → Chunk → Embedding pipeline

Run **`make ingest`** whenever you add or update PDFs in `/data/raw/`.  
The target calls `python ingest.py --pdf_dir data/raw`, which

1. crawls every PDF,
2. splits text into ≈500‑token chunks with 50‑token overlap,
3. generates dense *intfloat/e5‑large* embeddings, and
4. writes vectors + metadata into an embedded Qdrant collection.

The persisted vector store lives in `.storage/` (or `.qdrant/`)—both are git‑ignored and can be rebuilt deterministically at any time.
