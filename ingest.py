#!/usr/bin/env python
"""
Chunk → embed → persist pipeline used by `make ingest`.

Document -> Chunk -> Embed pipeline.

Reads all PDFs in a directory, splits them into sentence-preserving chunks,
embeds each chunk with a local **E5-large** encoder, and persists the vectors
plus metadata into an in-process Qdrant collection.
"""

from __future__ import annotations  # Ruff F404: must come first

import argparse
import logging
from pathlib import Path

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CHUNK_SIZE: int = 500  # tokens per chunk
OVERLAP: int = 50  # overlap between consecutive chunks
COLLECTION: str = "policies"  # Qdrant collection name

# Embedding wrapper (subclass of BaseEmbedding -> satisfies Ruff)
EMBED_MODEL = HuggingFaceEmbedding(model_name="intfloat/e5-large")

# Structured logging
LOGGER = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def build_index(pdf_dir: Path) -> int:
    """
    Embed PDFs in *pdf_dir* and store them in Qdrant.

    Args:
        pdf_dir: Directory containing PDF files.

    Returns:
        Number of chunks written to the vector store.

    """
    # 1. Load documents
    docs = SimpleDirectoryReader(str(pdf_dir)).load_data()

    # 2. Split into sentence-level chunks
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)

    # 4. Vector store - in-process Qdrant (swap to host/port for server mode)
    qdrant = QdrantClient(path="qdrant_db")
    vstore = QdrantVectorStore(client=qdrant, collection_name=COLLECTION)
    storage = StorageContext.from_defaults(vector_store=vstore)

    # 5. Build the index (LlamaIndex embeds via EMBED_MODEL)
    VectorStoreIndex(
        nodes,
        storage_context=storage,
        embed_model=EMBED_MODEL,
    )

    # 6. Persist so reruns are incremental
    storage.persist()

    return len(nodes)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed PDFs and load them into a local Qdrant store",
    )
    parser.add_argument(
        "--pdf_dir",
        default="data/raw",
        type=str,
        help="Directory with PDF files to ingest",
    )
    args = parser.parse_args()

    chunk_count = build_index(Path(args.pdf_dir))
    LOGGER.info("Indexed %s chunks into '%s'", chunk_count, COLLECTION)
