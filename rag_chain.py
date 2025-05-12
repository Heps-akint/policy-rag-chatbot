"""Vector-store retrieval + local Mistral-7B answer generator (Phase 5)."""

import argparse

from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM  # ← same import
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# ----------------- CLI -------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--question", required=True)
args = parser.parse_args()

# ----------------- Qdrant ----------------------------------------------
qdrant = QdrantClient(
    path="qdrant_db",
)  # local on-disk store :contentReference[oaicite:2]{index=2}
vstore = QdrantVectorStore(
    client=qdrant,
    collection_name="policies",
)  # must match ingest.py :contentReference[oaicite:3]{index=3}

# ---- Embedding (unchanged) ---------------------------------------------
embed_model = HuggingFaceEmbedding("intfloat/e5-large-v2", device="cuda")
index = VectorStoreIndex.from_vector_store(vstore, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=2)
retriever.embedding_model = embed_model

# ----------------- Prompt template -------------------------------------
SYSTEM = (
    "You are the company policy assistant. "
    "Answer clearly and cite the source title in square brackets."
)
TEMPLATE = PromptTemplate(
    "You are a policy assistant. Use ONLY the information in {context_str} "
    "to answer the single question below. If the answer is not there, say "
    '"I don\'t know".\n\nQuestion: {query_str}\n\nAnswer (cite source):',
)


# ----------------- Local Mistral-7B (no more from_pretrained) ----------
llm = HuggingFaceLLM(
    model_name="./models/mistral/gptq",  # local weights
    tokenizer_name="./models/mistral/gptq",
    model_kwargs={"trust_remote_code": True},
    device_map="auto",
    max_new_tokens=512,
    context_window=4096,
)

# ---- Query-engine *without* extra retriever --------------------------------
query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=TEMPLATE,
    system_prompt=SYSTEM,
    response_mode="compact",  # optional: strips raw context from output
    similarity_top_k=2,  # OK because we're not passing retriever
)


if __name__ == "__main__":
    import logging

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(query_engine.query(args.question))
