"""Vector-store retrieval + local Mistral-7B answer generator (Phase 5)."""

import argparse
import logging

from langchain_huggingface import HuggingFacePipeline
from llama_index.core import PromptTemplate, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from transformers import pipeline

logger = logging.getLogger(__name__)

# ----------------- Qdrant ----------------------------------------------
qdrant = QdrantClient(
    path="qdrant_db",
)  # local on-disk store :contentReference[oaicite:2]{index=2}
vstore = QdrantVectorStore(
    client=qdrant,
    collection_name="policies",
)  # must match ingest.py :contentReference[oaicite:3]{index=3}

# ---- Embedding (unchanged) ---------------------------------------------
embed_model = HuggingFaceEmbedding("intfloat/e5-large", device="cuda")
index = VectorStoreIndex.from_vector_store(vstore, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=4)
retriever.embedding_model = embed_model

# ----------------- Prompt template -------------------------------------
SYSTEM = (
    "You are the company policy assistant. "
    "Answer clearly and cite the source title in square brackets."
)


TEMPLATE = PromptTemplate(
    (
        "You are the company-policy assistant for internal employees.\n\n"
        "Context you may rely on:\n"
        "{context_str}\n\n"
        "Guidelines **(follow every time)**:\n"
        "• Use *only* the context above. If the context lacks the answer, "
        'reply exactly: "I don\'t know".\n'
        "• If asked for professional legal, medical, financial, or therapeutic "
        "advice, reply with:\n"
        "  \"I'm sorry, but I can't provide that.\" (nothing more).\n"
        "• Keep the answer to one clear paragraph.\n"
        "• After each claim, cite the source file name in square brackets "
        "(e.g. [boeing_conduct_guidelines]).\n\n"
        "Question: {query_str}\n\n"
        "Answer:"
    ),
)


# ----------------- Local Mistral-7B (no more from_pretrained) ----------
gen_pipe = pipeline(
    "text-generation",
    model="./models/mistral/gptq",
    tokenizer="./models/mistral/gptq",
    device_map="auto",  # <= plays nicely with Accelerate
    max_new_tokens=512,
    trust_remote_code=True,
    return_full_text=False,
    do_sample=True,
    temperature=0.7,
)

llm = HuggingFacePipeline(pipeline=gen_pipe)

# expose both for reuse ↓↓↓
__all__ = ["llm", "query_engine"]

# ---- Query-engine *without* extra retriever --------------------------------
query_engine = index.as_query_engine(
    llm=llm,
    text_qa_template=TEMPLATE,
    system_prompt=SYSTEM,
    response_mode="compact",  # optional: strips raw context from output
    similarity_top_k=4,  # OK because we're not passing retriever
)


# ----------------- CLI entry-point ------------------------------------
if __name__ == "__main__":
    import argparse
    import logging

    parser = argparse.ArgumentParser(description="Query the policy RAG engine")
    parser.add_argument("--question", required=True, help="Ask one question")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(query_engine.query(args.question))
