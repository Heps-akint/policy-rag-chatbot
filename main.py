"""FastAPI layer exposing the RAG engine (Phase 6)."""  # ← D100

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import nest_asyncio
from anyio import to_thread
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Guardrails-AI - `safe_text` output guard (Task 20)
from guardrails import Guard
from nemoguardrails import LLMRails, RailsConfig
from pydantic import BaseModel
from trulens.apps.llamaindex import TruLlama  # ✅ correct location
from trulens_eval import Tru  # DB/session

from rag_chain import (
    llm as policy_llm,
)
from rag_chain import (
    query_engine,  # ← reuse
)

# Apply event-loop patch *after* all imports to avoid E402
nest_asyncio.apply()

SAFE_TEXT_RAIL = """
<rail version="0.1">
  <output>
    <string name="response"/>
  </output>
  <validators>
    <SafeText/>           <!-- blocks medical / legal advice, illicit content, etc. -->
  </validators>
</rail>
"""
safe_guard = Guard.from_rail_string(SAFE_TEXT_RAIL)


# --- build a simple text-to-text wrapper once --------------------------
def rag_answer(question: str) -> str:
    """Return ONLY the text answer (TruBasicApp needs str)."""
    return query_engine.query(question).response  # no citations, just text


tru = Tru()
tru_recorder = TruLlama(  # 🚀 instruments Llama-Index for you
    query_engine,
    app_id="policy-rag",
)

rails_cfg = RailsConfig.from_path("rails")  # reads rails/config.yml
rails = LLMRails(rails_cfg, llm=policy_llm)  # no second load!


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

app = FastAPI(title="Policy-RAG API")

# CORS …
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):  # D101
    """Schema for POST /ask JSON payload."""

    question: str


@app.post("/ask")
async def ask(payload: Question) -> JSONResponse:  # ANN201, D103
    """
    Answer a natural-language question with citations.

    Guarded by Guardrails-AI (`safe_text`) and logged by TruLens-Eval
    (`with_record`).
    """
    try:
        # ── run the query once *inside* the TruLlama recorder ──
        with tru_recorder as recording:  # ✅
            resp = query_engine.query(payload.question)

        raw_answer = resp.response  # plain text
        _rec = recording  # store/use it

        result = await to_thread.run_sync(safe_guard.validate, raw_answer)

        # Pull the cleaned text, but be ready for None
        validated = (
            getattr(result, "validated_output", None)  # ≥ 0.4
            or getattr(result, "output", None)  # 0.3.x
            or (result if isinstance(result, str) else None)
        )

        # If Guardrails returned nothing, keep the original answer
        if not validated:
            validated = raw_answer

        validated = validated.strip()
        # If it passed, `validated` contains cleaned text; if it tripped, it raises.

        # Finally run NeMo Guardrails for policy / tone checks.
        guarded = await rails.generate_async(
            messages=[{"role": "assistant", "content": validated}],
        )

        # 1️⃣ list of messages  → take last one
        if isinstance(guarded, list) and guarded:
            answer = str(guarded[-1].get("content", "")).strip()

        # 2️⃣ single dict       → take its content
        elif isinstance(guarded, dict):
            answer = str(guarded.get("content", "")).strip()

        # 3️⃣ raw string / None → cast to string
        else:
            answer = str(guarded or "").strip()

        # Optional: if Guardrails returned an empty string, fall back to raw RAG answer
        if not answer:
            answer = raw_answer.strip()

        raw_sources = getattr(resp, "source_nodes", []) or []
        sources = [
            (getattr(node, "metadata", {}) or getattr(node, "node", {}).metadata).get(
                "file_name",
                "unknown",
            )
            for node in raw_sources
        ] or (
            [resp.get_formatted_sources(length=120)]
            if hasattr(resp, "get_formatted_sources")
            else []
        )

    except Exception as exc:
        logger.exception("Uncaught error in /ask")  # LOG015 fixed
        return JSONResponse(
            status_code=500,
            content={"detail": "Query engine failed", "error": str(exc)},
        )
    else:  # TRY300 fixed
        return JSONResponse(content={"answer": answer, "sources": sources})


@app.middleware("http")
async def add_process_time_header(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> JSONResponse:
    """Add X-Process-Time header and log request latency."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{duration:.4f}"
    logger.info("%s %s completed in %.4fs", request.method, request.url.path, duration)
    return response


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:  # ANN201
    """Return health-check JSON and a link to Swagger UI."""
    return {"status": "ok", "docs": "/docs", "ask": "/ask (POST)"}
