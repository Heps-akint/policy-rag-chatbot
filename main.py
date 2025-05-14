"""FastAPI layer exposing the RAG engine (Phase 6)."""  # ← D100

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from rag_chain import query_engine

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
    """Answer a single natural-language question with citations."""
    try:
        resp = query_engine.query(payload.question)
        answer = getattr(resp, "response", None) or str(resp)

        raw_sources = getattr(resp, "source_nodes", []) or []
        sources = [
            (getattr(node, "metadata", {}) or getattr(node, "node", {}).metadata).get(
                "title",
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
