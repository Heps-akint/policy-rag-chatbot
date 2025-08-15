"""FastAPI layer exposing the RAG engine (Phase 6)."""  # ← D100

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
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

from rag_chain import llm as policy_llm
from rag_chain import query_engine  # ← reuse

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


# --- helpers: clean up citations and file paths -------------------------
def _basename(path: str) -> str:
    """Return just the filename for either Windows or POSIX-style paths."""
    return Path(path).name


def normalize_citations(text: str) -> str:
    r"""
    Convert noisy path+page citations.

    Examples:
      [C:\path\file.pdf, page_label: 9] -> [file.pdf (p. 9)]
      [C:\path\file.pdf, pages: 4-5]    -> [file.pdf (pp. 4-5)]
      [C:\path\file.pdf]                -> [file.pdf]

    """
    text = re.sub(
        r"\[([^\]]*?\.pdf)[^\]]*?page_label:\s*(\d+)[^\]]*\]",
        lambda m: f"[{_basename(m.group(1))} (p. {m.group(2)})]",
        text,
    )
    # pages: 4-5 / 3, 7-8
    text = re.sub(
        r"\[([^\]]*?\.pdf)[^\]]*?pages?:\s*([0-9,\-\s]+)[^\]]*\]",
        lambda m: f"[{_basename(m.group(1))} (pp. {m.group(2).strip()})]",
        text,
    )
    # any remaining [ ...file.pdf ]
    text = re.sub(
        r"\[([^\]]*?\.pdf)\]",
        lambda m: f"[{_basename(m.group(1))}]",
        text,
    )
    return text  # noqa: RET504


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
async def ask(payload: Question) -> JSONResponse:  # noqa: C901, PLR0912  # ANN201, D103
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

        # Validate with SafeText, but never let it crash the request
        try:
            result = await to_thread.run_sync(safe_guard.validate, raw_answer)
            validated = (
                getattr(result, "validated_output", None)
                or getattr(result, "output", None)
                or (result if isinstance(result, str) else None)
                or raw_answer
            )
            validated = str(validated).strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("SafeText validation failed: %s", e)
            validated = raw_answer

        # Run NeMo Guardrails, but never let it crash the request
        try:
            guarded = await rails.generate_async(
                messages=[{"role": "assistant", "content": validated}],
            )
            if isinstance(guarded, list) and guarded:
                answer = str(guarded[-1].get("content", "")).strip()
            elif isinstance(guarded, dict):
                answer = str(guarded.get("content", "")).strip()
            else:
                answer = str(guarded or "").strip()
        except Exception as e:  # noqa: BLE001
            logger.warning("NeMo Guardrails failed: %s", e)
            answer = validated

        if not answer:
            answer = raw_answer.strip()

        # ✨ Clean inline citations (remove long paths / format pages neatly)
        answer = normalize_citations(answer)

        # ✨ Build a clean, de-duplicated sources list (filenames only)
        raw_nodes = getattr(resp, "source_nodes", []) or []
        sources_raw: list[str] = []
        for node in raw_nodes:
            md = (
                getattr(node, "metadata", None)
                or getattr(getattr(node, "node", None), "metadata", None)
                or {}
            )
            val = isinstance(md, dict) and (
                md.get("file_name")
                or md.get("source")
                or md.get("file")
                or md.get("filename")
            )
            if isinstance(val, str):
                sources_raw.append(val)

        if not sources_raw and hasattr(resp, "get_formatted_sources"):
            formatted = resp.get_formatted_sources(length=2000)
            sources_raw = re.findall(r"([^\s/\\]+\.pdf)", str(formatted))

        seen = set()
        sources: list[str] = []
        for s in sources_raw:
            base = _basename(s)
            if base not in seen:
                seen.add(base)
                sources.append(base)

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
