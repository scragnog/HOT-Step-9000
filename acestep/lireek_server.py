"""Lightweight Lireek API server — lyrics DB + LLM profiling.

Runs independently of the main ACE-Step generation server.
No ML model loading — starts in <1 second.

Usage:
    python acestep/lireek_server.py --port 8002 --host 127.0.0.1
"""

from __future__ import annotations

import argparse
import os
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Lireek API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "http://localhost", "http://127.0.0.1"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ---------------------------------------------------------------------------
# Initialise Lireek SQLite database
# ---------------------------------------------------------------------------

from acestep.api.lireek.lireek_db import init_db as _init_lireek_db  # noqa: E402

_init_lireek_db()

# ---------------------------------------------------------------------------
# Register routes (same functions used by the main server)
# ---------------------------------------------------------------------------

from acestep.api.http.lireek_routes import register_lireek_routes  # noqa: E402
from acestep.api.http.llm_routes import register_llm_routes  # noqa: E402

register_lireek_routes(app=app)
register_llm_routes(app=app)


@app.get("/health")
async def health():
    """Health probe for the loading screen and monitoring."""
    return {"status": "ok", "service": "lireek"}


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Lireek API server")
    parser.add_argument("--port", type=int, default=int(os.getenv("ACESTEP_LIREEK_PORT", "8002")))
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    import uvicorn

    print(f"[Lireek] Starting Lireek API server on {args.host}:{args.port}")
    uvicorn.run(
        "acestep.lireek_server:app",
        host=args.host,
        port=args.port,
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
