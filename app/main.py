from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.chat.rate_limit import SlidingWindowLimiter
from app.config import get_settings
from app.content.knowledge_base import KnowledgeBase
from app.db.database import init_db, make_engine
from app.llm.client import make_client, verify_models
from app.schemas import HealthResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    Path(settings.data_dir, "cache").mkdir(parents=True, exist_ok=True)
    Path(settings.data_dir, "index").mkdir(parents=True, exist_ok=True)

    engine = make_engine(settings.database_url)
    app.state.async_session = await init_db(engine)

    app.state.settings = settings
    app.state.kb = KnowledgeBase(settings)
    app.state.llm_client = make_client(settings)
    app.state.rate_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, window_seconds=60.0)

    if settings.openai_api_key:
        asyncio.create_task(verify_models(app.state.llm_client, settings))
    else:
        logger.error("OPENAI_API_KEY is not set in backend/.env — every chat request will fail")

    asyncio.create_task(app.state.kb.ensure_loaded(settings.default_plan_id_list))

    yield

    await app.state.llm_client.close()
    await engine.dispose()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="ToppScholars Doubt Solver", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from app.api import admin as admin_routes
    from app.api import catalog as catalog_routes
    from app.api import chat as chat_routes
    from app.api import debug as debug_routes
    from app.api import feedback as feedback_routes
    from app.api import sessions as sessions_routes

    app.include_router(catalog_routes.router, prefix="/api")
    app.include_router(sessions_routes.router, prefix="/api")
    app.include_router(chat_routes.router, prefix="/api")
    app.include_router(feedback_routes.router, prefix="/api")
    app.include_router(admin_routes.router, prefix="/api")
    app.include_router(debug_routes.router, prefix="/api")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        if isinstance(detail, dict) and "error" in detail:
            return JSONResponse(status_code=exc.status_code, content=detail)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": "error", "message": str(detail)}},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request, _exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_input", "message": "Invalid request"}},
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        kb: KnowledgeBase = request.app.state.kb
        return HealthResponse(
            status="ok",
            plans=kb.health_snapshot(),
            models={"router": settings.router_model, "answer": settings.answer_model},
        )

    return app


app = create_app()
