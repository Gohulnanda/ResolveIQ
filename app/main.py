"""FastAPI application factory.

Run with:  uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.core.config import APP_VERSION, Settings, get_settings
from app.models.schemas import ErrorBody, ErrorDetail, ErrorResponse
from app.services.container import build_container

logger = logging.getLogger("resolveiq")

_STATUS_CODES = {
    400: "bad_request", 404: "not_found", 405: "method_not_allowed", 411: "length_required",
    413: "payload_too_large", 422: "validation_error", 503: "service_unavailable",
}


def _error(status: int, message: str, details: list[ErrorDetail] | None = None) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorBody(code=_STATUS_CODES.get(status, "error"), message=message, details=details)
    )
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            app.state.container = build_container(settings)
            logger.info("Services ready.")
        except Exception as exc:  # noqa: BLE001 - reported via /ready, full trace logged
            logger.exception("Startup failed.")
            app.state.container = None
            app.state.startup_error = f"{type(exc).__name__}: service initialisation failed"
        yield

    app = FastAPI(
        title="ResolveIQ - AI-Powered IT Support Intelligence",
        description=(
            "Classifies IT support tickets, retrieves relevant troubleshooting articles and "
            "returns a resolution grounded in them. Runs in demo mode (deterministic, no LLM) "
            "unless an LLM provider is configured."
        ),
        version=APP_VERSION,
        lifespan=lifespan,
    )
    app.state.container = None
    app.include_router(router)

    @app.middleware("http")
    async def limit_body_and_time(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.method in {"POST", "PUT", "PATCH"}:
            length = request.headers.get("content-length")
            if length is None:
                return _error(411, "A Content-Length header is required.")
            if not length.isdigit():
                return _error(400, "Invalid Content-Length header.")
            if int(length) > settings.max_request_bytes:
                return _error(413, f"Request body exceeds {settings.max_request_bytes} bytes.")
        started = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - started) * 1000:.1f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        details = [
            ErrorDetail(
                field=".".join(str(p) for p in err["loc"][1:]) or None,
                message=str(err["msg"]),
            )
            for err in exc.errors()
        ]
        return _error(422, "The request body failed validation.", details)

    @app.exception_handler(StarletteHTTPException)
    async def http_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return _error(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(
            "Unhandled error on %s %s", request.method, request.url.path, exc_info=exc
        )
        return _error(500, "An internal error occurred.")

    return app
