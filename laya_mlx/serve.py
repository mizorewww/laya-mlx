"""HTTP server exposing Laya over the TypeSafe Jev ``/v1/systemone`` protocol.

The server is optional so importing :mod:`laya_mlx` remains lightweight. Install
``laya-mlx[serve]`` and run ``laya-mlx-serve`` to expose the same request and
response shape as the upstream Laya server.

Environment variables:

``LAYA_HOST`` (``0.0.0.0``), ``LAYA_PORT`` (``8000``), ``LAYA_DEVICE`` (MLX
default), ``LAYA_DTYPE`` (``float16``), ``LAYA_PRELOAD`` (``1``),
``LAYA_MODELS`` (all models), ``LAYA_AUTO_TASK`` (``0``), ``LAYA_API_KEY``
(unset), ``LAYA_MAX_LEN`` (checkpoint default), and ``LAYA_LOG_LEVEL``
(``info``).
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

_KNOWN_MODELS = {"english", "multilingual", "typed-decisions"}
_PUBLISHED_MODEL_IDS = {
    "convaiinnovations/laya-multilingual": "multilingual",
    "convaiinnovations/laya-typed-decisions": "typed-decisions",
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_positive_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _resolve_model(model: Optional[str]) -> Optional[str]:
    """Map a client model id to a local checkpoint, or return ``None`` to route."""
    if not model:
        return None
    published = _PUBLISHED_MODEL_IDS.get(str(model).strip().lower())
    if published is not None:
        return published
    from .router import normalise_name

    try:
        key = normalise_name(model)
    except ValueError:
        return None
    return key if key in _KNOWN_MODELS else None


def build_router():
    """Build and optionally preload a router from environment configuration."""
    from .router import Router

    models_env = os.environ.get("LAYA_MODELS", "").strip()
    names = [name.strip() for name in models_env.split(",") if name.strip()] or None
    router = Router(
        device=os.environ.get("LAYA_DEVICE") or None,
        auto_task_detection=_env_bool("LAYA_AUTO_TASK", False),
        dtype=os.environ.get("LAYA_DTYPE", "float16"),
        max_len=_env_positive_int("LAYA_MAX_LEN"),
    )
    if _env_bool("LAYA_PRELOAD", True):
        router.preload(names)
    return router


def create_app(router: Optional[Any] = None):
    """Create the FastAPI app, optionally injecting a router for testing."""
    from fastapi import FastAPI, Header, HTTPException, Request

    if router is None:
        router = build_router()
    api_key = os.environ.get("LAYA_API_KEY") or None
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="laya-mlx-infer")
    gate: Optional[asyncio.Lock] = None

    app = FastAPI(
        title="laya-mlx-serve",
        summary="Laya MLX decisions over the TypeSafe Jev /v1/systemone protocol",
    )

    def check_auth(authorization: Optional[str]) -> None:
        if api_key is not None and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid or missing bearer token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "loaded": router.loaded,
            "device": os.environ.get("LAYA_DEVICE") or "auto",
        }

    @app.post("/v1/systemone")
    async def systemone(request: Request, authorization: Optional[str] = Header(default=None)):
        nonlocal gate
        check_auth(authorization)
        try:
            body = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="request body must be valid JSON")
        if not isinstance(body, dict) or "questions" not in body:
            raise HTTPException(
                status_code=400,
                detail="request body must be an object with a 'questions' field",
            )
        if gate is None:
            gate = asyncio.Lock()
        state = body.get("state")
        questions = body["questions"]
        model = _resolve_model(body.get("model"))
        try:
            async with gate:
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(
                    pool,
                    lambda: router.predict(state, questions, model=model),
                )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main() -> None:
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("LAYA_HOST", "0.0.0.0"),
        port=int(os.environ.get("LAYA_PORT", "8000")),
        log_level=os.environ.get("LAYA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
