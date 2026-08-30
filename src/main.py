# src/main.py
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from src.auth.router import router as auth_router
from src.classify.router import router as classify_router
from src.config import settings
from src.health.router import router as health_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src import seed
    from src.classify import service as leaf_service

    seed.run()

    try:
        leaf_service.load_models()
    except Exception as e:
        raise RuntimeError(f"Failed to load models — API will not start: {e}") from e

    try:
        leaf_service.warmup_models()
    except Exception:
        logger.warning("Model warmup failed — server will still start, first request may be slower", exc_info=True)

    yield


app = FastAPI(
    title="Leaf Classification API",
    version="1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(auth_router)
app.include_router(classify_router)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Report request-shape errors in this API's own envelope, by index.

    FastAPI's default 422 is a flat `detail` list whose `loc` path the caller
    has to parse. Each error here names the row `index` (null when the field is
    not inside a list) and the `field`.
    """
    fields = []
    for error in exc.errors():
        # loc looks like ("body", "items", 0, "name") — the int is the row.
        location = [part for part in error["loc"] if part != "body"]
        index = next((part for part in location if isinstance(part, int)), None)
        names = [part for part in location if isinstance(part, str)]
        fields.append({
            "index": index,
            "field": names[-1] if names else None,
            "message": error["msg"],
        })

    return JSONResponse(
        status_code=422,
        content={
            "code": 422,
            "status": "error",
            "message": "Validation failed",
            "errors": {"type": "VALIDATION_ERROR", "fields": fields},
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
