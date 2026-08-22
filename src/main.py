# src/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from src.auth.router import router as auth_router
from src.classify.router import router as classify_router
from src.config import settings
from src.health.router import router as health_router
from src.rules.router import router as rules_router

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src import seed
    from src.classify import service as leaf_service

    # Signing admin sessions with a blank key would accept any forged cookie.
    if not settings.JWT_SECRET:
        raise RuntimeError("JWT_SECRET is not set — refusing to start")

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
app.include_router(rules_router)

# Rule-base admin page. Same origin as the API, so the session cookie rides
# along and no CORS configuration is needed.
app.mount("/admin", StaticFiles(directory=str(STATIC_DIR), html=True), name="admin")


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """Report request-shape errors in this API's own envelope, by index.

    FastAPI's default 422 is a flat `detail` list whose `loc` path the caller
    has to parse. The variety form submits many rows at once, so each error
    names the row `index` (null when the field is not inside a list) and the
    `field`, matching what the routes' own semantic checks return.
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


@app.get("/", include_in_schema=False)
async def root():
    """The bare host is a dead end otherwise — send it to the admin page."""
    return RedirectResponse(url="/admin/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
