# src/main.py
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
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


@app.get("/", include_in_schema=False)
async def root():
    """The bare host is a dead end otherwise — send it to the admin page."""
    return RedirectResponse(url="/admin/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
