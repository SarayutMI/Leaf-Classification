# src/main.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)
