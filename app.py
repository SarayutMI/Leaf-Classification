# app.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from routes.auth import router as auth_router
from routes.classify import router as classify_router
from routes.health import router as health_router
import config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    import seed
    from services import leaf as leaf_svc

    seed.run()

    try:
        leaf_svc.load_models()
    except Exception as e:
        raise RuntimeError(f"Failed to load models — API will not start: {e}") from e

    try:
        leaf_svc.warmup_models()
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
    uvicorn.run(app, host="0.0.0.0", port=config.PORT)
