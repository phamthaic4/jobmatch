"""JobMatch API — CV / JD skill gap analyzer.

Backend for the JobMatch project: a fine-tuned PhoBERT NER model extracts IT
skills from a CV and a job description, the two skill sets are compared, and
Gemini turns the gaps into interview questions.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .dependencies import get_extractor
from .routers import skills
from .schemas import HealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jobmatch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the model up at startup so the first request is not slow."""
    try:
        extractor = get_extractor()
        logger.info(
            "model ready: backend=%s dir=%s", extractor.backend, extractor.model_dir
        )
    except Exception as exc:  # noqa: BLE001 - keep the API up and report the problem
        logger.warning("model not loaded at startup: %s", exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=(
        "Extract IT skills from Vietnamese job descriptions and CVs with a "
        "custom fine-tuned PhoBERT NER model."
    ),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(skills.router)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe + model status."""
    try:
        extractor = get_extractor()
        loaded = True
        backend = extractor.backend
    except Exception:  # noqa: BLE001
        loaded = False
        backend = settings.model_backend

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        backend=backend,
        model_dir=str(settings.model_dir),
        version=settings.version,
    )
