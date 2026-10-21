"""Shared FastAPI dependencies."""

from functools import lru_cache

from .config import settings
from .services.ner_model import SkillExtractor


@lru_cache(maxsize=1)
def get_extractor() -> SkillExtractor:
    """Load the checkpoint once and keep it in memory."""
    return SkillExtractor(
        model_dir=settings.model_dir,
        backend=settings.model_backend,
        release_base=settings.model_release_base,
        allow_download=settings.allow_model_download,
    )
