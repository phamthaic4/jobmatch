"""Runtime configuration — every value can be overridden by an env var."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "models" / "phobert-ner"

# .env is optional: on a deployed platform the values come from real env vars.
load_dotenv(ROOT / ".env")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = "JobMatch API"
    version: str = "0.1.0"
    model_dir: Path = DEFAULT_MODEL_DIR
    # "onnx" (2.8x faster on CPU) or "transformers" (plain PyTorch)
    model_backend: str = "onnx"
    model_release_base: str = (
        "https://github.com/phamthaic4/vietnamese-job-skill-ner/releases/download/v1.0.0"
    )
    # set to 0 in production images where the weights are baked in
    allow_model_download: bool = True
    max_text_length: int = 8000

    @classmethod
    def from_env(cls) -> "Settings":
        model_dir = os.getenv("JOBMATCH_MODEL_DIR")
        return cls(
            model_dir=Path(model_dir).resolve() if model_dir else DEFAULT_MODEL_DIR,
            model_backend=os.getenv("JOBMATCH_MODEL_BACKEND", cls.model_backend).lower(),
            allow_model_download=_env_bool("JOBMATCH_ALLOW_DOWNLOAD", cls.allow_model_download),
            max_text_length=int(os.getenv("JOBMATCH_MAX_TEXT_LENGTH", cls.max_text_length)),
        )


settings = Settings.from_env()
