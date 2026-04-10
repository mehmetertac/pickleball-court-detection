"""Default court model path."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_COURT_MODEL_PATH = PROJECT_ROOT / "models" / "court_detector" / "best.pt"


def resolve_path_str(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolved_court_model_path() -> Path:
    return resolve_path_str(os.environ.get("PICKLEBALL_COURT_MODEL", DEFAULT_COURT_MODEL_PATH))
