from pathlib import Path

CWD = Path.cwd().resolve()
PROJECT_DIR = CWD.parent if CWD.name == "notebooks" else CWD

DATA_DIR = PROJECT_DIR / "data"
ARTIFACTS_DIR = PROJECT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42 

def artifact(*parts) -> str:
    """Return absolute path under artifacts/ as a string."""
    return str((ARTIFACTS_DIR / Path(*parts)).resolve())

def artifact(*parts) -> str:
    """Return absolute path under artifacts/ as a string (used by save_splits/load_splits)."""
    return str((ARTIFACTS_DIR / Path(*parts)).resolve())

def ensure_dirs() -> None:
    """Create required output dirs if missing (idempotent)."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def datafile(name: str) -> Path:
    """Convenience: path to a data file by name."""
    return DATA_DIR / name

__all__ = [
    "PROJECT_DIR", "DATA_DIR", "ARTIFACTS_DIR",
    "RANDOM_STATE", "ensure_dirs", "artifact", "datafile",
]