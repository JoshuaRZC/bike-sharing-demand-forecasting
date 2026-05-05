from pathlib import Path


def find_project_root(start=None):
    start = Path.cwd() if start is None else Path(start)
    for path in [start.resolve(), *start.resolve().parents]:
        if (path / "data" / "hour.csv").exists():
            return path
    raise FileNotFoundError("Could not find data/hour.csv from the current directory.")


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
FIGURES_DIR = PROJECT_ROOT / "figures"

HOUR_DATA_PATH = DATA_DIR / "hour.csv"
DAY_DATA_PATH = DATA_DIR / "day.csv"

# Short alias used in notebooks.
DATA_PATH = HOUR_DATA_PATH
