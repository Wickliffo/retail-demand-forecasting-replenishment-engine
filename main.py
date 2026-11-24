import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIRECTORY = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIRECTORY))

from replenishment.pipeline import main


if __name__ == "__main__":
    main()
