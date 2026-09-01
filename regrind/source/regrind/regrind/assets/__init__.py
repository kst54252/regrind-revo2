import os
from pathlib import Path

REGRIND_ASSETS_DIR = Path(__file__).resolve().parent

# Repository-level assets used by the RB3+Revo2 integration are intentionally
# reused rather than copied into the Python package.  The environment variable
# keeps editable/install layouts relocatable.
REGRIND_PROJECT_ROOT = Path(
    os.environ.get("REGRIND_PROJECT_ROOT", Path(__file__).resolve().parents[5])
).expanduser().resolve()
