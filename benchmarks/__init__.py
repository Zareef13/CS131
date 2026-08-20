"""RiskSight benchmark and evaluation tooling (not production pipeline code)."""

# Keep source-checkout benchmark commands usable without requiring an editable install.
import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parents[1] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
