"""Conftest: put src/ on sys.path so tests import market_ml without install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
