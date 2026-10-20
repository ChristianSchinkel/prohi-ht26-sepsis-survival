"""Predictive view: estimated in-hospital outcome (analysis by Tauska / Xuehai Wang).

The models, evaluation and page content live in the `src/predictive`
package; the frozen models are in `predictive_results/` at the repository
root. This file only makes the package importable and renders it.
"""
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from predictive.ui import render  # noqa: E402

render()
