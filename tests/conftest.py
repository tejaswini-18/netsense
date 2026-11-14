"""Shared test fixtures. Tests that need the real dataset are skipped when it is not available."""

import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _data_root():
    # 1) environment variable, 2) data.root from the baseline config (relative to the repo root)
    if os.environ.get("NETSENSE_DATA_ROOT"):
        return Path(os.environ["NETSENSE_DATA_ROOT"])
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "baseline.yaml").read_text())
    root = Path(cfg["data"]["root"])
    return root if root.is_absolute() else REPO_ROOT / root


@pytest.fixture(scope="session")
def data_root():
    root = _data_root()
    if not (root / "synthesisStatistics.pickle").exists():
        pytest.skip(f"dataset not found at {root} (set NETSENSE_DATA_ROOT)")
    return root
