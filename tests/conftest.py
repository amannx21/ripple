import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def base_instance():
    from freshroute.adapters import load_base_instance

    return load_base_instance("synthetic")
