import pytest
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def test_output_dir():
    """Create and return the tests/output directory for test output files."""
    output_dir = Path(__file__).parent / "output"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(exist_ok=True)

    return output_dir
