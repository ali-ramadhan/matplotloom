import pytest
import shutil
from pathlib import Path

@pytest.fixture(scope="session")
def test_output_dir():
    """
    Create and return the tests/output directory for test output files.
    This fixture creates the directory if it doesn't exist, cleans up any existing files
    for each test, and returns the Path object.
    """
    output_dir = Path(__file__).parent / "output"

    # Clean up any existing files
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Create fresh directory
    output_dir.mkdir(exist_ok=True)

    return output_dir
