import subprocess
import shutil
import warnings

from .loom import Loom

__version__ = "0.8.1"
__all__ = ["Loom"]


def _check_ffmpeg_availability():
    """Check if ffmpeg is available on the system."""
    try:
        # more reliable cross-platform
        if shutil.which("ffmpeg") is not None:
            return True

        # Fallback: try running ffmpeg with subprocess
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


if not _check_ffmpeg_availability():
    warnings.warn(
        "ffmpeg is not available on your system. "
        "matplotloom requires ffmpeg to create animations. "
        "Please install ffmpeg to use this library. "
        "Visit https://ffmpeg.org/download.html for installation instructions.",
        UserWarning,
        stacklevel=2
    )
