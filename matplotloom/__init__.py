import subprocess
import shutil
import warnings
from multiprocessing import current_process

from .loom import Loom, DEFAULT_FFMPEG_PATH, _LOOM_DEFAULT_ENVIRON_VAR

__version__ = "0.9.2"
__all__ = ["Loom"]


def _check_ffmpeg_availability():
    """Check if ffmpeg is available on the system."""
    try:
        # more reliable cross-platform
        if shutil.which(DEFAULT_FFMPEG_PATH) is not None:
            return True

        # Fallback: try running ffmpeg with subprocess
        subprocess.run(
            [DEFAULT_FFMPEG_PATH, "-version"],
            capture_output=True,
            check=True,
            timeout=5
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


if not _check_ffmpeg_availability() and current_process().name == 'MainProcess':
    warnings.warn(
        "ffmpeg is not available on your system. "
        "matplotloom requires ffmpeg to create animations. "
        "Please install ffmpeg to use this library. "
        "Visit https://ffmpeg.org/download.html for installation instructions. "
        f"Optionally ensure the environment variable `{_LOOM_DEFAULT_ENVIRON_VAR}`"
        " is set to a valid ffmpeg executable or configure "
        "`matplotlib.pyplot.rcParams['animation.ffmpeg_path']` to point to an "
        "ffmpeg executable. The current command used to run ffmpeg "
        f"is: `{DEFAULT_FFMPEG_PATH}`",
        UserWarning,
        stacklevel=2
    )
