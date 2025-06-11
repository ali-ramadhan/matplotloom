import subprocess

from pathlib import Path
from typing import Union, Optional, Dict, Type, List, Any
from types import TracebackType
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from IPython.display import Video, Image

class Loom:
    """
    A class for creating animations from matplotlib figures.

    This class provides functionality to save individual frames and compile them into
    an animation (video or GIF) using ffmpeg.

    Parameters
    ----------
    output_filepath : Union[Path, str]
        Path to save the final animation file.
    frames_directory : Union[Path, str, None], optional
        Directory to save individual frames. If None, a temporary directory is used.
    fps : int, optional
        Frames per second for the output animation. Default is 30.
    keep_frames : bool, optional
        Whether to keep individual frame files after creating the animation. Default is False.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Default is False.
    verbose : bool, optional
        Whether to print detailed information during the process. Default is False.
    parallel : bool, optional
        Whether to enable parallel frame saving. Default is False.
        When True, this enables a mode where frames can be saved concurrently,
        significantly speeding up the animation creation process for computationally
        intensive plots or large numbers of frames.

        In parallel mode:
            - The `save_frame` method requires an explicit frame number.
            - Frames can be created and saved in any order.
            - The user is responsible for parallelizing the frame creation process,
              typically using tools like joblib, multiprocessing, or concurrent.futures.

    savefig_kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib's savefig function. Default is {}.

    Raises
    ------
    FileExistsError
        If the output file already exists and overwrite is False.
    """
    def __init__(
        self,
        output_filepath: Union[Path, str],
        frames_directory: Optional[Union[Path, str]] = None,
        fps: int = 30,
        keep_frames: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        parallel: bool = False,
        savefig_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        self.output_filepath: Path = Path(output_filepath)
        self.fps: int = fps
        self.keep_frames: bool = keep_frames
        self.overwrite: bool = overwrite
        self.verbose: bool = verbose
        self.parallel: bool = parallel
        self.savefig_kwargs: Dict[str, Any] = savefig_kwargs or {}

        if self.output_filepath.exists() and not self.overwrite:
            raise FileExistsError(f"Output file '{self.output_filepath}' already exists. Set `overwrite=True` to overwrite the file.")

        self._temp_dir: Optional[TemporaryDirectory] = None
        if frames_directory is None:
            self._temp_dir = TemporaryDirectory()
            self.frames_directory = Path(self._temp_dir.name)
        else:
            self.frames_directory = Path(frames_directory)

        # We don't use the frame counter in parallel mode.
        self.frame_counter: Optional[int] = 0 if not self.parallel else None

        self.frame_filepaths: List[Path] = []
        self.file_format: str = self.output_filepath.suffix[1:]

        self.output_directory = self.output_filepath.parent
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"output_filepath: {self.output_filepath}")
            print(f"frames_directory: {self.frames_directory}")

    def __enter__(self) -> 'Loom':
        """
        Enter the runtime context related to this object.

        Returns
        -------
        Loom
            The Loom instance.
        """
        return self

    def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[TracebackType]
        ) -> bool:
        """
        Exit the runtime context related to this object.

        This method ensures that the video is saved if no exception occurred,
        and cleans up temporary files.

        Parameters
        ----------
        exc_type : type
            The exception type if an exception was raised, else None.
        exc_value : Exception
            The exception value if an exception was raised, else None.
        traceback : traceback
            The traceback if an exception was raised, else None.

        Returns
        -------
        bool
            False to propagate exceptions if any occurred.
        """
        try:
            if exc_type is None:
                self.save_video()
            else:
                if self.verbose:
                    print(f"An error occurred: {exc_type.__name__}: {exc_value}")
                    print("Animation was not saved.")
        finally:
            if not self.keep_frames:
                for frame_filepath in self.frame_filepaths:
                    if frame_filepath.exists():
                        frame_filepath.unlink()

            if self._temp_dir:
                self._temp_dir.cleanup()

        return False  # Propagate the exception if there was one

    def save_frame(
            self,
            fig: Figure,
            frame_number: Optional[int] = None
        ) -> None:
        """
        Save a single frame of the animation.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The matplotlib figure to save.
        frame_number : int, optional
            The frame number (required if parallel=True).
        """
        if self.parallel and frame_number is None:
            raise ValueError("frame_number must be provided when parallel=True")

        if not self.parallel:
            frame_filepath = self.frames_directory / f"frame_{self.frame_counter:06d}.png"
            self.frame_counter += 1
        else:
            frame_filepath = self.frames_directory / f"frame_{frame_number:06d}.png"

        self.frame_filepaths.append(frame_filepath)

        if self.verbose:
            if not self.parallel:
                print(f"Saving frame {self.frame_counter - 1} to {frame_filepath}")
            else:
                print(f"Saving frame {frame_number} to {frame_filepath}")

        fig.savefig(frame_filepath, **self.savefig_kwargs)
        plt.close(fig)

    def save_video(self) -> None:
        """
        Compile saved frames into a video or GIF using ffmpeg.

        This method uses ffmpeg to create the final animation from the saved frames.
        The output format is determined by the file extension of the output filepath.
        """
        # Scale video in case number of pixels in either dimensions is odd.
        # See: https://github.com/ali-ramadhan/matplotloom/issues/1
        scale_filter = "scale='if(mod(iw,2),-2,iw)':'if(mod(ih,2),-2,ih)':flags=lanczos"

        if self.file_format == "mp4":
            command = [
                "ffmpeg",
                "-y",
                "-framerate", str(self.fps),
                "-i", str(self.frames_directory / "frame_%06d.png"),
                "-vf", scale_filter,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(self.output_filepath)
            ]
        elif self.file_format == "gif":
            command = [
                "ffmpeg",
                "-y",
                "-framerate", str(self.fps),
                "-f", "image2",
                "-i", str(self.frames_directory / "frame_%06d.png"),
                # See: https://superuser.com/a/556031 for the split and palette filters
                "-vf", f"{scale_filter},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
                str(self.output_filepath)
            ]

        PIPE = subprocess.PIPE
        process = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if self.verbose:
            print(" ".join(command))
            print(stdout.decode())
            print(stderr.decode())

        if not self.keep_frames:
            for frame_filename in self.frame_filepaths:
                if frame_filename.exists():
                    frame_filename.unlink()

    def show(self, **kwargs) -> Union[Video, Image]:
        """
        Display the created animation in a Jupyter notebook.

        This method returns an IPython display object that can be used to show
        the animation directly in a Jupyter notebook cell. The type of object
        returned depends on the file format of the animation.

        Parameters
        ----------
        **kwargs
            Keyword arguments that will be passed to either IPython.display.Video
            or IPython.display.Image constructor depending on the file format.

        Returns
        -------
        Union[IPython.display.Video, IPython.display.Image]
            An IPython Video object for MP4 or MKV formats, or an IPython Image
            object for GIF or APNG formats. These objects can be displayed
            directly in a Jupyter notebook.

        Notes
        -----
        This method is designed to work in Jupyter notebooks. It may not have
        the desired effect in other Python environments.

        The animation file must have been successfully created by the `save_video`
        method before calling this method.
        """
        if self.file_format in {"mp4", "mkv"}:
            return Video(self.output_filepath, **kwargs)
        elif self.file_format in {"gif", "apng"}:
            return Image(self.output_filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")
