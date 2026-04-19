import subprocess, os
import warnings

from pathlib import Path
from typing import Literal, Union, Optional, Dict, Type, List, Any
from types import TracebackType
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from IPython.display import Video, Image

# This should allow 
_LOOM_DEFAULT_ENVIRON_VAR = "LOOM_FFMPEG_PATH"

if _LOOM_DEFAULT_ENVIRON_VAR in os.environ:
    DEFAULT_FFMPEG_PATH = os.environ[_LOOM_DEFAULT_ENVIRON_VAR]
else:
    DEFAULT_FFMPEG_PATH: str = plt.rcParams['animation.ffmpeg_path']
    os.environ[_LOOM_DEFAULT_ENVIRON_VAR] = DEFAULT_FFMPEG_PATH

VALID_SCALE_ODD_OPTIONS = {"round_up", "round_down", "crop", "pad", "none"}

class Loom:
    """
    A class for creating animations from matplotlib figures.

    This class provides functionality to save individual frames and compile them into
    an animation (video or GIF) using ffmpeg.

    Parameters
    ----------
    output_filepath : Union[Path, str]
        Path to save the final animation file.
    fps : int, optional
        Frames per second for the output animation. Default is 30.
    frames_directory : Union[Path, str, None], optional
        Directory to save individual frames. If None, a temporary directory is used.
    keep_frames : bool, optional
        Whether to keep individual frame files after creating the animation.
        Default is False.
    overwrite : bool, optional
        Whether to overwrite the output file if it already exists. Default is False.
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
    odd_dimension_handling : str, optional
        How to handle odd pixel dimensions that some codecs (like H.264) cannot process.
        Options:
        - "round_up": Scale filter rounds up to next even dimension (default)
        - "round_down": Scale filter rounds down to previous even dimension
        - "crop": Crop 1 pixel from bottom/right edges if dimensions are odd
        - "pad": Add 1 pixel of padding to bottom/right edges if dimensions are odd
        - "none": Do nothing, let FFmpeg handle it (may fail for some codecs)
        Default is "round_up".
    savefig_kwargs : dict, optional
        Additional keyword arguments to pass to matplotlib's savefig function.
        Default is {}.
    verbose : bool, optional
        Whether to print detailed information during the process. Default is False.
    show_ffmpeg_output : bool, optional
        Whether to show ffmpeg output when saving the video. Default is False.
        When True, the ffmpeg command and its stdout/stderr output will be printed
        during video creation, regardless of the verbose setting.
    ffmpeg_path : Union[Path, str, None], optional
        Path to ffmpeg, if not provided will use the default path.
            Default path is configured to use matplotlib's rcParams ffmpeg path

    Raises
    ------
    FileExistsError
        If the output file already exists and overwrite is False.
    """
    def __init__(
        self,
        output_filepath: Union[Path, str],
        fps: int = 30,
        frames_directory: Optional[Union[Path, str]] = None,
        keep_frames: bool = False,
        overwrite: bool = False,
        parallel: bool = False,
        odd_dimension_handling: str = "round_up",
        savefig_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
        show_ffmpeg_output: bool = False,
        ffmpeg_path: Optional[Union[Path, str]] = None,
        enable_ffmpeg_path_fallback: bool = True,
    ) -> None:
        self.output_filepath: Path = Path(output_filepath)
        self.fps: int = fps
        self.keep_frames: bool = keep_frames
        self.overwrite: bool = overwrite
        self.verbose: bool = verbose
        self.parallel: bool = parallel
        self.show_ffmpeg_output: bool = show_ffmpeg_output
        self.savefig_kwargs: Dict[str, Any] = savefig_kwargs or {}
        self.enable_ffmpeg_path_fallback = enable_ffmpeg_path_fallback
       
        if odd_dimension_handling not in VALID_SCALE_ODD_OPTIONS:
            raise ValueError(
                f"odd_dimension_handling must be one of {VALID_SCALE_ODD_OPTIONS}, "
                f"got {odd_dimension_handling}"
            )
        self.odd_dimension_handling: str = odd_dimension_handling
        self._get_scale_filter() # Should throw value error if wrong

        if self.output_filepath.exists() and not self.overwrite:
            raise FileExistsError(
                f"Output file '{self.output_filepath}' already exists."
                f"Set `overwrite=True` to overwrite the file."
            )

        self._temp_dir: Optional[TemporaryDirectory] = None
        if frames_directory is None:
            self._temp_dir = TemporaryDirectory()
            self.frames_directory = Path(self._temp_dir.name)
        else:
            self.frames_directory = Path(frames_directory)
        
        # Allow providing of ffmpeg path to class instance
        self.ffmpeg_path: Path = Path(DEFAULT_FFMPEG_PATH)

        # Only throws an error if a path was provided
        if ffmpeg_path is not None:
            _ffmpeg_path = Path(ffmpeg_path)

            # If the path exists use it
            if _ffmpeg_path.exists():
                # Store the absolute path as things can get a bit funky with
                # path enrolment & multiprocessing.
                self.ffmpeg_path = _ffmpeg_path.absolute()

            else:
                # Otherwise check if path fallback is enabled & warn the user
                if self.enable_ffmpeg_path_fallback: 
                    warnings.warn(
                        f"Provided ffmpeg path of `{ffmpeg_path}` (resolving " +
                        f"to `{_ffmpeg_path}`) was not found! Using default " +
                        f"path of `{DEFAULT_FFMPEG_PATH}`"
                    )
                    
                # If path fallback is not enabled, raise an error
                else:
                    raise FileNotFoundError(
                        f"Provided ffmpeg path of `{ffmpeg_path}` (resolving " +
                        f"to `{_ffmpeg_path}`) was not found!"
                    )            
                
        # In theory this should never fail.
        assert isinstance(self.ffmpeg_path, Path), "ffmpeg path is not a valid Path object?"

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
            frame_number: Optional[int] = None,
            close_fig: bool = True,
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
            assert self.frame_counter is not None

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
        if close_fig:
            plt.close(fig)

    def _get_scale_filter(self) -> str:
        """
        Generate the appropriate scale filter based on odd_dimension_handling setting.

        Returns
        -------
        str
            FFmpeg scale filter string for handling odd dimensions.
        """
        if self.odd_dimension_handling == "none":
            return ""
        elif self.odd_dimension_handling == "round_up":
            return "scale='if(mod(iw,2),iw+1,iw)':'if(mod(ih,2),ih+1,ih)':flags=lanczos"
        elif self.odd_dimension_handling == "round_down":
            return "scale='if(mod(iw,2),iw-1,iw)':'if(mod(ih,2),ih-1,ih)':flags=lanczos"
        elif self.odd_dimension_handling == "crop":
            return "crop='if(mod(iw,2),iw-1,iw)':'if(mod(ih,2),ih-1,ih)':0:0"
        elif self.odd_dimension_handling == "pad":
            return "pad='if(mod(iw,2),iw+1,iw)':'if(mod(ih,2),ih+1,ih)':0:0:color=white"
        else:
            raise ValueError(f"Scale Settings not one of `{VALID_SCALE_ODD_OPTIONS}`, " +
                             f"got `{self.odd_dimension_handling}`")

    def save_video(self) -> None:
        """
        Compile saved frames into a video or GIF using ffmpeg.

        This method uses ffmpeg to create the final animation from the saved frames.
        The output format is determined by the file extension of the output filepath.
        """
        # Handle odd pixel dimensions based on user preference
        # See: https://github.com/ali-ramadhan/matplotloom/issues/1
        scale_filter = self._get_scale_filter()

        if self.file_format == "mp4":
            command = [
                str(self.ffmpeg_path),
                "-y",
                "-framerate", str(self.fps),
                "-i", str(self.frames_directory / "frame_%06d.png"),
            ]

            if scale_filter:
                command.extend(["-vf", scale_filter])

            command.extend([
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                str(self.output_filepath)
            ])
        elif self.file_format == "gif":
            command = [
                str(self.ffmpeg_path),
                "-y",
                "-framerate", str(self.fps),
                "-f", "image2",
                "-i", str(self.frames_directory / "frame_%06d.png"),
            ]

            # See: https://superuser.com/a/556031 for the split and palette filters
            if scale_filter:
                gif_filter = f"{scale_filter},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"
            else:
                gif_filter = "split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse"

            command.extend(["-vf", gif_filter, str(self.output_filepath)])
        else:
            raise ValueError("Export File Format Not Valid!")

        PIPE = subprocess.PIPE
        process = subprocess.Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()

        if self.verbose or self.show_ffmpeg_output:
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
