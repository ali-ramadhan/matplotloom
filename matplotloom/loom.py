import subprocess
from typing import Union
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from IPython.display import Video, Image

class Loom:
    def __init__(
        self,
        output_filepath: Union[Path, str],
        frames_directory: Union[Path, str] = Path(TemporaryDirectory().name),
        fps: int = 30,
        keep_frames: bool = False,
        overwrite: bool = False,
        verbose: bool = False,
        parallel: bool = False,
        savefig_kwargs: dict = {}
    ) -> None:
        self.output_filepath = Path(output_filepath)
        self.frames_directory = Path(frames_directory)
        self.fps = fps
        self.keep_frames = keep_frames
        self.overwrite = overwrite
        self.verbose = verbose
        self.parallel = parallel
        self.savefig_kwargs = savefig_kwargs
        
        if not self.parallel:
            self.frame_counter = 0
        else:
            # We don't use the frame counter in parallel mode.
            self.frame_counter = None

        self.frame_filepaths = []
        self.file_format = self.output_filepath.suffix[1:]

        self.output_directory = self.output_filepath.parent
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.frames_directory.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"output_filepath: {self.output_filepath}")
            print(f"frames_directory: {self.frames_directory}")
        
        if self.output_filepath.exists() and not self.overwrite:
            raise FileExistsError(f"Output file '{self.output_filepath}' already exists. Set `overwrite=True` to overwrite the file.")
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.save_video()
        return
    
    def save_frame(self, fig, frame_number=None):
        if not self.parallel:
            frame_filepath = self.frames_directory / f"frame_{self.frame_counter:06d}.png"
            self.frame_counter += 1
        else:
            frame_filepath = self.frames_directory / f"frame_{frame_number:06d}.png"
        
        self.frame_filepaths.append(frame_filepath)
        
        if self.verbose:
            if not self.parallel:
                print(f"Saving frame {self.frame_counter} to {frame_filepath}")
            else:
                print(f"Saving frame {frame_number} to {frame_filepath}")
        
        fig.savefig(frame_filepath, **self.savefig_kwargs)
        plt.close(fig)

    def save_video(self):
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

    def show(self):
        if self.file_format in {"mp4", "mkv"}:
            return Video(str(self.output_filepath))
        elif self.file_format in {"gif", "apng"}:
            return Image(str(self.output_filepath))
