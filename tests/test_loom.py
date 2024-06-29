import pytest

import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.figure import Figure
from matplotloom import Loom

def test_init_loom(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    loom = Loom(output_filepath=output_filepath, verbose=True)

    assert loom.output_directory.exists()
    assert loom.frames_directory.exists()

def test_init_loom_with_custom_frames_directory(tmp_path):
    output_filepath = tmp_path / "output.gif"
    custom_frames_dir = tmp_path / "frames"
    
    loom_with_custom_dir = Loom(
        output_filepath=output_filepath,
        frames_directory=custom_frames_dir,
        verbose=True
    )

    assert loom_with_custom_dir.output_directory.exists()
    assert loom_with_custom_dir.frames_directory.exists()

def test_save_frame(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    loom = Loom(output_filepath=output_filepath, verbose=True)
    
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    
    loom.save_frame(fig)
    assert len(loom.frame_filepaths) == 1
    assert loom.frame_filepaths[0].exists()

def test_video_creation(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    loom = Loom(output_filepath=output_filepath, verbose=True)
    
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    loom.save_frame(fig)
    loom.save_video()
    assert loom.output_filepath.exists()

def test_dont_keep_frames(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    loom = Loom(
        output_filepath=output_filepath,
        keep_frames=False,
        verbose=True
    )
    
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    loom.save_frame(fig)
    loom.save_video()

    assert not loom.frame_filepaths[0].exists()

def test_keep_frames(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    loom = Loom(
        output_filepath=output_filepath,
        keep_frames=True,
        verbose=True
    )

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    loom.save_frame(fig)
    loom.save_video()

    assert loom.frame_filepaths[0].exists()

def test_context_manager(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    with Loom(output_filepath=output_filepath) as loom:
        fig = Figure()
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1])
        loom.save_frame(fig)

    assert output_filepath.exists()

def test_overwrite(tmp_path):
    output_filepath = tmp_path / "output.mp4"

    with open(output_filepath, "w") as f:
        f.write("Dummy content")

    with pytest.raises(FileExistsError):
        Loom(output_filepath)

    loom = Loom(output_filepath, overwrite=True)

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    loom.save_frame(fig)
    loom.save_video()

    assert output_filepath.exists()
    assert output_filepath.stat().st_size > len("Dummy content")

def test_loom_error_handling(tmp_path):
    output_file = tmp_path / "test_error.mp4"

    with pytest.raises(ValueError, match="Test error"):
        with Loom(output_file, verbose=True) as loom:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            loom.save_frame(fig)
            raise ValueError("Test error")

    assert not output_file.exists()

    frames_dir = Path(loom.frames_directory)
    assert not any(frames_dir.glob("frame_*.png"))
