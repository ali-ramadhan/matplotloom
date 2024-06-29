import pytest

import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.figure import Figure
from matplotloom import Loom

@pytest.fixture
def basic_loom(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    return Loom(output_filepath=output_filepath, verbose=True)

@pytest.fixture
def loom_with_custom_dir(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    frames_directory = tmp_path / "frames"
    return Loom(output_filepath=output_filepath, frames_directory=frames_directory, verbose=True)

def test_init(basic_loom, loom_with_custom_dir):
    assert basic_loom.output_directory.exists()
    assert basic_loom.frames_directory.exists()
    assert loom_with_custom_dir.frames_directory.exists()

# Test the saving of a single frame
def test_save_frame(basic_loom):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    
    basic_loom.save_frame(fig)
    assert len(basic_loom.frame_filepaths) == 1
    assert basic_loom.frame_filepaths[0].exists()

def test_video_creation(basic_loom):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    basic_loom.save_frame(fig)
    basic_loom.save_video()
    assert basic_loom.output_filepath.exists()

# Ensure frames are kept or deleted according to the `keep_frames` flag.
def test_keep_frames(basic_loom):
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    basic_loom.save_frame(fig)
    basic_loom.save_video()

    # Check if frames are deleted (default behavior)
    assert not basic_loom.frame_filepaths[0].exists()

    # Test with keep_frames=True
    basic_loom.keep_frames = True
    basic_loom.save_frame(fig)
    basic_loom.save_video()

    # Check if frames are kept
    assert basic_loom.frame_filepaths[1].exists()

def test_context_manager(tmp_path):
    output_filepath = tmp_path / "output.mp4"
    with Loom(output_filepath=output_filepath) as loom:
        fig = Figure()
        ax = fig.subplots()
        ax.plot([0, 1], [0, 1])
        loom.save_frame(fig)

    # After exiting the context, the video file should exist
    assert output_filepath.exists()

def test_overwrite(tmp_path):
    output_filepath = tmp_path / "output.mp4"

    # Create a dummy file to simulate an existing output file
    with open(output_filepath, "w") as f:
        f.write("Dummy content")

    # Test with `overwrite=False`` (default)
    with pytest.raises(FileExistsError):
        Loom(output_filepath)

    # Test with `overwrite=True`
    loom = Loom(output_filepath, overwrite=True)
    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])
    loom.save_frame(fig)
    loom.save_video()

    # Check if the output file was overwritten
    assert output_filepath.exists()
    assert output_filepath.stat().st_size > len("Dummy content")

# Test that `Loom`` handles errors correctly and doesn't invoke ffmpeg when an exception occurs.
def test_loom_error_handling(tmp_path):
    output_file = tmp_path / "test_error.mp4"    

    with pytest.raises(ValueError, match="Test error"):
        with Loom(output_file, verbose=True) as loom:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            loom.save_frame(fig)
            raise ValueError("Test error")

    # Check that the output file was not created
    assert not output_file.exists()

    # Check that no frame files remain
    frames_dir = Path(loom.frames_directory)
    assert not any(frames_dir.glob("frame_*.png"))
