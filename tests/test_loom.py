import pytest

import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.figure import Figure
from matplotloom import Loom
from PIL import Image

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

def test_show_ffmpeg_output(tmp_path):
    output_filepath = tmp_path / "output.mp4"

    loom = Loom(
        output_filepath=output_filepath,
        show_ffmpeg_output=True,
        verbose=False
    )

    assert loom.show_ffmpeg_output is True
    assert loom.verbose is False

    fig = Figure()
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1])

    loom.save_frame(fig)
    loom.save_video()

    assert loom.output_filepath.exists()

    output_filepath2 = tmp_path / "output2.mp4"
    loom2 = Loom(output_filepath=output_filepath2)

    assert loom2.show_ffmpeg_output is False

    fig2 = Figure()
    ax2 = fig2.subplots()
    ax2.plot([1, 2], [1, 2])

    loom2.save_frame(fig2)
    loom2.save_video()

    assert loom2.output_filepath.exists()

@pytest.mark.parametrize("odd_handling", ["round_up", "round_down", "crop", "pad"])
def test_odd_pixel_dimensions(tmp_path, odd_handling):
    """
    Test that Loom can handle odd pixel dimensions properly with different handling
    options.
    """
    output_filepath = tmp_path / f"output_odd_{odd_handling}.mp4"

    loom = Loom(
        output_filepath=output_filepath,
        show_ffmpeg_output=True,
        verbose=True,
        keep_frames=True,  # Keep frames so we can verify their dimensions
        odd_dimension_handling=odd_handling,
        savefig_kwargs={
            "dpi": 100,
            "bbox_inches": None,  # Don't crop, keep exact dimensions
            "pad_inches": 0
        }
    )

    # Create figure with odd dimensions: 4.01 x 3.01 inches at 100 DPI = 401x301 pixels
    for i in range(3):
        fig = Figure(figsize=(4.01, 3.01), dpi=100)
        ax = fig.subplots()

        x = [0, 1, 2, 3]
        y = [i*0.5, 1+i*0.2, 0.5+i*0.3, i*0.4]

        ax.plot(x, y, "b-o")
        ax.set_title(f"Frame {i} - odd_dimension_handling={odd_handling} test")
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        loom.save_frame(fig)

    # Verify frame dimensions are still 401x301 (handling is done during video creation)
    for frame_path in loom.frame_filepaths:
        with Image.open(frame_path) as img:
            width, height = img.size
            print(f"Frame {frame_path.name} dimensions: {width}x{height}")
            assert width == 401, f"Frame {frame_path.name} width should be 401, got {width}"
            assert height == 301, f"Frame {frame_path.name} height should be 301, got {height}"

    loom.save_video()

    # All handling options should create a valid video
    video_created = output_filepath.exists()
    if video_created:
        file_size = output_filepath.stat().st_size
        if file_size == 0:
            pytest.fail(f"Video file was created but is empty with {odd_handling} handling")
    else:
        pytest.fail(f"Video file was not created with {odd_handling} handling")

    assert video_created and file_size > 0, f"Video should be created successfully with {odd_handling} handling"


def test_odd_dimension_handling_none_fails(tmp_path):
    """Test that 'none' handling fails with H.264 codec and odd dimensions."""
    output_filepath = tmp_path / "output_odd_none.mp4"

    loom = Loom(
        output_filepath=output_filepath,
        show_ffmpeg_output=True,
        verbose=True,
        keep_frames=True,
        odd_dimension_handling="none",
        savefig_kwargs={
            "dpi": 100,
            "bbox_inches": None,
            "pad_inches": 0
        }
    )

    # Create figure with odd dimensions
    fig = Figure(figsize=(4.01, 3.01), dpi=100)
    ax = fig.subplots()
    ax.plot([0, 1, 2, 3], [0, 1, 0.5, 0.4], "b-o")
    ax.set_title("Frame - none handling test")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    loom.save_frame(fig)

    loom.save_video()

    # With 'none' handling, the video should fail to be created or be empty
    video_created = output_filepath.exists()
    if video_created:
        file_size = output_filepath.stat().st_size
        # File might be created but empty due to ffmpeg failure
        assert file_size == 0, "Video with 'none' handling should fail due to odd dimensions"
    # If no file is created, that's also expected behavior


def test_odd_dimension_handling_validation():
    """Test that invalid odd_dimension_handling values raise ValueError."""
    with pytest.raises(ValueError, match="odd_dimension_handling must be one of"):
        Loom(
            output_filepath="test.mp4",
            odd_dimension_handling="invalid_option"
        )
