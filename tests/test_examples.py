import pytest

from pathlib import Path

def run_example(name):
    filepath = Path(__file__).resolve().parent / ".." / "examples" / f"{name}.py"
    exec(filepath.read_text())
    return

def test_sine_wave():
    run_example("sine_wave")
    assert Path("sine_wave.gif").is_file()

def test_rotating_circular_sine_wave():
    run_example("rotating_circular_sine_wave")
    assert Path("rotating_circular_sine_wave.mp4").is_file()
