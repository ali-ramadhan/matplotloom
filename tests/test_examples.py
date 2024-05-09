import pytest

from pathlib import Path

def test_sine_wave():
    exec(Path("../examples/sine_wave.py").read_text())
    assert Path("sine_wave.gif").is_file()

def test_rotating_circular_sine_wave():
    exec(Path("../examples/rotating_circular_sine_wave.py").read_text())
    assert Path("rotating_circular_sine_wave.mp4").is_file()
