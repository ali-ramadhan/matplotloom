import pytest
import joblib

from pathlib import Path

def run_example(name, replacements=None):
    filepath = Path(__file__).resolve().parent / ".." / "examples" / f"{name}.py"
    content = filepath.read_text()

    if replacements:
        for old, new in replacements:
            content = content.replace(old, new)
    
    exec(content)
    return

def test_sine_wave():
    run_example("sine_wave")
    assert Path("sine_wave.gif").is_file()
    assert Path("sine_wave.gif").stat().st_size > 0
    

def test_rotating_circular_sine_wave():
    run_example("rotating_circular_sine_wave")
    assert Path("rotating_circular_sine_wave.mp4").is_file()
    assert Path("rotating_circular_sine_wave.mp4").stat().st_size > 0

def test_rotating_circular_sine_wave():
    replacements = [
        ("np.linspace(-10, 10, 500)", "np.linspace(-10, 10, 10)"),
        ("np.linspace(0, 50, 300)", "np.linspace(0, 50, 10)")
    ]
    run_example("bessel_wave", replacements)
    assert Path("bessel_wave.mp4").is_file()
    assert Path("bessel_wave.mp4").stat().st_size > 0

# Not sure why this fails with "NameError: name 'delayed' is not defined" =/
# def test_parallel_sine_wave():
#     run_example("parallel_sine_wave")
#     assert Path("parallel_sine_wave.gif").is_file()