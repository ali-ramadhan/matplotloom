# Welcome to matplotloom's documentation!

Matplotloom is a Python library for creating animations from matplotlib figures. It provides a simple interface to save individual frames and compile them into animations (video or GIF) using ffmpeg.

## Features

- Easy-to-use interface for creating animations
- Support for both sequential and parallel frame saving
- Automatic handling of temporary directories for frame storage
- Integration with Jupyter notebooks for easy viewing of created animations
- Support for various output formats including MP4, GIF, MKV, and APNG

## Installation

You can install matplotloom using pip:

```bash
pip install matplotloom
```

## Quick Start

Here's a simple example of how to use matplotloom:

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

def create_frame(frame_number):
    t = np.linspace(0, 10, 100)
    y = np.sin(t + frame_number / 10)
    
    fig, ax = plt.subplots()
    ax.plot(t, y)
    ax.set_title(f"Frame {frame_number}")
    return fig

with Loom("animation.mp4") as loom:
    for i in range(50):
        fig = create_frame(i)
        loom.save_frame(fig)

# The animation will be saved as "animation.mp4"
```

## Contents

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
usage
api
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`