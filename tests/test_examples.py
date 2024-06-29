import pytest

import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from cmocean import cm
from scipy.special import j0
from joblib import Parallel, delayed
from matplotloom import Loom

def test_sine_wave():
    with Loom("sine_wave.gif", fps=30) as loom:
        for phase in np.linspace(0, 2*np.pi, 100):
            fig, ax = plt.subplots()

            x = np.linspace(0, 2*np.pi, 200)
            y = np.sin(x + phase)
            
            ax.plot(x, y)
            ax.set_xlim(0, 2*np.pi)
            
            loom.save_frame(fig)

    assert Path("sine_wave.gif").is_file()
    assert Path("sine_wave.gif").stat().st_size > 0
    

def test_rotating_circular_sine_wave():
    with Loom("rotating_circular_sine_wave.mp4", fps=10) as loom:
        for i in range(5):
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": "3d"})
            
            X = np.arange(-5, 5, 0.25)
            Y = np.arange(-5, 5, 0.25)
            X, Y = np.meshgrid(X, Y)
            R = np.sqrt(X**2 + Y**2)
            Z = np.sin(R)
            
            surf = ax.plot_surface(X, Y, Z, cmap="coolwarm")
            
            ax.view_init(azim=i*10)
            ax.set_zlim(-1.01, 1.01)
            fig.colorbar(surf, shrink=0.5, aspect=5)

            loom.save_frame(fig)

    assert Path("rotating_circular_sine_wave.mp4").is_file()
    assert Path("rotating_circular_sine_wave.mp4").stat().st_size > 0

def test_bessel_wave():
    def bessel_wave(r, t, k, omega, A):
        return A * j0(k*r - omega*t)

    def create_frame(x, y, t):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        r = np.sqrt(x**2 + y**2)
        z = bessel_wave(r, t, k=2, omega=1, A=1)
        
        pcm = ax1.pcolormesh(x, y, z, cmap=cm.balance, shading='auto', vmin=-1, vmax=1)
        ax1.set_title(f"Bessel wave: t = {t:.3f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        fig.colorbar(pcm, ax=ax1)
        
        mid = z.shape[0] // 2
        ax2.plot(x[mid], z[mid])
        ax2.set_xlim(x.min(), x.max())
        ax2.set_ylim(-1.1, 1.1)
        ax2.set_title("Cross-section at y = 0")
        ax2.set_xlabel("x")
        ax2.set_ylabel("z")
        
        return fig

    loom = Loom(
        "bessel_wave.mp4",
        fps = 30,
        overwrite = True,
        verbose = True,
        savefig_kwargs = {
            "dpi": 100,
            "bbox_inches": "tight"
        }
    )

    with loom:
        x = np.linspace(-10, 10, 10)
        y = np.linspace(-10, 10, 10)
        x, y = np.meshgrid(x, y)

        for t in np.linspace(0, 50, 5):
            fig = create_frame(x, y, t)
            loom.save_frame(fig)

    assert Path("bessel_wave.mp4").is_file()
    assert Path("bessel_wave.mp4").stat().st_size > 0

def test_parallel_sine_wave():
    def plot_frame(phase, frame_number, loom):
        fig, ax = plt.subplots()

        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x + phase)
        
        ax.plot(x, y)
        ax.set_xlim(0, 2*np.pi)
        
        loom.save_frame(fig, frame_number)

    with Loom("parallel_sine_wave.gif", fps=30, parallel=True) as loom:
        phases = np.linspace(0, 2*np.pi, 10)
        
        Parallel(n_jobs=-1)(
            delayed(plot_frame)(phase, i, loom) 
            for i, phase in enumerate(phases)
        )

    assert Path("parallel_sine_wave.gif").is_file()