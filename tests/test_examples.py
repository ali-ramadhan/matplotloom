import datetime
import pytest

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from pathlib import Path
from cmocean import cm

from cartopy.feature.nightshade import Nightshade
from joblib import Parallel, delayed
from scipy.integrate import solve_ivp
from scipy.special import j0
from tqdm import tqdm

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
    assert Path("parallel_sine_wave.gif").stat().st_size > 0

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

def test_double_pendulum():
    g = 9.80665 # standard acceleration of gravity [m/s²]
    l1, l2 = 1, 1  # pendulum arms lengths [m]
    m1, m2 = 1, 1  # pendulum masses [kg]

    # Calculate dy/dt where y = [θ1, ω1, θ2, ω2].
    def derivatives(t, state):
        θ1, ω1, θ2, ω2 = state
        dydt = np.zeros_like(state)
        
        dydt[0] = ω1
        dydt[2] = ω2
        
        Δθ = θ2 - θ1

        denominator1 = (m1 + m2) * l1 - m2 * l1 * np.cos(Δθ)**2
        dydt[1] = (m2 * l1 * ω1**2 * np.sin(Δθ) * np.cos(Δθ)
                + m2 * g * np.sin(θ2) * np.cos(Δθ)
                + m2 * l2 * ω2**2 * np.sin(Δθ)
                - (m1 + m2) * g * np.sin(θ1)) / denominator1

        denominator2 = (l2 / l1) * denominator1
        dydt[3] = (-m2 * l2 * ω2**2 * np.sin(Δθ) * np.cos(Δθ)
                + (m1 + m2) * g * np.sin(θ1) * np.cos(Δθ)
                - (m1 + m2) * l1 * ω1**2 * np.sin(Δθ)
                - (m1 + m2) * g * np.sin(θ2)) / denominator2
        
        return dydt

    t_span = (0, 1)
    y0 = [np.pi/2, 0, np.pi/2, 0]
    sol = solve_ivp(derivatives, t_span, y0, dense_output=True)

    times = np.linspace(t_span[0], t_span[1], 10)

    θ1, ω1, θ2, ω2 = sol.sol(times)
    x1 =  l1 * np.sin(θ1)
    y1 = -l1 * np.cos(θ1)
    x2 = x1 + l2 * np.sin(θ2)
    y2 = y1 - l2 * np.cos(θ2)

    loom = Loom(
        "double_pendulum.mp4",
        fps = 60,
        overwrite = True,
        savefig_kwargs = {"bbox_inches": "tight"}
    )

    with loom:
        for i, t in tqdm(enumerate(times), total=len(times)):
            fig, ax = plt.subplots(figsize=(8, 8))
            
            ax.plot(
                [0, x1[i], x2[i]],
                [0, y1[i], y2[i]],
                linestyle = "solid",
                marker = "o",
                color = "black",
                linewidth = 3
            )

            ax.plot(
                x2[:i+1],
                y2[:i+1],
                linestyle = "solid",
                linewidth = 2,
                color = "red",
                alpha = 0.5
            )
            
            ax.set_title(f"Double Pendulum: t = {t:.3f}s")
            
            ax.set_xlim(-2.2, 2.2)
            ax.set_ylim(-2.2, 2.2)
            ax.set_aspect("equal", adjustable="box")

            loom.save_frame(fig)

    assert Path("double_pendulum.mp4").is_file()
    assert Path("double_pendulum.mp4").stat().st_size > 0

def test_night_time_shading():
    def plot_frame(day_of_year, loom, frame_number):
        date = datetime.datetime(2024, 1, 1, 12) + datetime.timedelta(days=day_of_year-1)

        fig = plt.figure(figsize=(15, 5))

        proj1 = ccrs.Orthographic(central_longitude=0, central_latitude=30)
        proj2 = ccrs.Orthographic(central_longitude=120, central_latitude=0)
        proj3 = ccrs.Orthographic(central_longitude=240, central_latitude=-30)

        ax1 = fig.add_subplot(1, 3, 1, projection=proj1)
        ax2 = fig.add_subplot(1, 3, 2, projection=proj2)
        ax3 = fig.add_subplot(1, 3, 3, projection=proj3)

        fig.suptitle(f"Night time shading for {date} UTC")
        
        ax1.stock_img()
        ax1.add_feature(Nightshade(date, alpha=0.2))

        ax2.stock_img()
        ax2.add_feature(Nightshade(date, alpha=0.2))

        ax3.stock_img()
        ax3.add_feature(Nightshade(date, alpha=0.2))

        loom.save_frame(fig, frame_number)

    loom = Loom(
        "night_time_shading.mp4",
        fps = 10,
        verbose = True,
        overwrite = True,
        parallel = True
    )

    with loom:
        n_days_to_test = 5
        days_of_year = range(1, n_days_to_test + 1)
        
        Parallel(n_jobs=-1)(
            delayed(plot_frame)(day_of_year, loom, i)
            for i, day_of_year in enumerate(days_of_year)
    )

    assert Path("night_time_shading.mp4").is_file()
    assert Path("night_time_shading.mp4").stat().st_size > 0
