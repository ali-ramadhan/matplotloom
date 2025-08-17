import datetime
from dataclasses import dataclass, field

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs


from cmocean import cm
from cartopy.feature.nightshade import Nightshade
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.integrate import solve_ivp
from scipy.special import j0
from tqdm import tqdm


from matplotloom import Loom

def test_sine_wave(test_output_dir):
    with Loom(test_output_dir / "test_sine_wave.gif", fps=30) as loom:
        for phase in np.linspace(0, 2*np.pi, 100):
            fig, ax = plt.subplots()

            x = np.linspace(0, 2*np.pi, 200)
            y = np.sin(x + phase)

            ax.plot(x, y)
            ax.set_xlim(0, 2*np.pi)

            loom.save_frame(fig)

    assert (test_output_dir / "test_sine_wave.gif").is_file()
    assert (test_output_dir / "test_sine_wave.gif").stat().st_size > 0

def test_parallel_sine_wave(test_output_dir):
    def plot_frame(phase, frame_number, loom):
        fig, ax = plt.subplots()

        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x + phase)

        ax.plot(x, y)
        ax.set_xlim(0, 2*np.pi)

        loom.save_frame(fig, frame_number)

    with Loom(test_output_dir / "test_parallel_sine_wave.gif", fps=30, parallel=True) as loom:
        phases = np.linspace(0, 2*np.pi, 10)

        Parallel(n_jobs=-1)(
            delayed(plot_frame)(phase, i, loom)
            for i, phase in enumerate(phases)
        )

    assert (test_output_dir / "test_parallel_sine_wave.gif").is_file()
    assert (test_output_dir / "test_parallel_sine_wave.gif").stat().st_size > 0

def test_rotating_circular_sine_wave(test_output_dir):
    with Loom(test_output_dir / "test_rotating_circular_sine_wave.mp4", fps=10) as loom:
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

    assert (test_output_dir / "test_rotating_circular_sine_wave.mp4").is_file()
    assert (test_output_dir / "test_rotating_circular_sine_wave.mp4").stat().st_size > 0

def test_bessel_wave(test_output_dir):
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
        test_output_dir / "test_bessel_wave.mp4",
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

    assert (test_output_dir / "test_bessel_wave.mp4").is_file()
    assert (test_output_dir / "test_bessel_wave.mp4").stat().st_size > 0

def test_double_pendulum(test_output_dir):
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
        test_output_dir / "test_double_pendulum.mp4",
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

    assert (test_output_dir / "test_double_pendulum.mp4").is_file()
    assert (test_output_dir / "test_double_pendulum.mp4").stat().st_size > 0

def test_night_time_shading(test_output_dir):
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
        test_output_dir / "test_night_time_shading.mp4",
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

    assert (test_output_dir / "test_night_time_shading.mp4").is_file()
    assert (test_output_dir / "test_night_time_shading.mp4").stat().st_size > 0

def test_lorenz(test_output_dir):
    @dataclass
    class Lorenz:
        dt: float = 0.01
        sigma: float = 10.0
        rho: float = 28.0
        beta: float = 8.0 / 3.0
        x: float = 1.0
        y: float = 1.0
        z: float = 1.0

        def step(self):
            dx = self.sigma * (self.y - self.x)
            dy = self.x * (self.rho - self.z) - self.y
            dz = self.x * self.y - self.beta * self.z
            self.x += dx * self.dt
            self.y += dy * self.dt
            self.z += dz * self.dt

        @property
        def position(self) -> tuple[float, float, float]:
            return self.x, self.y, self.z

    @dataclass
    class LorenzPlotter:
        steps_per_frame: int = 20
        attractor = Lorenz()
        points: list[tuple[float, float, float]] = field(default_factory=list)

        def initialize(self, steps: int):
            self.points = [self.attractor.position]
            for _ in range(steps):
                self.attractor.step()
                self.points.append(self.attractor.position)

        @property
        def frames(self) -> list[int]:
            return list(range(1, len(self.points) // self.steps_per_frame))

        def get_frame(self, i: int, loom: Loom):
            fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': '3d'})
            points = np.array(self.points[: i * self.steps_per_frame])
            xs, ys, zs = points.T
            segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
            norm = Normalize(vmin=0, vmax=len(xs))
            colors = plt.get_cmap('inferno')(norm(np.arange(len(xs) - 1)))
            lc = Line3DCollection(segments, colors=colors, linewidth=0.5)
            ax.add_collection3d(lc)
            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.set_zlim(0, 50)
            ax.view_init(
                azim=(np.pi * 1.7 + 0.8 * np.sin(2.0 * np.pi * i * self.steps_per_frame / len(self.frames) / 10))
                * 180.0
                / np.pi
            )
            ax.set_axis_off()
            ax.grid(visible=False)
            loom.save_frame(fig, i - 1)

    with Loom(test_output_dir / 'test_lorenz.mp4', fps=60, parallel=True) as loom:
        attractor = LorenzPlotter()
        attractor.initialize(1000)
        Parallel(n_jobs=-1)(delayed(attractor.get_frame)(i, loom) for i in attractor.frames[:5])

    assert (test_output_dir / "test_lorenz.mp4").is_file()
    assert (test_output_dir / "test_lorenz.mp4").stat().st_size > 0
