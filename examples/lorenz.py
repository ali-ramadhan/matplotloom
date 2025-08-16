# Inspired by <https://docs.makie.org/stable/>
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib.colors import Normalize
from matplotloom import Loom
from mpl_toolkits.mplot3d.art3d import Line3DCollection


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


with Loom('lorenz.mp4', fps=60, parallel=True, overwrite=True) as loom:
    attractor = LorenzPlotter()
    attractor.initialize(10000)
    Parallel(n_jobs=-1)(delayed(attractor.get_frame)(i, loom) for i in attractor.frames)
