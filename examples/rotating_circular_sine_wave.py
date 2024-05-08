import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

with Loom("rotating_circular_sine_wave.mp4", fps=10) as loom:
    for i in range(36):
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
