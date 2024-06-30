import numpy as np
import matplotlib.pyplot as plt
from scipy.special import j0
from cmocean import cm
from matplotloom import Loom

def bessel_wave(r, t, k, omega, A):
    return A * j0(k*r - omega*t)

def create_frame(x, y, t):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    r = np.sqrt(x**2 + y**2)
    z = bessel_wave(r, t, k=2, omega=1, A=1)
    
    pcm = ax1.pcolormesh(x, y, z, cmap=cm.balance, shading='auto', vmin=-1, vmax=1)
    fig.colorbar(pcm, ax=ax1)

    ax1.set_title(f"Bessel wave: t = {t:.3f}")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)
    ax1.set_aspect("equal", adjustable="box")
    
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
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    x, y = np.meshgrid(x, y)

    for t in np.linspace(0, 50, 300):
        fig = create_frame(x, y, t)
        loom.save_frame(fig)
