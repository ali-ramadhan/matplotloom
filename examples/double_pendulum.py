import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from tqdm import tqdm
from matplotloom import Loom

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

t_span = (0, 20)
y0 = [np.pi/2, 0, np.pi/2, 0]
sol = solve_ivp(derivatives, t_span, y0, dense_output=True)

times = np.linspace(t_span[0], t_span[1], 1000)

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
