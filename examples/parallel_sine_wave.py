import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from matplotloom import Loom

def plot_frame(phase, frame_number, loom):
    fig, ax = plt.subplots()

    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x + phase)
    
    ax.plot(x, y)
    ax.set_xlim(0, 2*np.pi)
    
    loom.save_frame(fig, frame_number)

with Loom("parallel_sine_wave.gif", fps=30, parallel=True) as loom:
    phases = np.linspace(0, 2*np.pi, 100)
    
    Parallel(n_jobs=-1)(
        delayed(plot_frame)(phase, i, loom) 
        for i, phase in enumerate(phases)
    )
