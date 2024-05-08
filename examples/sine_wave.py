import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

with Loom("sine_wave.gif", fps=30) as loom:
    for phase in np.linspace(0, 2*np.pi, 100):
        fig, ax = plt.subplots()

        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x + phase)
        
        ax.plot(x, y)
        ax.set_xlim(0, 2*np.pi)
        
        loom.save_frame(fig)
