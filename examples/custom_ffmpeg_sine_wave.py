import numpy as np
import matplotlib.pyplot as plt

from joblib import Parallel, delayed

# Either the environ var OR matplotlib rcPram *needs* to be set BEFORE
#  importing matplotloom
import imageio_ffmpeg # python library that provides ffmpeg binary 
FFMPEG_PATH: str = imageio_ffmpeg.get_ffmpeg_exe()

# Configure matplotlib rcPrams, if your global / project rcPrams file
#  already sets this then you don't need to overload it. 
plt.rcParams['animation.ffmpeg_path'] = FFMPEG_PATH

# Alternatively could set the environment variable however this *does not*
#  inform matplotlib there is a valid ffmpeg binary so should be avoided.
#  The intention is to allow any more complex toolchains the option if needed.
# import os
#os.environ["LOOM_FFMPEG_PATH"] = FFMPEG_PATH

from matplotloom import Loom

def plot_frame(phase, frame_number, loom):
    fig, ax = plt.subplots()

    x = np.linspace(0, 2*np.pi, 200)
    y = np.sin(x + phase)
    
    ax.plot(x, y)
    ax.set_xlim(0, 2*np.pi)
    
    loom.save_frame(fig, frame_number)

with Loom("custom_parallel_sine_wave.gif", fps=30, parallel=True) as loom:
    phases = np.linspace(0, 2*np.pi, 100)
    
    Parallel(n_jobs=-1)(
        delayed(plot_frame)(phase, i, loom) 
        for i, phase in enumerate(phases)
    )

