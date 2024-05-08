# matplotloom

Weave your frames into easy matplotlib animations.

## Examples

### Sine wave

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

with Loom("sine_wave_animation.gif", fps=30) as loom:
    for phase in np.linspace(0, 2*np.pi, 100):
        fig, ax = plt.subplots()

        x = np.linspace(0, 2*np.pi, 200)
        y = np.sin(x + phase)
        
        ax.plot(x, y)
        ax.set_xlim(0, 2*np.pi)
        
        loom.save_frame(fig)
```

### Rotating circular sine wave

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotloom import Loom

with Loom("3d_rotate.gif", fps=10) as loom:
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

```
