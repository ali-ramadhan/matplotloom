Matplotloom |version|
=====================

Matplotloom is a Python library for creating animations from matplotlib figures. It provides a simple interface to save individual frames and compile them into animations (video or GIF) using ffmpeg.

.. contents:: Table of Contents
   :depth: 2
   :local:

Why use matplotloom?
--------------------

To visualize simulation output for computational fluid dynamics I've had to make long animations with complex figures for a long time. The animations consist of thousands of frames and the figures are too complex for ``FuncAnimation`` and ``ArtistAnimation``. This package aims to simplify and massively speed up the process of making these kinds of animations.

* The main idea behind matplotloom is to describe how to generate each frame of your animation from scratch, instead of generating an animation by modifying one existing plot. This simplifies generating animations. See the examples below and how the code inside the ``for`` loops is plain and familiar matplotlib. It also ensures that every feature can be animated and that the generation process can be easily parallelized.
* matplotlib has two tools for making animations: ``FuncAnimation`` and ``ArtistAnimation``. But to use them you have to write your plotting code differently to modify an existing frame. This makes it difficult to go from plotting still figures to making animations. And some features are non-trivial to animate.
* `celluloid <https://github.com/jwkvam/celluloid>`_ is a nice package for making matplotlib animations easily, but as it relies on ``ArtistAnimation`` under the hood it does come with some `limitations <https://github.com/jwkvam/celluloid?tab=readme-ov-file#limitations>`_ such as not being able to animate titles. It also hasn't been maintained since 2018.
* Plotting many frames (hundreds to thousands+) can be slow but with matplotloom you can use a parallel ``Loom`` to plot each frame in parallel, speeding up the animation process significantly especially if you can dedicate many cores to plotting.

Some notes to users
-------------------

* You can use ``loom.show()`` to display animations in Jupyter notebooks.
* Anxious about animation progress? Pass ``verbose=True`` or use `tqdm <https://github.com/tqdm/tqdm>`_ to monitor progress.
* Animations taking too long to make or do you have tons of frames? You can parallelize frame creating by `looming in parallel <#looming-in-parallel>`_.
* You have to call ``loom.save_frame(fig)`` for each frame (see the examples). While the ``Loom`` object can be made to do this automatically it would have to create and own the ``Figure`` instance and I wanted full control over the creation of the ``Figure`` for maximum flexibility.
* matplotloom is going to be slow. But it's flexible and compatible with all of matplotlib! The real speedup comes from parallelizing frame creation, especially if you have a ton of frames to make.

Installation
------------

matplotloom is published on PyPI so you can install matplotloom via ``pip``

.. code-block:: bash

    pip install matplotloom

or `uv`

.. code-block:: bash

    uv add matplotloom

or ``poetry``

.. code-block:: bash

    poetry add matplotloom

or ``conda``

.. code-block:: bash

    conda install matplotloom

matplotloom requires Python 3.10+ and is continuously tested on Linux, Windows, and Mac. Ensure you have ``ffmpeg`` installed so that animations can be generated.

Examples
--------

Sine wave
~~~~~~~~~

.. code-block:: python

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

.. image:: https://raw.githubusercontent.com/ali-ramadhan/matplotloom/main/examples/sine_wave.gif

Rotating circular sine wave
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

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

.. raw:: html

   <video style="width: 100%; height: auto;" controls>
     <source src="https://github.com/ali-ramadhan/matplotloom/assets/20099589/d7a6f5aa-a5b8-4e1f-9287-61aae154255a" type="video/mp4">
   </video>

Bessel wave
~~~~~~~~~~~

Compare with `animatplot <https://github.com/t-makaro/animatplot>`_'s `blocks example <https://github.com/t-makaro/animatplot/blob/master/docs/source/tutorial/blocks.ipynb>`_. With matplotloom you just use regular matplotlib abstractions.

.. code-block:: python

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

.. raw:: html

   <video style="width: 100%; height: auto;" controls>
     <source src="https://github.com/ali-ramadhan/matplotloom/assets/20099589/fdbe9549-1e98-40b4-80ad-fc99f1531e39" type="video/mp4">
   </video>

Double pendulum
~~~~~~~~~~~~~~~

Compare with `matplotlib's double pendulum <https://matplotlib.org/stable/gallery/animation/double_pendulum.html#sphx-glr-gallery-animation-double-pendulum-py>`_ example.

.. code-block:: python

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

.. raw:: html

   <video style="width: 100%; height: auto;" controls>
     <source src="https://github.com/ali-ramadhan/matplotloom/assets/20099589/1224f357-c435-4cd2-abb3-970d8e42eeb3" type="video/mp4">
   </video>

Night time shading
~~~~~~~~~~~~~~~~~~

matplotloom works out of the box with anything that is built on top of matplotlib. Here we're extending a `Cartopy example <https://scitools.org.uk/cartopy/docs/latest/gallery/lines_and_polygons/nightshade.html#sphx-glr-gallery-lines-and-polygons-nightshade-py>`_.

.. code-block:: python

    import datetime

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    from cartopy.feature.nightshade import Nightshade
    from joblib import Parallel, delayed
    from matplotloom import Loom

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
        overwrite = True,
        parallel = True,
        verbose = True,
        savefig_kwargs = {
            "bbox_inches": "tight"
        }
    )

    with loom:
        n_days_2024 = 366
        days_of_year = range(1, n_days_2024 + 1)

        Parallel(n_jobs=-1)(
            delayed(plot_frame)(day_of_year, loom, i)
            for i, day_of_year in enumerate(days_of_year)
        )

.. raw:: html

   <video style="width: 100%; height: auto;" controls>
     <source src="https://github.com/ali-ramadhan/matplotloom/assets/20099589/dad48d56-73ae-4cb2-ba71-d21855c72215" type="video/mp4">
   </video>

Looming in parallel
-------------------

By passing ``parallel=True`` when creating a ``Loom``, you can save frames using ``loom.save_frame(fig, frame_number)`` which allows you to plot and save all your frames in parallel. One easy way to leverage this is by using joblib to parallelize the for loop. For example, here's how you can parallelize the simple sine wave example:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotloom import Loom
    from joblib import Parallel, delayed

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

Reference
---------

.. autoclass:: matplotloom.Loom
   :members:
   :undoc-members:
