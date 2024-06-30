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
