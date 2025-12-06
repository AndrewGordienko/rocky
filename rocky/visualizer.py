# This handles only plotting

import numpy as np
import matplotlib.pyplot as plt
from .config import *
from .geometry import compute_walls

def show_track(coords):
    left, right = compute_walls(coords)
    track_outline = np.vstack((left, right[::-1]))

    xmin,xmax = track_outline[:,0].min(), track_outline[:,0].max()
    ymin,ymax = track_outline[:,1].min(), track_outline[:,1].max()
    dx,dy = xmax-xmin, ymax-ymin
    pad_x,pad_y = dx*AUTO_PAD_RATIO, dy*AUTO_PAD_RATIO

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="black")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])

    ax.fill(track_outline[:,0], track_outline[:,1], color="#3c3c3c")
    ax.plot(left[:,0], left[:,1], color="white", linewidth=WALL_LINEWIDTH)
    ax.plot(right[:,0], right[:,1], color="white", linewidth=WALL_LINEWIDTH)

    ax.scatter(coords[0,0], coords[0,1], color="lime", s=60)

    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_facecolor("black")
    ax.set_aspect("equal")

    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title("Track", color="white")

    plt.show()
