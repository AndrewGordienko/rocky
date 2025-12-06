# This handles only plotting

from matplotlib.collections import LineCollection
from ..core.geometry import compute_walls
from ..core.racing_line import compute_racing_line_with_speed
from ..util.config import DISPLAY_WIDTH, FIGURE_SIZE, AUTO_PAD_RATIO, WALL_LINEWIDTH
import numpy as np
import matplotlib.pyplot as plt

def _prepare_track_figure(coords):
    left, right = compute_walls(coords)
    outline = np.vstack((left, right[::-1]))

    # auto-zoom
    xmin, xmax = outline[:,0].min(), outline[:,0].max()
    ymin, ymax = outline[:,1].min(), outline[:,1].max()
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = dx * AUTO_PAD_RATIO, dy * AUTO_PAD_RATIO

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="black")
    ax = fig.add_axes([0.05, 0.05, 0.90, 0.90])

    # draw asphalt + walls
    ax.fill(outline[:,0], outline[:,1], color="#3c3c3c")
    ax.plot(left[:,0], left[:,1], color="white", linewidth=WALL_LINEWIDTH)
    ax.plot(right[:,0], right[:,1], color="white", linewidth=WALL_LINEWIDTH)

    # start marker
    ax.scatter(coords[0,0], coords[0,1], color="lime", s=60)

    # formatting
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_visible(False)

    return fig, ax, left, right, outline

def show_track(coords):
    fig, ax, left, right, outline = _prepare_track_figure(coords)
    ax.set_title("Track", color="white")
    plt.show()

def show_racing_line(coords):
    fig, ax, left, right, outline = _prepare_track_figure(coords)

    racing, speeds, _, _ = compute_racing_line_with_speed(coords)
    segments = np.stack([racing, np.roll(racing, -1, axis=0)], axis=1)

    lc = LineCollection(segments, array=speeds, cmap="viridis", linewidth=2.0)
    ax.add_collection(lc)

    ax.set_title("Racing Line", color="white")
    plt.show()