# This handles only plotting

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon, Rectangle

from ..core.geometry import compute_walls, compute_tangent_normals
from ..core.racing_line import compute_racing_line_with_speed
from ..util.config import (
    DISPLAY_WIDTH,
    FIGURE_SIZE,
    AUTO_PAD_RATIO,
    WALL_LINEWIDTH,
    CAR_LENGTH,
    CAR_WIDTH,
    MAG_WIDTH,
    MAG_HEIGHT,
)

def _prepare_track_figure(coords, main_rect=(0.05, 0.05, 0.90, 0.90)):
    left, right = compute_walls(coords)
    outline = np.vstack((left, right[::-1]))

    xmin, xmax = outline[:,0].min(), outline[:,0].max()
    ymin, ymax = outline[:,1].min(), outline[:,1].max()
    dx, dy = xmax - xmin, ymax - ymin
    pad_x, pad_y = dx*AUTO_PAD_RATIO, dy*AUTO_PAD_RATIO

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="black")
    ax = fig.add_axes(list(main_rect))

    ax.fill(outline[:,0], outline[:,1], color="#3c3c3c")
    ax.plot(left[:,0],  left[:,1],  color="white", linewidth=WALL_LINEWIDTH)
    ax.plot(right[:,0], right[:,1], color="white", linewidth=WALL_LINEWIDTH)
    ax.scatter(coords[0,0], coords[0,1], color="lime", s=60)

    ax.set_xlim(xmin-pad_x, xmax+pad_x)
    ax.set_ylim(ymin-pad_y, ymax+pad_y)
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_visible(False)

    return fig, ax, left, right, outline

def show_track(coords):
    fig, ax, _, _, _ = _prepare_track_figure(coords)
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

def show_car_on_track(coords, magnifier=True):
    # use left-side layout so magnifier doesn't overlap
    fig, ax, left, right, outline = _prepare_track_figure(
        coords, main_rect=(0.05, 0.05, 0.55, 0.90)
    )

    racing, speeds, _, _ = compute_racing_line_with_speed(coords)
    tang, norms = compute_tangent_normals(racing)

    seg = np.stack([racing, np.roll(racing, -1, axis=0)], axis=1)
    lc = LineCollection(seg, array=speeds, cmap="viridis", linewidth=2.0)
    ax.add_collection(lc)

    car_poly = Polygon([[0,0],[0,0],[0,0]], closed=True,
                       color="red", zorder=200)
    ax.add_patch(car_poly)

    if magnifier:
        mag_rect = Rectangle((0,0), MAG_WIDTH, MAG_HEIGHT,
                             edgecolor="white", facecolor="none",
                             linewidth=0.8, zorder=180)
        ax.add_patch(mag_rect)

        axz = fig.add_axes([0.70, 0.20, 0.28, 0.28])
        axz.set_facecolor("black")
        axz.set_aspect("equal")
        axz.tick_params(left=False, bottom=False,
                        labelleft=False, labelbottom=False)
        for s in axz.spines.values():
            s.set_color("white")
            s.set_linewidth(0.7)

        axz.fill(outline[:,0], outline[:,1], color="#3c3c3c")
        axz.plot(left[:,0],  left[:,1],  color="white", linewidth=0.7)
        axz.plot(right[:,0], right[:,1], color="white", linewidth=0.7)
        axz.add_collection(LineCollection(seg, array=speeds,
                                          cmap="viridis", linewidth=1.2))

        zoom_car = Polygon([[0,0],[0,0],[0,0]], closed=True,
                           color="red", zorder=200)
        axz.add_patch(zoom_car)

    N = len(racing)
    s_pos = 0.0
    BASE_INDEX_STEP = 1.0     # a bit faster so movement is visible

    def update(frame):
        nonlocal s_pos

        idx = int(s_pos) % N
        pos = racing[idx]
        d   = tang[idx]
        n   = norms[idx]

        tri = [
            pos + d*CAR_LENGTH*0.6,
            pos - d*CAR_LENGTH*0.4 + n*CAR_WIDTH,
            pos - d*CAR_LENGTH*0.4 - n*CAR_WIDTH
        ]

        car_poly.set_xy(tri)

        if magnifier:
            zoom_car.set_xy(tri)
            mag_rect.set_xy((pos[0]-MAG_WIDTH/2, pos[1]-MAG_HEIGHT/2))

            cx, cy = pos
            axz.set_xlim(cx - MAG_WIDTH/2, cx + MAG_WIDTH/2)
            axz.set_ylim(cy - MAG_HEIGHT/2, cy + MAG_HEIGHT/2)

        v_norm = max(speeds[idx] / speeds.max(), 0.01)
        s_pos = (s_pos + v_norm * BASE_INDEX_STEP) % N

        return []

    ani = FuncAnimation(fig, update, frames=20000, interval=30, blit=False)

    ax.set_title("Car Animation", color="white")
    plt.show()
