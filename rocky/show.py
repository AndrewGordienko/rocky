# rocky/show.py

import matplotlib.pyplot as plt
import numpy as np

from .core.visualizer import show_track, show_racing_line, show_car

def show(*, track=False, line=False, car=False, coords=None, index=0, block=True):
    if coords is None:
        raise ValueError("coords must be provided")

    if track:
        show_track(coords)

    if line:
        show_racing_line(coords)

    if car:
        show_car(coords, index=index)

    # Final blocking show (default)
    if block:
        plt.show()
