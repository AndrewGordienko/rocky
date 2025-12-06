import numpy as np
import rocky as ry

"""
ry.load_random_track()
ry.show_racing_line(coords)
"""

coords, name = ry.load_random_track()
print("Loaded:", name)

ry.show_racing_line(coords)