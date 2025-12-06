import numpy as np
import rocky as ry

coords = np.load("tracks/albert_park_circuit.npy")
ry.show_racing_line(coords)