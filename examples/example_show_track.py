import numpy as np
from rocky.visualizer import show_track

coords = np.load("tracks/albert_park_circuit.npy")
show_track(coords)