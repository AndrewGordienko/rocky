import numpy as np
import os
import random

def load_random_track(tracks_dir="tracks"):
    """
    Picks a random .npy track file from the given directory and loads it.
    """
    # list npy files in folder
    files = [f for f in os.listdir(tracks_dir) if f.endswith(".npy")]
    
    if not files:
        raise FileNotFoundError(f"No .npy track files found in {tracks_dir}")

    # choose random file
    choice = random.choice(files)
    path = os.path.join(tracks_dir, choice)

    # load coords
    coords = np.load(path).astype(float)
    return coords, choice
