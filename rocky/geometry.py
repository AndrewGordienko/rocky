# This file contains math utilities: tangent, normals, walls.

import numpy as np
from .config import DISPLAY_WIDTH

def compute_tangent_normals(points):
    tang = np.diff(points, axis=0, prepend=points[0:1])
    n = np.linalg.norm(tang, axis=1)
    n[n == 0] = 1e-6
    tang = tang / n[:, None]
    norms = np.column_stack((-tang[:,1], tang[:,0]))
    return tang, norms

def compute_walls(coords):
    tang, norms = compute_tangent_normals(coords)
    left  = coords + norms*(DISPLAY_WIDTH/2)
    right = coords - norms*(DISPLAY_WIDTH/2)
    return left, right
