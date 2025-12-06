import numpy as np
from ..util.config import DISPLAY_WIDTH

def compute_tangent_normals(points):
    tang = np.diff(points, axis=0, prepend=points[0:1])
    n = np.linalg.norm(tang, axis=1); n[n == 0] = 1e-6
    tang = tang / n[:,None]
    norms = np.column_stack((-tang[:,1], tang[:,0]))
    return tang, norms

def compute_walls(coords):
    tang, norms = compute_tangent_normals(coords)
    left  = coords + norms*(DISPLAY_WIDTH/2)
    right = coords - norms*(DISPLAY_WIDTH/2)
    return left, right

def curvature(points, tangents=None):
    if tangents is None:
        tangents, _ = compute_tangent_normals(points)
    t_next = np.roll(tangents, -1, axis=0)
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    ds[ds == 0] = 1e-6
    cross = tangents[:,0]*t_next[:,1] - tangents[:,1]*t_next[:,0]
    return cross/ds, ds

def smooth_path(p, iters=3):
    p = p.copy()
    for _ in range(iters):
        p = 0.25*np.roll(p,1,0) + 0.5*p + 0.25*np.roll(p,-1,0)
    return p
