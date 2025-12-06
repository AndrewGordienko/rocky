import numpy as np
from ..core.geometry import compute_walls, curvature, compute_tangent_normals, smooth_path
from ..util.config import DISPLAY_WIDTH

def compute_racing_line(coords, left, right, w, outer_iters=10, smooth_iters=5, gain=0.45):
    racing = coords.copy()
    for _ in range(outer_iters):
        kappa, _ = curvature(racing)
        tang, norms = compute_tangent_normals(racing)

        shift = -np.sign(kappa)*np.abs(kappa)**0.35
        shift *= gain*(w/2*0.9)
        racing = racing + norms*shift[:,None]

        vw = right - left
        vv = np.sum(vw*vw,1); vv[vv == 0] = 1e-9
        wv = racing - left
        t = np.sum(wv*vw,1)/vv
        t = np.clip(t,0.25,0.75)
        racing = left + vw*t[:,None]

        racing = smooth_path(racing, smooth_iters)

    return racing

def compute_speed_profile(points, accel=12.0, brake=18.0, mu=1.7, vmax=120.0):
    kappa, ds = curvature(points)
    G = 9.81

    v_lat = np.full(len(points), vmax)
    m = np.abs(kappa) > 1e-6
    v_lat[m] = np.sqrt(mu*G/np.abs(kappa[m]))
    v_lat = np.minimum(v_lat, vmax)

    v = np.zeros(len(points))
    v[0] = min(10.0, v_lat[0])

    for i in range(len(points)-1):
        v[i+1] = min(np.sqrt(v[i]**2 + 2*accel*ds[i]), v_lat[i+1])

    for i in range(len(points)-2, -1, -1):
        v[i] = min(v[i], np.sqrt(v[i+1]**2 + 2*brake*ds[i]), v_lat[i])

    return v

def compute_racing_line_with_speed(coords):
    left, right = compute_walls(coords)
    r = compute_racing_line(coords, left, right, DISPLAY_WIDTH)
    s = compute_speed_profile(r)
    return r, s, left, right
