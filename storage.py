import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon, Rectangle
from matplotlib.animation import FuncAnimation

# ===== VISUAL SETTINGS =====
DISPLAY_WIDTH   = 200
AUTO_PAD_RATIO  = 0.02
FIGURE_SIZE     = (16, 9)   # wide enough for zoom windows on the right
WALL_LINEWIDTH  = 0.4

# ---- Car visual size ----
CAR_LENGTH = DISPLAY_WIDTH * 0.20
CAR_WIDTH  = DISPLAY_WIDTH * 0.10

# ---- Magnifier window size (world coordinates) ----
MAG_WIDTH  = DISPLAY_WIDTH * 4.0
MAG_HEIGHT = DISPLAY_WIDTH * 2.5

# ---- 2 Car Params (slow vs fast) ----
CAR_PARAMS = [
    {"color": "red",   "accel": 12.0, "brake": 18.0, "grip": 1.70, "speed_scale": 0.45},  # very slow
    {"color": "cyan",  "accel": 12.5, "brake": 19.0, "grip": 1.75, "speed_scale": 1.00},  # fast
]
NUM_CARS = 2

# Movement scale (indices per frame at v=120 and speed_scale=1)
BASE_INDEX_STEP = 0.9

# Overtake parameters
GAP_TRIGGER_IDX   = 40.0    # start overtake if within this many indices
GAP_CLEAR_IDX     = 10.0    # consider pass complete once ahead by this much
OVERTAKE_OFFSET   = DISPLAY_WIDTH * 0.25  # lateral distance for overtake line
RETURN_BLEND_RATE = 0.04    # how fast car blends back to racing line

# ==============================

coords = np.load("tracks/albert_park_circuit.npy").astype(float)
N = len(coords)

# ---------- Utility Functions ----------

def compute_tangent_normals(points):
    tang = np.diff(points, axis=0, prepend=points[0:1])
    n = np.linalg.norm(tang, axis=1)
    n[n == 0] = 1e-6
    tang = tang / n[:, None]
    norms = np.column_stack((-tang[:,1], tang[:,0]))
    return tang, norms

def curvature(points):
    tang, _ = compute_tangent_normals(points)
    t_next = np.roll(tang, -1, axis=0)
    ds = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    ds[ds == 0] = 1e-6
    cross = tang[:,0]*t_next[:,1] - tang[:,1]*t_next[:,0]
    return cross / ds, ds

def smooth_path(p, iters=3):
    for _ in range(iters):
        p = 0.25*np.roll(p,1,axis=0) + 0.5*p + 0.25*np.roll(p,-1,axis=0)
    return p

# ---------- Wall Geometry ----------
tang_center, norms_center = compute_tangent_normals(coords)
left  = coords + norms_center*(DISPLAY_WIDTH/2)
right = coords - norms_center*(DISPLAY_WIDTH/2)
track_outline = np.vstack((left, right[::-1]))

# ---------- Racing Line Initialization ----------
racing = coords.copy()

# ---------- Lateral Optimization ----------
LATERAL_GAIN  = 0.45
SMOOTH_ITERS  = 5
OUTER_ITERS   = 10

for _ in range(OUTER_ITERS):
    kappa, _ = curvature(racing)
    tang_tmp, norms_tmp = compute_tangent_normals(racing)

    lateral_shift = -np.sign(kappa)*np.abs(kappa)**0.35
    lateral_shift *= LATERAL_GAIN*(DISPLAY_WIDTH/2*0.9)
    racing = racing + norms_tmp*lateral_shift[:,None]

    # project inside track walls
    vw = right - left
    w  = racing - left
    vv = np.sum(vw*vw,axis=1)
    vv[vv<1e-9] = 1e-9
    t  = np.sum(w*vw,axis=1)/vv
    t  = np.clip(t, 0.25, 0.75)
    racing = left + vw*t[:,None]

    racing = smooth_path(racing, iters=SMOOTH_ITERS)

# final geometry for racing line
tang_race, norms_race = compute_tangent_normals(racing)

# ---------- Speed Profile (base, before per-car modifiers) ----------
def compute_speed_profile(points):
    kappa, ds = curvature(points)
    V_CAP   = 120.0
    MU_TYRE = 1.7
    G       = 9.81
    A_ACCEL = 12.0
    A_BRAKE = 18.0

    v_lat = np.full(N, V_CAP)
    mask = np.abs(kappa) > 1e-6
    v_lat[mask] = np.sqrt(MU_TYRE * G / np.abs(kappa[mask]))
    v_lat = np.minimum(v_lat, V_CAP)

    v_f = np.zeros(N)
    v_f[0] = min(10, v_lat[0])
    for i in range(N - 1):
        v = np.sqrt(v_f[i]**2 + 2 * A_ACCEL * ds[i])
        v_f[i+1] = min(v, v_lat[i+1])

    v_opt = np.copy(v_f)
    for i in range(N-2, -1, -1):
        v = np.sqrt(v_opt[i+1]**2 + 2*A_BRAKE*ds[i])
        v_opt[i] = min(v_opt[i], v, v_lat[i])

    return v_opt

base_speed = compute_speed_profile(racing)  # 0..~120 m/s equivalent

# ---------- Auto Zoom ----------
xmin,xmax = track_outline[:,0].min(), track_outline[:,0].max()
ymin,ymax = track_outline[:,1].min(), track_outline[:,1].max()
dx,dy = xmax-xmin, ymax-ymin
pad_x,pad_y = dx*AUTO_PAD_RATIO, dy*AUTO_PAD_RATIO

# ---------- Figure ----------
fig = plt.figure(figsize=FIGURE_SIZE, facecolor="black")

# main track axes on the left
ax = fig.add_axes([0.05, 0.05, 0.55, 0.90])

ax.fill(track_outline[:,0], track_outline[:,1], color="#3c3c3c")
ax.plot(left[:,0], left[:,1], color="white", linewidth=WALL_LINEWIDTH)
ax.plot(right[:,0], right[:,1], color="white", linewidth=WALL_LINEWIDTH)

segments = np.stack([racing, np.roll(racing,-1,axis=0)], axis=1)
lc = LineCollection(segments, array=base_speed, cmap="viridis", linewidth=2.0)
ax.add_collection(lc)

ax.scatter(coords[0,0],coords[0,1],color="lime",s=60)

ax.set_xlim(xmin-pad_x, xmax+pad_x)
ax.set_ylim(ymin-pad_y, ymax+pad_y)
ax.set_facecolor("black")
ax.set_aspect("equal")
ax.tick_params(colors="white")
for s in ax.spines.values():
    s.set_visible(False)

ax.set_title("Two Cars – Simple F1-Style Overtaking", color="white")

# ---------- Create Car Shapes + Magnifier Boxes ----------
cars       = []
magnifiers = []
zoom_axes  = []
zoom_positions = [0.60, 0.15]  # [bottom] positions for 2 zoom windows

for idx, params in enumerate(CAR_PARAMS):
    car_poly = Polygon([[0,0],[0,0],[0,0]], closed=True, color=params["color"], zorder=200)
    ax.add_patch(car_poly)
    cars.append(car_poly)

    mag = Rectangle((0,0), MAG_WIDTH, MAG_HEIGHT,
                    edgecolor=params["color"], facecolor="none", zorder=150)
    ax.add_patch(mag)
    magnifiers.append(mag)

    axz = fig.add_axes([0.70, zoom_positions[idx], 0.28, 0.25])
    axz.set_facecolor("black")
    axz.set_aspect("equal")
    axz.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for s in axz.spines.values():
        s.set_color("white")
        s.set_linewidth(0.7)
    zoom_axes.append(axz)

# ---- draw static track once in each zoom axis ----
for axz in zoom_axes:
    axz.fill(track_outline[:,0], track_outline[:,1], color="#3c3c3c")
    axz.plot(left[:,0], left[:,1], color="white", linewidth=0.7)
    axz.plot(right[:,0], right[:,1], color="white", linewidth=0.7)
    lc_zoom = LineCollection(segments, array=base_speed, cmap="viridis", linewidth=1.5)
    axz.add_collection(lc_zoom)

# ---------- Overtake line artists (main + zooms) ----------
overtake_line_main, = ax.plot([], [], color="orange",
                              linestyle="--", linewidth=1.5, alpha=0.9)
overtake_zoom_lines = []
for axz in zoom_axes:
    ln, = axz.plot([], [], color="orange",
                   linestyle="--", linewidth=1.2, alpha=0.9)
    overtake_zoom_lines.append(ln)

# ---------- Per-zoom car polygons (each zoom shows both cars) ----------
zoom_car_polys = []  # [zoom_idx][car_idx]
for j, axz in enumerate(zoom_axes):
    row = []
    for k, params in enumerate(CAR_PARAMS):
        cp = Polygon([[0,0],[0,0],[0,0]], closed=True,
                     color=params["color"], zorder=200)
        axz.add_patch(cp)
        row.append(cp)
    zoom_car_polys.append(row)

# ---------- Per-car continuous lap position ----------
s_positions = np.array([0.0, 0.0], dtype=float)

# ---------- Overtake state for each car ----------
# states: "NORMAL", "OVERTAKE", "RETURN"
car_state      = np.array(["NORMAL", "NORMAL"], dtype=object)
car_side       = np.array([0.0, 0.0])  # -1 right, +1 left
return_alpha   = np.array([0.0, 0.0])  # blend factor for RETURN

# ---------- Helper: choose side to pass (for car1 vs car0) ----------
def choose_overtake_side(idx):
    # more space on left or right from racing line?
    center = racing[idx]
    l = left[idx]
    r = right[idx]
    dist_left  = np.linalg.norm(center - l)
    dist_right = np.linalg.norm(r - center)
    # if more space to the left, pass on left (side +1), else right (-1)
    return 1.0 if dist_left > dist_right else -1.0

# ---------- Helper: compute full overtake line for a given side ----------
def compute_overtake_line(side):
    # side = +1 (left) or -1 (right), offset then project between walls
    line = np.empty_like(racing)
    for i in range(N):
        base = racing[i] + side * norms_race[i] * OVERTAKE_OFFSET

        v = right[i] - left[i]
        w = base - left[i]
        vv = np.dot(v, v)
        if vv < 1e-9:
            line[i] = racing[i]
            continue
        t = np.dot(w, v) / vv
        t = np.clip(t, 0.15, 0.85)
        line[i] = left[i] + v * t
    return line

# cache last overtake line to draw it smoothly
current_overtake_line = None

# ---------- Animation Update ----------
def update(frame):
    global current_overtake_line

    # --- 1. State machine for overtaking (car1 vs car0) ---

    # indices and speeds for both cars at current positions
    idx0 = int(s_positions[0]) % N
    idx1 = int(s_positions[1]) % N

    speed0 = base_speed[idx0] * CAR_PARAMS[0]["speed_scale"]
    speed1 = base_speed[idx1] * CAR_PARAMS[1]["speed_scale"]

    # distance from car1 to car0 ahead along the lap
    gap_1_to_0 = s_positions[0] - s_positions[1]
    if gap_1_to_0 < 0:
        gap_1_to_0 += N

    # distance from car0 to car1 ahead along lap (for clear condition)
    gap_0_to_1 = s_positions[1] - s_positions[0]
    if gap_0_to_1 < 0:
        gap_0_to_1 += N

    # trigger overtake only for car1 (fast) when behind car0 (slow)
    if car_state[1] == "NORMAL":
        if 0.0 < gap_1_to_0 < GAP_TRIGGER_IDX and speed1 > speed0 + 1.0:
            car_state[1] = "OVERTAKE"
            car_side[1]  = choose_overtake_side(idx1)
            return_alpha[1] = 0.0
            current_overtake_line = compute_overtake_line(car_side[1])

    elif car_state[1] == "OVERTAKE":
        # once car1 is clearly ahead of car0, start blending back
        if gap_0_to_1 > GAP_CLEAR_IDX:
            car_state[1] = "RETURN"
            return_alpha[1] = 0.0

    elif car_state[1] == "RETURN":
        return_alpha[1] += RETURN_BLEND_RATE
        if return_alpha[1] >= 1.0:
            return_alpha[1] = 1.0
            car_state[1] = "NORMAL"
            current_overtake_line = None  # stop drawing special line

    # --- 2. Build car triangles and move them ---

    triangles_all = []

    for k in range(NUM_CARS):
        params = CAR_PARAMS[k]
        idx = int(s_positions[k]) % N

        # base direction/normal always from main racing line
        d = tang_race[idx]
        n = norms_race[idx]

        # where to place this car
        if k == 1 and car_state[1] in ("OVERTAKE", "RETURN") and current_overtake_line is not None:
            # point on overtake line
            overtake_pos = current_overtake_line[idx]
            if car_state[1] == "OVERTAKE":
                pos = overtake_pos
            else:  # RETURN: blend back toward racing line
                pos = (1.0 - return_alpha[1]) * overtake_pos + return_alpha[1] * racing[idx]
        else:
            pos = racing[idx]

        tri = [
            pos + d*CAR_LENGTH*0.6,
            pos - d*CAR_LENGTH*0.4 + n*CAR_WIDTH,
            pos - d*CAR_LENGTH*0.4 - n*CAR_WIDTH,
        ]
        triangles_all.append((tri, params))

        # update main car shape
        cars[k].set_xy(tri)

        # move magnifier box on track
        magnifiers[k].set_xy((pos[0]-MAG_WIDTH/2, pos[1]-MAG_HEIGHT/2))

        # advance along lap
        v_local = base_speed[idx]           # 0..120
        v_norm  = v_local / 120.0           # 0..1
        ds      = v_norm * params["speed_scale"] * BASE_INDEX_STEP
        s_positions[k] = (s_positions[k] + ds) % N

    # --- 3. Update overtake line drawing (main + zooms) ---

    if current_overtake_line is not None:
        overtake_line_main.set_data(current_overtake_line[:,0],
                                    current_overtake_line[:,1])
        for j, axz in enumerate(zoom_axes):
            overtake_zoom_lines[j].set_data(current_overtake_line[:,0],
                                            current_overtake_line[:,1])
    else:
        overtake_line_main.set_data([], [])
        for ln in overtake_zoom_lines:
            ln.set_data([], [])

    # --- 4. Update zoom windows: each zoom follows its car, but shows both cars ---

    for j in range(NUM_CARS):
        axz = zoom_axes[j]
        nose = np.array(triangles_all[j][0][0])  # nose of car j
        cx, cy = nose

        axz.set_xlim(cx - MAG_WIDTH/2, cx + MAG_WIDTH/2)
        axz.set_ylim(cy - MAG_HEIGHT/2, cy + MAG_HEIGHT/2)

        # update all car polygons in that zoom
        for k in range(NUM_CARS):
            tri_k = triangles_all[k][0]
            zoom_car_polys[j][k].set_xy(tri_k)

    return []

ani = FuncAnimation(fig, update, frames=3000, interval=30, blit=False)

plt.show()