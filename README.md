# rocky

**rocky** is a lightweight Python toolkit for F1-style track visualization, racing-line computation, geometric utilities, and animated car playback.  
It is built for research, teaching, and experimentation with track layouts, racing lines, and dynamic visualizations.

---

## Key Features

### Track Visualization
- Plot real or synthetic circuits
- Auto-padding, auto-scaling
- Clean dark-mode presentation

### Racing Line Tools
- Compute a smoothed racing line
- Speed-colored LineCollection
- Tangent + normal vectors (for car orientation)

### Car Animation
- Single-car or multi-car animation
- Per-car zoom windows (magnifiers)
- Dynamic camera rectangles following each vehicle

### Geometry Utilities
- Left/right wall extraction
- Tangent/normal computation
- Outline and track-width reconstruction

---

## Installation

```bash
pip install -e .
```

After installation:

```python
import rocky as ry
```

---

## Quickstart

### 1. Load a random track

```python
coords, name = ry.load_random_track()
print("Loaded:", name)
```

### 2. Show the track

```python
ry.show_track(coords)
```

### 3. Plot the racing line

```python
ry.show_racing_line(coords)
```

### 4. Animate a car on the track

```python
ry.show_car_on_track(coords)             # 1 car
ry.show_car_on_track(coords, n=3)        # 3 cars
```

Each car receives:
- Its own orientation (via tangent vectors)
- Its own magnifier window
- Its own world-space zoom rectangle

---

## Example

The `examples/` directory contains minimal usage patterns.

```python
import numpy as np
import rocky as ry

coords, name = ry.load_random_track()
print(f"Track loaded: {name}")

ry.show_car_on_track(coords)
```

---

## Contributing

Pull requests are welcome.

---

## License

MIT License
