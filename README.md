# rocky
A lightweight Python library for F1-style track visualization, geometry computation, and tooling for racing-line research.

## Features
- Track visualization
- Geometry utilities
- Auto-zoom rendering
- Modular design

## Installation
```bash
pip install -e .
```

## Usage
```python
import numpy as np
from rocky.visualizer import show_track

coords = np.load("tracks/albert_park_circuit.npy")
show_track(coords)
```

## Structure
```
rocky/
    pyproject.toml
    README.md
    rocky/
        __init__.py
        config.py
        geometry.py
        visualizer.py
    tracks/
        albert_park_circuit.npy
    examples/
        example_show_track.py
```

## License
MIT
