import numpy as np
import rocky as ry

"""
Example: Running multiple cars with custom CarModel profiles.

CarModel parameters:
    accel        - maximum forward acceleration (m/s² approx)
    brake        - maximum deceleration
    grip         - affects cornering speed / stability
    speed_scale  - scales the target speed along the racing line
    name         - optional identifier for clarity/debugging
"""

# Load a random track
coords, name = ry.load_random_track()
print(f"Loaded track: {name}")

# Define car profiles
attacker = ry.CarModel(
    accel=13.0,
    brake=18.0,
    grip=1.70,
    speed_scale=1.00,
    name="attacker"
)

defender = ry.CarModel(
    accel=11.0,
    brake=16.0,
    grip=1.55,
    speed_scale=0.85,
    name="defender"
)

car_models = [attacker, defender]

# Optional: draw the racing line during animation
ry.show_racing_line = True

# Run animation
ry.show_car_on_track(coords, car_models=car_models)
