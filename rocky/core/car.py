# rocky/core/car.py

class CarModel:
    """
    Defines the performance profile of a car.
    All physics and overtake behaviour will reference these parameters.

    Parameters
    ----------
    accel : float
        Maximum forward acceleration (m/s^2 approx).
    brake : float
        Maximum deceleration when braking.
    grip : float
        Lateral grip coefficient; affects cornering speed and stability.
    speed_scale : float
        Scales the target speed along the racing line.
    name : str
        Optional identifier for the model.
    """

    def __init__(self,
                 accel=12.0,
                 brake=18.0,
                 grip=1.70,
                 speed_scale=1.00,
                 name="car"):
        self.accel = accel
        self.brake = brake
        self.grip = grip
        self.speed_scale = speed_scale
        self.name = name

    def __repr__(self):
        return f"CarModel(name={self.name}, accel={self.accel}, brake={self.brake}, grip={self.grip}, scale={self.speed_scale})"