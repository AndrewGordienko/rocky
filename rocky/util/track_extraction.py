import os
import fastf1
import matplotlib.pyplot as plt
import numpy as np

# Make sure folders exist
os.makedirs('cache', exist_ok=True)
os.makedirs('tracks', exist_ok=True)

fastf1.Cache.enable_cache('cache')

# 2024 calendar (Race name -> Circuit name)
races_2024 = [
    ("Australian Grand Prix", "Albert Park Circuit"),
    ("Chinese Grand Prix", "Shanghai International Circuit"),
    ("Japanese Grand Prix", "Suzuka Circuit"),
    ("Bahrain Grand Prix", "Bahrain International Circuit"),
    ("Saudi Arabian Grand Prix", "Jeddah Corniche Circuit"),
    ("Miami Grand Prix", "Miami International Autodrome"),
    ("Emilia Romagna Grand Prix", "Autodromo Enzo e Dino Ferrari"),
    ("Monaco Grand Prix", "Circuit de Monaco"),
    ("Spanish Grand Prix", "Circuit de Barcelona-Catalunya"),
    ("Canadian Grand Prix", "Circuit Gilles Villeneuve"),
    ("Austrian Grand Prix", "Red Bull Ring"),
    ("British Grand Prix", "Silverstone Circuit"),
    ("Belgian Grand Prix", "Circuit de Spa-Francorchamps"),
    ("Hungarian Grand Prix", "Hungaroring"),
    ("Dutch Grand Prix", "Circuit Zandvoort"),
    ("Italian Grand Prix", "Autodromo Nazionale Monza"),
    ("Azerbaijan Grand Prix", "Baku City Circuit"),
    ("Singapore Grand Prix", "Marina Bay Street Circuit"),
    ("United States Grand Prix", "Circuit of the Americas"),
    ("Mexican Grand Prix", "Autódromo Hermanos Rodríguez"),
    ("São Paulo Grand Prix", "Autódromo José Carlos Pace"),
    ("Las Vegas Grand Prix", "Las Vegas Street Circuit"),
    ("Qatar Grand Prix", "Losail International Circuit"),
    ("Abu Dhabi Grand Prix", "Yas Marina Circuit"),
]

# Grid settings
n = len(races_2024)
cols = 6
rows = (n + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(20, 12))
axes = axes.flatten()

for i, (race, circuit) in enumerate(races_2024):
    ax = axes[i]

    try:
        print(f"Loading {race}...")

        session = fastf1.get_session(2024, race, 'R')
        session.load()

        lap = session.laps.pick_fastest()
        pos = lap.get_pos_data()

        x = pos['X']
        y = pos['Y']

        ax.plot(x, y, linewidth=1)
        ax.set_title(circuit, fontsize=8)
        ax.axis('equal')
        ax.axis('off')

        coords = np.column_stack((x, y))
        filename = circuit.replace(" ", "_").lower()
        np.save(f"tracks/{filename}.npy", coords)

    except Exception as e:
        print(f"Failed: {race} → {e}")
        ax.set_title(f"{circuit}\nNo data", fontsize=8)
        ax.axis('off')

# Hide unused plots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.suptitle("2024 Formula 1 Tracks (via FastF1 Telemetry)", fontsize=16)
plt.tight_layout()
plt.show()
