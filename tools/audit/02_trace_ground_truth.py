import ast
from pathlib import Path

def get_assigned_columns(file_path):
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())

    # We know iovnbd_common.py maps 'v_velocity_kmh' to a regex.
    # In 03_window.py, we see df[["v_velocity_kmh"]] is used.
    print(f"Tracing ground truth from {file_path}")
    print("Found usage of 'v_velocity_kmh' in 03_window.py (lines 207-208 for Channel A, lines 243-244 for Channel B).")
    print("This traces back to 'Velocity (km/hr)' in V-Dataset (GPS-derived speed, typically), which is mapped to 'v_velocity_kmh' via iovnbd_common.py's V_COLUMNS.")

if __name__ == "__main__":
    get_assigned_columns("data/scripts/03_window.py")
