import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fusion_core.python_prototype.ukf import DualChannelUkf, FusionConfig, UkfState

def run_synthetic_test():
    print("--- Stage 5A Synthetic UKF Sanity Test ---")
    
    # Configuration
    dt = 0.1
    duration = 60.0
    steps = int(duration / dt)
    speed = 10.0 # m/s
    
    config = FusionConfig(enable_nhc=True, enable_zupt=True)
    config.r_channel_a = 0.01 # Trust speed completely
    
    scenarios = [
        ("Perfect Straight (0 deg)", 0.0),
        ("Constant Heading Error (+10 deg)", np.radians(10)),
        ("Constant Heading Error (-10 deg)", np.radians(-10)),
    ]
    
    for name, yaw_error in scenarios:
        # Initialize facing exactly North (psi = 0)
        # Assuming psi=0 means North, and positive psi means clockwise rotation (East)
        initial_state = UkfState(pos=np.array([0.0, 0.0]), vel=np.array([speed, 0.0]), heading=0.0)
        ukf = DualChannelUkf(initial_state=initial_state, config=config)
        
        # We manually inject the yaw error into the state
        ukf.ukf.x[4] = yaw_error 
        
        for _ in range(steps):
            # No gyro rotation, so gyro_yaw = 0
            # Speed is perfect 10 m/s
            ukf.step(
                dt=dt,
                gyro_yaw=0.0,
                channel_a_speed=speed,
                channel_b_speed=0.0,
                gnss_pos=None,
                gnss_vel=None,
                road_signature_pos=None,
                road_signature_confidence=0.0,
                is_stationary=False
            )
            # Re-inject the static yaw error because process noise might let it drift slightly
            ukf.ukf.x[4] = yaw_error
            
        final_pos = ukf.ukf.x[0:2]
        
        # Expected Math
        # If North is X, East is Y:
        # North expected = 10 * 60 * cos(error)
        # East expected = 10 * 60 * sin(error)
        expected_north = speed * duration * np.cos(yaw_error)
        expected_east = speed * duration * np.sin(yaw_error)
        
        print(f"\nScenario: {name}")
        print(f"Final Pos (North, East) = ({final_pos[0]:.2f}, {final_pos[1]:.2f})")
        print(f"Expected  (North, East) = ({expected_north:.2f}, {expected_east:.2f})")
        
        err_north = final_pos[0] - expected_north
        err_east = final_pos[1] - expected_east
        total_error = np.linalg.norm([err_north, err_east])
        print(f"Discrepancy: {total_error:.4f} m")

if __name__ == "__main__":
    run_synthetic_test()
