import numpy as np
from scipy.spatial.distance import cdist

class CorridorRoad:
    """
    Represents a continuous road polyline with along-track distance parameterization.
    """
    def __init__(self, segments, initial_idx=0):
        # Chain connected segments into a polyline
        pts = [segments[initial_idx]['p1'], segments[initial_idx]['p2']]
        curr_p = segments[initial_idx]['p2']
        
        # Sequentially attach downstream segments if connected
        remaining = [s for i, s in enumerate(segments) if i != initial_idx]
        while remaining:
            best_i = None
            best_dist = 5.0 # meters
            for i, s in enumerate(remaining):
                d1 = np.linalg.norm(s['p1'] - curr_p)
                if d1 < best_dist:
                    best_dist = d1
                    best_i = i
            if best_i is not None:
                next_s = remaining.pop(best_i)
                pts.append(next_s['p2'])
                curr_p = next_s['p2']
            else:
                break
                
        self.points = np.array(pts) # (N, 2) [North, East]
        seg_vecs = np.diff(self.points, axis=0)
        self.seg_lengths = np.linalg.norm(seg_vecs, axis=1)
        self.cum_lengths = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        self.total_length = self.cum_lengths[-1]
        
        # Segment headings [atan2(East, North)]
        self.seg_headings = np.arctan2(seg_vecs[:, 1], seg_vecs[:, 0])

    def interpolate_position(self, s):
        """
        Converts arc length s (meters) into 2D [North, East] position and heading.
        """
        s_clamped = np.clip(s, 0.0, self.total_length)
        # Find which segment contains s
        idx = np.searchsorted(self.cum_lengths, s_clamped, side='right') - 1
        idx = min(max(0, idx), len(self.seg_lengths) - 1)
        
        seg_s = s_clamped - self.cum_lengths[idx]
        t = seg_s / (self.seg_lengths[idx] + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        
        pos = self.points[idx] + t * (self.points[idx+1] - self.points[idx])
        heading = self.seg_headings[idx]
        return pos, heading

    def get_curvature_profile(self, ds=1.0):
        """
        Discretizes road heading and curvature at uniform arc-length intervals ds.
        """
        s_grid = np.arange(0, self.total_length, ds)
        headings = np.zeros(len(s_grid))
        for i, s in enumerate(s_grid):
            _, h = self.interpolate_position(s)
            headings[i] = h
        headings_unwrapped = np.unwrap(headings)
        curvatures = np.gradient(headings_unwrapped, ds)
        return s_grid, headings_unwrapped, curvatures

class CorridorFilter:
    """
    1-D Corridor Kalman Filter (Stage 12, Phase 11-14).
    State: [s, v, b_v]
      s: along-track distance along candidate road (m)
      v: forward speed (m/s)
      b_v: forward speed bias (m/s)
    """
    def __init__(self, road: CorridorRoad, s_init=0.0, v_init=0.0, P_init=None):
        self.road = road
        self.x = np.array([s_init, v_init, 0.0], dtype=np.float64)
        if P_init is None:
            self.P = np.diag([4.0, 1.0, 0.1]) # Initial variances: (2m)^2, (1m/s)^2, (0.3m/s)^2
        else:
            self.P = np.array(P_init, dtype=np.float64)
            
        self.q_s = 0.01**2
        self.q_v = (1.0)**2
        self.q_b = (0.01)**2

    def predict(self, dt):
        F = np.array([
            [1.0, dt, dt],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        Q = np.array([
            [self.q_s * dt, 0.0, 0.0],
            [0.0, self.q_v * dt, 0.0],
            [0.0, 0.0, self.q_b * dt]
        ])
        self.x = F @ self.x
        self.x[1] = max(0.0, self.x[1]) # speed cannot be negative
        self.P = F @ self.P @ F.T + Q

    def update_speed(self, speed_meas, uncertainty_meas, safety_floor=0.25):
        """
        Update with learned / calibrated speed measurement.
        z_v = v + b_v
        """
        H = np.array([[0.0, 1.0, 1.0]])
        R = uncertainty_meas + safety_floor
        
        y = speed_meas - (H @ self.x)[0]
        S = (H @ self.P @ H.T)[0, 0] + R
        K = (self.P @ H.T) / S
        
        self.x = self.x + K.flatten() * y
        self.x[1] = max(0.0, self.x[1])
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def update_progress_anchor(self, s_anchor, confidence, base_std=3.0):
        """
        Soft progress anchor update (Phase 14).
        z_s = s
        """
        if confidence < 0.5:
            return # Gated out
            
        R_s = (base_std / max(1e-3, confidence)) ** 2
        H = np.array([[1.0, 0.0, 0.0]])
        
        y = s_anchor - self.x[0]
        S = (H @ self.P @ H.T)[0, 0] + R_s
        K = (self.P @ H.T) / S
        
        self.x = self.x + K.flatten() * y
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

    def get_position(self):
        """
        Returns metric 2D position [North, East] and heading on the corridor polyline.
        """
        return self.road.interpolate_position(self.x[0])

class RoadProgressMatcher:
    """
    Deterministic Road Geometry Progress Anchors (Phase 13).
    Aligns cumulative observed IMU yaw-rate turn events with candidate road curvature profile.
    """
    def __init__(self, road: CorridorRoad):
        self.road = road
        self.s_grid, self.headings, self.curvatures = road.get_curvature_profile(ds=1.0)
        self.cum_turn_map = self.headings - self.headings[0]

    def match_progress(self, s_est, imu_heading_change, search_window=20.0):
        """
        Given the current estimated along-track progress s_est and the cumulative
        IMU heading change (rad), finds the nearest arc length s with matching turn signature.
        """
        # Window around current estimate
        s_min = max(0.0, s_est - search_window)
        s_max = min(self.road.total_length, s_est + search_window)
        
        mask = (self.s_grid >= s_min) & (self.s_grid <= s_max)
        if not np.any(mask):
            return s_est, 0.0
            
        cand_s = self.s_grid[mask]
        cand_turns = self.cum_turn_map[mask]
        
        # Turn discrepancy
        diff = np.abs(cand_turns - imu_heading_change)
        best_idx = np.argmin(diff)
        best_diff = diff[best_idx]
        
        # Confidence decays exponentially with heading angle error
        # If angle matches within 0.1 rad (~5.7 deg), confidence is high (~0.9)
        conf = float(np.exp(-0.5 * (best_diff / 0.15)**2))
        return float(cand_s[best_idx]), conf
