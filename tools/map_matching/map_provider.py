import os
import networkx as nx
import osmnx as ox
import numpy as np
from scipy.spatial import cKDTree

class RoadNetwork:
    def __init__(self, lat_min, lat_max, lon_min, lon_max, lat0, lon0, cache_file='data/s_s2_map.graphml'):
        self.lat0 = lat0
        self.lon0 = lon0
        self.lat_to_m = 111000.0
        self.lon_to_m = 111000.0 * np.cos(np.radians(lat0))
        
        if os.path.exists(cache_file):
            print(f"Loading cached map from {cache_file}...")
            self.G = ox.load_graphml(cache_file)
        else:
            print("Downloading map data using osmnx (this may take a minute)...")
            # Pad bounding box slightly to ensure all roads are caught
            pad = 0.01
            # ox.graph_from_bbox takes bbox=(left, bottom, right, top) i.e. (west, south, east, north)
            self.G = ox.graph_from_bbox(
                bbox=(lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad),
                network_type='drive'
            )
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            ox.save_graphml(self.G, cache_file)
            print("Saved map to cache.")
            
        self.segments = self._build_segments()
        
        # Build KDTree using segment midpoints for fast lookup
        midpoints = []
        for seg in self.segments:
            mid = (seg['p1'] + seg['p2']) / 2.0
            midpoints.append(mid)
        self.kdtree = cKDTree(midpoints)
        
    def _proj(self, lat, lon):
        x = (lon - self.lon0) * self.lon_to_m
        y = (lat - self.lat0) * self.lat_to_m
        return np.array([x, y])
        
    def _build_segments(self):
        segments = []
        # Extract nodes
        nodes = {n: (d['y'], d['x']) for n, d in self.G.nodes(data=True)}
        
        for u, v, key, data in self.G.edges(keys=True, data=True):
            # osmnx can have simplified geometry in 'geometry' key
            if 'geometry' in data:
                coords = list(data['geometry'].coords)
            else:
                coords = [(nodes[u][1], nodes[u][0]), (nodes[v][1], nodes[v][0])] # (lon, lat)
                
            for i in range(len(coords) - 1):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[i+1]
                
                p1 = self._proj(lat1, lon1)
                p2 = self._proj(lat2, lon2)
                
                dp = p2 - p1
                length = np.linalg.norm(dp)
                if length > 0.1:
                    heading = np.arctan2(dp[1], dp[0]) # mathematical angle: 0=East, pi/2=North
                    # Wait, our UKF uses 0=North, pi/2=East, -pi/2=West?
                    # Let's align with the UKF coordinates.
                    # In evaluate.py: gnss_pos[:, 0] = x (lat?), gnss_pos[:, 1] = y (lon?)
                    # Wait, evaluate.py:
                    # gnss_pos[:, 0] = (lat - lat0) * lat_to_m  --> X is North
                    # gnss_pos[:, 1] = (lon - lon0) * lon_to_m  --> Y is East
                    # So x_n = (lat-lat0), y_e = (lon-lon0)
                    
                    # p1[0] = X (North), p1[1] = Y (East)
                    x1 = (lat1 - self.lat0) * self.lat_to_m
                    y1 = (lon1 - self.lon0) * self.lon_to_m
                    x2 = (lat2 - self.lat0) * self.lat_to_m
                    y2 = (lon2 - self.lon0) * self.lon_to_m
                    
                    pp1 = np.array([x1, y1])
                    pp2 = np.array([x2, y2])
                    
                    dpp = pp2 - pp1
                    heading_ne = np.arctan2(dpp[1], dpp[0]) # angle from North towards East
                    
                    segments.append({
                        'p1': pp1,
                        'p2': pp2,
                        'length': length,
                        'heading': heading_ne,
                        'edge_id': (u, v)
                    })
        return segments
    
    def get_nearby_segments(self, pos, radius=50.0):
        # Query KDTree for segments whose midpoint is within radius + max segment length / 2
        # To be safe, query within radius + 100 meters
        idx = self.kdtree.query_ball_point(pos, r=radius + 100.0)
        
        candidates = []
        for i in idx:
            seg = self.segments[i]
            p1 = seg['p1']
            p2 = seg['p2']
            l2 = seg['length']**2
            if l2 == 0:
                continue
            
            # t is projection of pos onto line p1-p2
            t = max(0, min(1, np.dot(pos - p1, p2 - p1) / l2))
            proj = p1 + t * (p2 - p1)
            dist = np.linalg.norm(pos - proj)
            
            if dist < radius:
                candidates.append({
                    'segment': seg,
                    'proj': proj,
                    'dist': dist
                })
                
        # Sort by distance
        candidates.sort(key=lambda x: x['dist'])
        return candidates
