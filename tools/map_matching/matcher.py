import numpy as np

def angle_diff(a, b):
    diff = (a - b + np.pi) % (2 * np.pi) - np.pi
    return abs(diff)

class RealisticMatcher:
    def __init__(self, net, dist_weight=1.0, heading_weight=20.0, continuity_penalty=10.0):
        self.net = net
        self.dist_weight = dist_weight
        self.heading_weight = heading_weight
        self.continuity_penalty = continuity_penalty
        self.last_matched_segment = None
        
    def step(self, ukf_pos, ukf_heading):
        candidates = self.net.get_nearby_segments(ukf_pos, radius=30.0)
        
        if not candidates:
            self.last_matched_segment = None
            return None, None, 0.0
            
        best_score = float('inf')
        best_cand = None
        
        for cand in candidates:
            seg = cand['segment']
            
            # Distance penalty
            d_score = self.dist_weight * cand['dist']
            
            # Heading penalty (map heading is bi-directional, so check both ways)
            h_diff1 = angle_diff(seg['heading'], ukf_heading)
            h_diff2 = angle_diff(seg['heading'] + np.pi, ukf_heading)
            h_score = self.heading_weight * min(h_diff1, h_diff2)
            
            # Continuity penalty
            c_score = 0.0
            if self.last_matched_segment is not None:
                last_edge = self.last_matched_segment['segment']['edge_id']
                curr_edge = seg['edge_id']
                
                # Check if edges share a node (u, v)
                last_u, last_v = last_edge[:2]
                curr_u, curr_v = curr_edge[:2]
                
                if not (last_v == curr_u or last_u == curr_v or last_u == curr_u or last_v == curr_v):
                    c_score = self.continuity_penalty
                    
            score = d_score + h_score + c_score
            
            if score < best_score:
                best_score = score
                best_cand = cand
                
        self.last_matched_segment = best_cand
        
        # Best direction
        best_h = best_cand['segment']['heading']
        if angle_diff(best_h + np.pi, ukf_heading) < angle_diff(best_h, ukf_heading):
            best_h = (best_h + np.pi) % (2*np.pi) - np.pi
            
        return best_cand['proj'], best_h, best_score
