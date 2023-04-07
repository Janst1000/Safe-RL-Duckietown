import numpy as np
from pprint import pprint

class SafetyLayer():
    def __init__(self, max_lane_d, max_lane_phi, max_tof_d):
        self.max_lane_d = max_lane_d
        self.max_lane_phi = max_lane_phi
        self.max_tof_d = max_tof_d
        self.tof = 0
        
    def check_safety(self, action, predicted_state):
        if predicted_state[0] > self.max_lane_d or predicted_state[0] < -self.max_lane_d:
            return False
        elif predicted_state[1] > self.max_lane_phi or predicted_state[1] < -self.max_lane_phi:
            return True
        else:
            return True