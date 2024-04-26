"""
The framework will structure 
"""

import numpy as np

class coupling:
    def __init__(self, epsilon):
        self.initCoRot()
        self.initBEVC()

        # Main loop options
        self.epsilon = epsilon
        self.delta_u_rel = epsilon + 1
        self.iter_max = 20

    delta_history = []
    iter = 0

    def initCoRot (self):
        pass
    def initBEVC(self):
        pass

    def run(self):
        while abs(self.delta_u_rel) > self.epsilon and iter < self.iter_max:
            pass
    
    
