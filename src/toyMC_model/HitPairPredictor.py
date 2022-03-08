import numpy as np
class HitPairPredictor() :
    def __init__(self, start_x, y0_range, tau0_range) :
        self.start = start_x
        self.min_y0 = -y0_range
        self.max_y0 =  y0_range
        self.min_tau = -tau0_range
        self.max_tau =  tau0_range
    
    def predict(self, m1, m2, start = 0) :
        dx = m2.x-m1.x
        # tau0 = (m2.y-m1.y)/dx # gradient
        y0 =(m1.y*m2.x-m2.y*m1.x+self.start*(m2.y-m1.y))/dx
        # if tau0 > self.max_tau or tau0 < self.min_tau : return 0
        if y0 > self.max_y0 or y0 < self.min_y0 : return 0
        return 1