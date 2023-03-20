class GNN_Measurement(object) :
    def __init__(self, x, y, z, r, truth_particle = -1, n = None) :
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        # self.sigma0 = sigma0                    # r.m.s of track position measurements
        self.truth_particle = truth_particle    # MC truth track label (particle reference)
        self.node = n