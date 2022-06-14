class GNN_Measurement(object) :
    def __init__(self, x, y, z, r, s, m, t, label = -1, n = None) :
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.t = t                # track inclination - gradient, used in HitPairPredictor
        self.sigma0 = s           # r.m.s of track position measurements
        self.mu = m               # uncertainty due to multiple scattering - process noise
        self.track_label = label  # MC truth track label (particle reference)
        self.node = n