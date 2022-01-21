class GNN_Measurement(object) :
    def __init__(self, x, y, z, t, s, label = -1, n = None) :
        self.x = x
        self.y = y
        self.z = z
        self.t = t                  # track inclination - gradient
        self.track_label = label    # MC truth track label (particle reference)
        self.node = n
        self.sigma0 = s             #r.m.s of track position measurements