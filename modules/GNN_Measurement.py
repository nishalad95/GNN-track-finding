class GNN_Measurement(object) :
    def __init__(self, x, y, t, s, label = -1, n = None) :
        self.x = x
        self.y = y
        self.t = t                # track inclination - gradient, used in HitPairPredictor
        self.track_label = label  # MC truth track label (particle reference)
        self.node = n
        self.sigma0 = s           # r.m.s of track position measurements


class GNN_Measurement_3D(object) :
    def __init__(self, x, y, z, t, s, label = -1, n = None) :
        self.x = x
        self.y = y
        self.t = t                # track inclination - gradient, used in HitPairPredictor
        self.z = z                
        self.track_label = label  # MC truth track label (particle reference)
        self.node = n
        self.sigma0 = s           # r.m.s of track position measurements