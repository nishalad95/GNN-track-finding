class GNN_Measurement(object) :
    def __init__(self, x, y, t, label = -1, n = None) :
        self.x = x
        self.y = y
        self.t = t # track inclination - gradient
        self.track_label = label
        self.node = n