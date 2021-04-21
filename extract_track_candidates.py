from filterpy.kalman import FixedLagSmoother, KalmanFilter
from filterpy import common
from scipy.stats import chisquare
import matplotlib.pyplot as plt
import bisect
import networkx as nx
import numpy as np
import os


DIR = "./02_outliers_removed/"
subgraph_path = "_subgraph.gpickle"
SAVE_DIR = "./track_candidates/"
REMAINING_NETWORK = "./remaining_network/"
TRACK_ACC_LEVEL = 0.6

# TODO: faster to implement as sets rather than lists?
# read in subgraph data
subGraphs = []
i = 0
path = DIR + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = DIR + str(i) + subgraph_path


track_candidates = np.zeros((len(subGraphs)), dtype=int)
for i, sub in enumerate(subGraphs):
    if len(sub.nodes()) <= 1 : continue

    print("SUBGRAPH")
    obs_xpos = []
    obs_ypos = []
    for k,v in sub.nodes(data=True):
        x_pos = v['GNN_Measurement'].x
        y_pos = v['GNN_Measurement'].y

        # check for 1 measurement per layer
        if x_pos in obs_xpos: 
            obs_xpos = []
            obs_ypos = []
            break
        # sort measurements by x positions
        bisect.insort(obs_xpos, x_pos)
        idx = obs_xpos.index(x_pos)
        obs_ypos.insert(idx, y_pos)
    
    if len(obs_xpos) == 0 : continue

    # handling holes
    holes = False
    holes = not (obs_xpos == list(range(min(obs_xpos), max(obs_xpos)+1)))
    if holes : continue

    # reverse the lists
    obs_xpos = obs_xpos[::-1]
    obs_ypos = obs_ypos[::-1]

    # initialize fixed lag smoother
    fls = FixedLagSmoother(dim_x=2, dim_z=1)
    fls.x = np.array([obs_ypos[0], 0.])  # X state vector
    fls.F = np.array([[1.,1.],
                    [0.,1.]])   # F state transition matrix
    fls.H = np.array([[1.,0.]]) # H measurement matrix
    sigma0 = 0.5
    fls.P = np.array([[sigma0**2,    0.],
                    [0.,         1000.]])  # P: covariance
    fls.R = sigma0**2
    fls.Q = 0.
    x_smoothed, _ = fls.smooth_batch(obs_ypos, N=len(obs_ypos))  

    y_a = x_smoothed[:, 0] - x_smoothed[:, 1]
    plt.scatter(obs_xpos, obs_ypos, label="Measurement")
    plt.scatter(obs_xpos, y_a, alpha=0.6, label="x_smoothed")

    # initialize KF
    # f = KalmanFilter(dim_x=2, dim_z=1)

    # f.x = np.array([obs_ypos[0], 0.])  # X state vector
    
    # f.F = np.array([[1.,1.],
    #                 [0.,1.]])   # F state transition matrix
    
    # f.H = np.array([[1.,0.]]) # H measurement matrix
    # sigma0 = 0.5
    # f.P = np.array([[sigma0**2,    0.],
    #                 [0.,         1000.]])  # P: covariance

    # f.R = sigma0**2
    # f.Q = 0.

    # # save data for kf filter
    # saver = common.Saver(f)
    # for y_pos in obs_ypos:
    #     f.predict()
    #     f.update(y_pos)
    #     saver.save()

    # x_state = np.array(saver['x'])
    # y_a = x_state[:, 0] - x_state[:, 1] # y_a = y_b + t_b(x_a - x_b)
    
    # print("observed y", obs_ypos)
    # print("x[:, 0]", x_state[:, 0])
    # print("x[:, 1]", x_state[:, 1])

    # plt.scatter(obs_xpos, y_a, alpha=0.5, label="KF")
    # plt.scatter(obs_xpos[::-1], obs_ypos[::-1], alpha=0.5, label="Measurement")
    # plt.legend()
    # plt.show()

    # chi2 calculation
    num_track_params = 2
    ddof = len(obs_ypos) - num_track_params
    chisq, p = chisquare(obs_ypos, f_exp=x_smoothed[:, 0], ddof=ddof)
    print("chi2 ", chisq, p)

    if p >= TRACK_ACC_LEVEL:
        track_candidates[i] = 1

plt.xlim([0, 11])
plt.ylim([-25, 15])
plt.legend()
plt.show()

def save_network(directory, i, subGraph):
    filename = directory + str(i) + "_subgraph.gpickle"
    nx.write_gpickle(subGraph, filename)
    A = nx.adjacency_matrix(subGraph).todense()
    np.savetxt(directory + str(i) + "_subgraph_matrix.csv", A)

# save the track candidates
idxs = np.where(track_candidates == 1)[0]

for i, subGraph in enumerate(subGraphs):
    if i not in idxs:
        # not a candidate
        save_network(REMAINING_NETWORK, i, subGraph)
    else:
        # track candidate
        save_network(SAVE_DIR, i, subGraph)