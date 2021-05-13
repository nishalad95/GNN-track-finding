from filterpy.kalman import KalmanFilter
from filterpy import common
from scipy.stats import chisquare, distributions
import matplotlib.pyplot as plt
import bisect
import networkx as nx
import numpy as np
import os
from collections import deque
import argparse
from plotting import *


# def save_network(directory, i, subGraph):
#     filename = directory + str(i) + "_subgraph.gpickle"
#     nx.write_gpickle(subGraph, filename)
#     A = nx.adjacency_matrix(subGraph).todense()
#     np.savetxt(directory + str(i) + "_subgraph_matrix.csv", A)

def main():
    
    # parse command line args
    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-c', '--candidates', help='output directory to save track candidates')
    parser.add_argument('-r', '--remain', help='output directory to save remaining network')
    parser.add_argument('-cs', '--chisq', help='chi-squared track candidate acceptance level')
    args = parser.parse_args()

    subgraph_path = "_subgraph.gpickle"
    inputDir = args.input
    outputDir = args.candidates
    remaining_network = args.remain
    track_acceptance = float(args.chisq)
    
    sigma0 = 0.5 #r.m.s of track position measurements
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements

    # TODO: faster to implement as sets rather than lists?
    # read in subgraph data
    subGraphs = []
    i = 0
    path = inputDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        subGraphs.append(sub)
        i += 1
        path = inputDir + str(i) + subgraph_path

    track_candidates = np.zeros((len(subGraphs)), dtype=int)
    chi2_inc = np.array([])
    ddof_values = []

    for i, sub in enumerate(subGraphs):
        if len(sub.nodes()) <= 1 : continue

        obs_xpos = []
        obs_ypos = []
        for _,v in sub.nodes(data=True):
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

        # initialize KF
        f = KalmanFilter(dim_x=2, dim_z=1)

        f.x = np.array([obs_ypos[0], 0.])  # X state vector
        
        f.F = np.array([[1.,1.],
                        [0.,1.]])   # F state transition matrix
        
        f.H = np.array([[1.,0.]]) # H measurement matrix
        f.P = np.array([[sigma0**2,    0.],
                        [0.,         1000.]])  # P: covariance

        f.R = sigma0**2
        f.Q = 0.

        # save data for kf filter
        saver = common.Saver(f)
        for y_pos in obs_ypos:
            f.predict()
            f.update(y_pos)
            saver.save()

        x_state = np.array(saver['x'])
        y_a = x_state[:, 0] # y_a = y_b + t_b(x_a - x_b)

        # plot the smoothed tracks
        # plt.scatter(obs_xpos, y_a, alpha=0.5, label="KF")
        # plt.scatter(obs_xpos[::-1], obs_ypos[::-1], alpha=0.5, label="Measurement")

        # observed state vectors from measurements (coords)
        obs_ypos_deq = deque(obs_ypos)
        obs_ypos_deq.rotate(-1)
        shifted_y = np.array(obs_ypos_deq)
        dy = np.array(obs_ypos) - shifted_y
        gradient = dy / -1
        gradient = np.insert(gradient, 0, 0.)
        gradient = gradient[0:-1]
        obs_state = np.array(list(zip(obs_ypos, gradient)))
        
        # residual = measurement - H.x_state
        x_state[:,1] = 0
        H_x_state = x_state
        residual = obs_state - H_x_state

        # inv covariance
        inv_cov_sum = np.linalg.inv(np.array(S) - saver['P_prior'])

        # chi2 increment
        residual_row_vectors = residual[:, np.newaxis, :]
        residual_column_vectors = residual[:, :, np.newaxis]
        matmul = np.matmul(inv_cov_sum, residual_column_vectors)
        chi2_inc = np.matmul(residual_row_vectors, matmul)
        chi2_inc = chi2_inc.reshape(-1, len(chi2_inc))

        # degrees of freedom
        num_track_params = 2
        ddof = len(obs_ypos) - num_track_params
        ddof_values.append(ddof)
        
        # TODO: only using [:,0] for now
        _, p = chisquare(obs_state[:,0], x_state[:,0], ddof=ddof)

        # TODO: maybe use chi2 distance and variance as a threshold?
        if p >= track_acceptance:
            track_candidates[i] = 1

    ddof_values = np.array(ddof_values).reshape(-1, 1)
    chi2_inc_deq = deque(chi2_inc[0])
    chi2_inc_deq.rotate(1)
    shifted_chi2_inc = np.array(chi2_inc_deq).reshape(1, -1)
    chi2 = chi2_inc + shifted_chi2_inc
    chi2[0][0] = 0.
    result = distributions.chi2.sf(chi2, ddof_values)

    # chisq histogram distribution should be uniform!
    # plt.hist(result, density=True)
    # plt.show()

    # plotting for the smoothed tracks
    # plt.xlim([0, 11])
    # plt.ylim([-25, 15])
    # plt.legend()
    # plt.show()

    print("track candidate idxs", track_candidates)

    # save the track candidates
    idxs = np.where(track_candidates == 1)[0]
    r_network = []
    for i, subGraph in enumerate(subGraphs):
        if i not in idxs:
            # not a candidate
            save_network(remaining_network, i, subGraph)
            r_network.append(subGraph)
        else:
            # track candidate
            save_network(outputDir, i, subGraph)

    
    # plot remaining network to visualise graphs that still need processing
    title = "Remaining Networks to be processed"
    plot_save_subgraphs(r_network, remaining_network, title)


if __name__ == "__main__":
    main()