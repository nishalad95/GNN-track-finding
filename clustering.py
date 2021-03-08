import os
import sys
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
from scipy.stats import norm
import scipy.stats
from numpy.linalg import inv


# global variables
DIR = "./simulation/"
subgraph_path = "_subgraph.gpickle"
KL_thres = 1.0
iters = 2


def gaussian_pdf(x, cov, mean):
    det = np.array([np.linalg.det(cov)]).reshape(-1,1)
    inv = np.linalg.inv(cov)
    norm_factor = 1 / (2 * np.pi * (det)**0.5 )
    expo = np.exp( -0.5 * (x - mean).T.dot(inv).dot(x - mean) )
    return norm_factor.T * expo


def KLDistance(p1, p2):
    KL = 2 * ( (np.where(p1 != 0, p1 * np.log(p1/p2), 0)) + (np.where(p2 != 0, p2 * np.log(p2/p1), 0)) )
    return KL


# read in subgraph data
subGraphs = []
i = 0
path = DIR + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = DIR + str(i) + subgraph_path


# k-means clustering on edges with KL-distance
num_clusters = 1

sub = 1
for subGraph in subGraphs:
    print("subgraph ", sub)
    sub +=1
    for node in subGraph.nodes():

        # convert graph attributes to arrays
        track_state_estimates = subGraph.nodes[node]['track_state_estimates']
        track_state_estimates_df = pd.DataFrame(track_state_estimates)
        num_edges = subGraph.nodes[node]['degree']
        print("num edges:", num_edges)

        if num_edges <= 1: continue
        edge_sv = np.vstack(track_state_estimates_df.loc['edge_state_vector'].to_numpy())
        edge_cov = np.vstack(track_state_estimates_df.loc['edge_covariance'].to_numpy())
        # adding new dimension for broadcasting
        edge_cov = edge_cov[:, :, np.newaxis]
        edge_cov = np.reshape(edge_cov, (num_edges, 2, 2)) 
        mean_state_vector = subGraph.nodes[node]['mean_state_vector'].reshape(-1,1)     # column vector
        mean_svs = np.repeat(mean_state_vector.reshape(1,-1), repeats=num_edges, axis=0)

        # TODO: repeat for number of clusters
        # initialize the center to a random candidate    
        centre_sv = edge_sv[0].reshape(-1,1)  # column vector
        centre_cov = edge_cov[0]
        cluster_labels = np.empty((0, num_edges))

        # TODO: need to repeat the following until convergence, convergence measure?
        n = 0
        while n < iters:
            print("iteration num: ", n, "\n")         
            distances = np.empty((0, num_edges))

            for i in range(num_clusters):
                # compute KL distance
                p1 = gaussian_pdf(centre_sv, centre_cov, mean_state_vector)
                p1 = np.repeat(p1, repeats=num_edges, axis=0)
                # TODO: faster way to execute this for loop - np broadcasting?
                p2 = np.zeros((num_edges,1))
                for j in range(num_edges):
                    current_p2 = gaussian_pdf(edge_sv[j], edge_cov[j], mean_svs[j])
                    p2[j,0] = current_p2         
                distances = KLDistance(p1, p2)
            print("node: ", node)
            print("distances: ", distances, "distances shape", distances.shape, "\n")

            # assign each data point to the nearest cluster centre
            cluster_labels = np.where(distances < KL_thres, 1, 0)

            # recompute centre position of cluster
            sv_cluster = edge_sv[cluster_labels[:, 0] == 1]
            if len(sv_cluster) > 0:
                new_centre_sv = np.mean(sv_cluster, axis=0).reshape(-1,1)
                centre_sv = new_centre_sv
                cov_cluster = edge_cov[cluster_labels[:, 0] == 1]
                new_centre_cov = np.mean(cov_cluster, axis=0)
                centre_cov = new_centre_cov

            if (n == iters - 1):
                print("Clustering complete after " + str(iters) + " iterations\n")

            # repeat this until convergence
            n += 1

        # add labels to track_state_estimates_df
        cluster_labels = cluster_labels.reshape(1,-1)[0]
        track_state_estimates_df.loc['cluster_labels'] = cluster_labels
        # print("type:", type(cluster_labels[0]))

        # remove edges not in cluster
        # only run this if the length of 1's < original number of edges
        if len(cluster_labels[cluster_labels != 0]) < num_edges:
            print("test:")
            print("num_edges:", num_edges)
            print("filteres num edges:", len(cluster_labels[cluster_labels != 0]))
            # remove outlier edges
            # update the mean_state_vector and other attributes of the subgraph
            
# plot the subgraphs to view the difference after clustering



# Kalman filter - fixed lag smoother
# from filterpy.kalman import FixedLagSmoother
# fls = FixedLagSmoother(dim_x=2, dim_z=1)

# i = 0
# for subGraph in subGraphs:
#     print("subGraph ", i)
#     i += 1
#     print(subGraph.nodes(data=True))

# fls.x = np.array([[0.],
#                   [.5]])

# fls.F = np.array([[1.,1.],
#                   [0.,1.]])

# fls.H = np.array([[1.,0.]])