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

def KLDistance(mean1, cov1, inv1, mean2, cov2, inv2):
    trace = np.trace((cov1 - cov2) * (inv2 - inv1))
    return trace + (mean1 - mean2).T.dot(inv1 + inv2).dot(mean1 - mean2)


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
for subGraph in subGraphs:
    print("---------------")
    print("SUBGRAPH")
    print("---------------")
    for node in subGraph.nodes(data=True):
        print("---------------")
        
        # convert graph attributes to arrays
        num_edges = node[1]['degree']
        node_state_estimates = pd.DataFrame(node[1]['track_state_estimates'])
        edge_svs = np.vstack(node_state_estimates.loc['edge_state_vector'].to_numpy())
        edge_covs = np.vstack(node_state_estimates.loc['edge_covariance'].to_numpy())
        edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 2, 2))
        inv_covs = np.linalg.inv(edge_covs)
        print("node num:", node)
        print("num edges:", num_edges)

        if num_edges <= 1: continue

        # calculate pairwise edge state vector KL distances
        pairwise_distances = np.zeros(shape=(num_edges, num_edges))
        for i in range(num_edges):
            for j in range(i):
                distance = KLDistance(edge_svs[i], edge_covs[i], inv_covs[i], edge_svs[j], edge_covs[j], inv_covs[j])
                pairwise_distances[i][j] = distance
                pairwise_distances[j][i] = distance
        print("pairwise distances")
        print(pairwise_distances)

        # if no outliers --> merge state vectors and covariances into a single estimate
        # if outliers --> remove them as attributes
        
        print("-----------------")


        # # TODO: repeat for number of clusters
        # # initialize the center to a random candidate    
        # centre_sv = edge_sv[0].reshape(-1,1)  # column vector
        # centre_cov = edge_cov[0]
        # cluster_labels = np.empty((0, num_edges))

        # # TODO: need to repeat the following until convergence, convergence measure?
        # n = 0
        # while n < iters:
        #     print("iteration num: ", n, "\n")         
        #     distances = np.empty((0, num_edges))

        #     for i in range(num_clusters):
        #         # compute KL distance
        #         p1 = gaussian_pdf(centre_sv, centre_cov, mean_state_vector)
        #         p1 = np.repeat(p1, repeats=num_edges, axis=0)
        #         # TODO: faster way to execute this for loop - np broadcasting?
        #         p2 = gaussian_pdf(edge_sv, edge_cov, mean_svs)

        #         # p2 = np.zeros((num_edges,1))
        #         # for j in range(num_edges):
        #         #     current_p2 = gaussian_pdf(edge_sv[j], edge_cov[j], mean_svs[j])
        #         #     p2[j,0] = current_p2         
        #         distances = KLDistance(p1, p2)
        #     print("node: ", node)
        #     print("distances: ", distances, "distances shape", distances.shape, "\n")

        #     # assign each data point to the nearest cluster centre
        #     cluster_labels = np.where(distances < KL_thres, 1, 0)

        #     # recompute centre position of cluster
        #     sv_cluster = edge_sv[cluster_labels[:, 0] == 1]
        #     if len(sv_cluster) > 0:
        #         new_centre_sv = np.mean(sv_cluster, axis=0).reshape(-1,1)
        #         centre_sv = new_centre_sv
        #         cov_cluster = edge_cov[cluster_labels[:, 0] == 1]
        #         new_centre_cov = np.mean(cov_cluster, axis=0)
        #         centre_cov = new_centre_cov

        #     if (n == iters - 1):
        #         print("Clustering complete after " + str(iters) + " iterations\n")

        #     # repeat this until convergence
        #     n += 1

        # # add labels to track_state_estimates_df
        # cluster_labels = cluster_labels.reshape(1,-1)[0]
        # track_state_estimates_df.loc['cluster_labels'] = cluster_labels
        # # print("type:", type(cluster_labels[0]))

        # # remove edges not in cluster
        # # only run this if the length of 1's < original number of edges
        # if len(cluster_labels[cluster_labels != 0]) < num_edges:
        #     print("test:")
        #     print("num_edges:", num_edges)
        #     print("filteres num edges:", len(cluster_labels[cluster_labels != 0]))
        #     # remove outlier edges
        #     # update the mean_state_vector and other attributes of the subgraph
            
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