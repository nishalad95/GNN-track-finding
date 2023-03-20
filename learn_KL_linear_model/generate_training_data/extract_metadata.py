import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import *
import pprint
import pickle
import csv


def KLDistance(mean1, cov1, inv1, mean2, cov2, inv2):
    trace = np.trace((cov1 - cov2) * (inv2 - inv1))
    return trace + (mean1 - mean2).T.dot(inv1 + inv2).dot(mean1 - mean2)

def calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs):
    pairwise_distances = np.zeros(shape=(num_edges, num_edges))
    for i in range(num_edges):
        for j in range(i):
            distance = KLDistance(edge_svs[i], edge_covs[i], inv_covs[i], edge_svs[j], edge_covs[j], inv_covs[j])
            pairwise_distances[i][j] = distance
    return pairwise_distances


parser = argparse.ArgumentParser(description='track hit-pair simulator')
parser.add_argument('-i', '--inputDir', help='input directory containing network gpickle file')
parser.add_argument('-o', '--outputDir', help='output directory to save metadata')
parser.add_argument('-n', '--numEvents', help='number of events to simulate')
args = parser.parse_args()

inputDir = args.inputDir
outputDir = args.outputDir
num_events = args.numEvents

# read in subgraph data
open_file = open(inputDir, "rb")
events = pickle.load(open_file)

print("EVENTS:\n", events)

# header = ['kl_dist', 'emp_var', 'truth']

# with open(outputDir + str(num_events) + '_events_training_data.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)

#     for _, subGraphs in events.items():
#         for n, subGraph in enumerate(subGraphs):
#             if n != 1: continue

#             for node in subGraph.nodes(data=True):
#                 node_num = node[0]
#                 node_attr = node[1]

#                 emp_var = node_attr['xy_edge_gradient_mean_var'][1]
#                 num_edges = query_node_degree_in_edges(subGraph, node_num)
#                 if num_edges <= 1: continue

#                 # convert attributes to arrays
#                 track_state_estimates = node_attr["track_state_estimates"]

#                 edge_connections = []
#                 for connection in list(track_state_estimates.keys()):
#                     edge_tuple = (connection, node_num)
#                     edge_connections.append(edge_tuple)
#                 edge_connections = np.array(edge_connections)

#                 # edge_connections = np.array([tuple(connection) for connection in list(track_state_estimates.keys())])
#                 edge_svs = np.array([component['edge_state_vector'] for component in track_state_estimates.values()])
#                 edge_covs = np.array([component['edge_covariance'] for component in track_state_estimates.values()])
#                 edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 3, 3))

#                 inv_covs = np.linalg.inv(edge_covs)

#                 pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)

#                 # compute truth edges
#                 node_truth = node_attr['truth_particle']
#                 for i in range(num_edges):
#                     for j in range(i):
#                         neighbor_node1 = edge_connections[i][0]
#                         neighbor_node2 = edge_connections[j][0]

#                         node1_truth = subGraph.nodes[neighbor_node1]['truth_particle']
#                         node2_truth = subGraph.nodes[neighbor_node2]['truth_particle']
#                         print("node1", node[0], "truth:", node_truth)
#                         print("neigbour1", neighbor_node1, "truth:", node1_truth)
#                         print("neighbour2", neighbor_node2, "truth:", node2_truth)

#                         truth = 0
#                         # pairwise between edge 1 and edge 2
#                         if (node_truth == node1_truth) and (node1_truth == node2_truth) and (node_truth == node2_truth):
#                             truth = 1   # truth_edge_pairs[i][j] = 1
#                         data = [pairwise_distances[i][j], emp_var, truth]
#                         writer.writerow(data)
                # print("truth_edge_pairs: \n", truth_edge_pairs)