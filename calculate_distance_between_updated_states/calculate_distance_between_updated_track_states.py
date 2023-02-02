# updated_track_state = [a, b, c] --> can use chi2 distance here as the coordinate info is known
# first try clustering with [a, b, c] and then see if we need to switch to [a, b, tau]
# extract all chi2 distances for nodes with updated_track_state

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# from utils import *
import pprint
import pickle
import csv
import os
import networkx as nx
import glob
from math import *


def query_node_degree_in_edges(subGraph, node_num):
    in_edges = subGraph.in_edges(node_num) # one direction only, not double counted
    node_degree = 0
    for edge in in_edges:
        neighbour_num = edge[0]
        if (subGraph[neighbour_num][node_num]["activated"] == 1) : node_degree += 1
    return node_degree

def mahalanobis_distance(mean1, cov1, mean2, cov2, node_coords, neighbour1_coords, neighbour2_coords):
    
    edge1 = mean1[:2]
    edge2 = mean2[:2]
    residual = edge1 - edge2

    # covariance of delta_a and delta_b
    ab_block1 = cov1[0:2, 0:2]
    ab_block2 = cov2[0:2, 0:2]
    covariance_delta_ab = ab_block1 + ab_block2
    inv_covariance_delta_ab = np.linalg.inv(covariance_delta_ab)
    
    # chi2 contribution from [a, b] part of joint state vector
    distance1 = residual.T.dot(inv_covariance_delta_ab).dot(residual)

    # neighbours and node coords
    x_a = node_coords[0]
    x_b = neighbour1_coords[0]
    x_c = neighbour2_coords[0]
    y_a = node_coords[1]
    y_b = neighbour1_coords[1]
    y_c = neighbour2_coords[1]
    z_a, r_a = node_coords[2], node_coords[3]
    z_b, r_b = neighbour1_coords[2], neighbour1_coords[3]
    z_c, r_c = neighbour2_coords[2], neighbour2_coords[3]

    # jacobian for covariance of delta tau
    j2 = 1/(r_b - r_a)
    j3 = -1/(r_c - r_a)
    j1 = - j3 - j2
    j5 = -(z_b - z_a)/(r_b - r_a)**2
    j6 = (z_c - z_a)/(r_c - r_a)**2
    j4 = - j5 - j6
    J = np.array([j1, j2, j3, j4, j5, j6])

    # error in barrel
    sigma_za, sigma_zb, sigma_zc = 0.5, 0.5, 0.5
    sigma_ra, sigma_rb, sigma_rc = 0.1, 0.1, 0.1
    # if node in endcap, then reduce z error
    if np.abs(x_a) >= 600.0: 
        sigma_za = 0.1
        sigma_ra = 0.5
    if np.abs(x_b) >= 600.0: 
        sigma_zb = 0.1
        sigma_rb = 0.5
    if np.abs(x_c) >= 600.0: 
        sigma_zc = 0.1
        sigma_rc = 0.5

    # covariance
    S = np.array([  [sigma_za**2, 0, 0, 0, 0, 0],
                    [0, sigma_zb**2, 0, 0, 0, 0],
                    [0, 0, sigma_zc**2, 0, 0, 0],
                    [0, 0, 0, sigma_ra**2, 0, 0],
                    [0, 0, 0, 0, sigma_rb**2, 0],
                    [0, 0, 0, 0, 0, sigma_rc**2]])
    
    # covariance of delta tau
    cov_delta_tau = J.dot(S).dot(J.T)
    inv_cov_delta_tau = 1/cov_delta_tau

    # chi2 contribution from delta tau
    dz1 = z_b - z_a
    dr1 = r_b - r_a
    tau1 = dz1 / dr1
    dz2 = z_c - z_a
    dr2 = r_c - r_a
    tau2 = dz2 / dr2
    theta1 = atan2(dz1, dr1)
    theta2 = atan2(dz2, dr2)
    average_tau = (tau1 + tau2) / 2
    average_theta = (theta1 + theta2) / 2
    delta_theta = theta1 - theta2
    residual = tau1 - tau2
    distance2 = residual**2 * inv_cov_delta_tau

    chi2 = distance1 + distance2
    return chi2, average_tau, average_theta, delta_theta



parser = argparse.ArgumentParser(description='track hit-pair simulator')
parser.add_argument('-i', '--inputDir', help='input directory containing network gpickle file')
# parser.add_argument('-o', '--outputDir', help='output directory to save metadata')
args = parser.parse_args()

# read in subgraph data
inputDir = args.inputDir
subgraph_path = "_subgraph.gpickle"
subGraphs = []
os.chdir(".")
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)
    subGraphs.append(sub)

# outputDir = args.outputDir

# header = ['kl_dist', 'chi2_dist', 'truth']
# with open(outputDir + 'KL_chi2_data.csv', 'w', encoding='UTF8', newline='') as f:
    # writer = csv.writer(f)
    # writer.writerow(header)

for index, subGraph in enumerate(subGraphs):
    
    print("Processing subgraph number: ", index)
    print("Number of nodes: ", len(subGraph.nodes()))

    for node in subGraph.nodes(data=True):
        
        node_num = node[0]
        node_attr = node[1]

        num_edges = query_node_degree_in_edges(subGraph, node_num)
        if num_edges <= 1: continue

        # convert attributes to arrays
        if 'updated_track_states' in node_attr:
            updated_track_states = node_attr["updated_track_states"]
            print("updated_track_states:")
            pprint.pprint(updated_track_states)



            # edge_connections = []
            # for connection in list(track_state_estimates.keys()):
            #     edge_tuple = (connection, node_num)
            #     edge_connections.append(edge_tuple)
            # edge_connections = np.array(edge_connections)

            # # edge_connections = np.array([tuple(connection) for connection in list(track_state_estimates.keys())])
            # edge_svs = np.array([component['joint_vector'] for component in track_state_estimates.values()])
            # edge_covs = np.array([component['joint_vector_covariance'] for component in track_state_estimates.values()])
            # edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 3, 3))
            # inv_covs = np.linalg.inv(edge_covs)
            
            # node_coords = node_attr['xyzr']
            # all_neighbours_coords = np.array([subGraph.nodes[neighbour_num]['xyzr'] for neighbour_num in track_state_estimates.keys()])
            # pairwise_distances, pairwise_distances_chi2, delta_a, delta_b, delta_tau, cov_elem_1, cov_elem_2, cov_elem_3, tau_average, theta_average, delta_thetas = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs, node_coords, all_neighbours_coords)

            # for i in range(len(cov_elem_1)):
            #     data = [cov_elem_1[i], cov_elem_2[i], cov_elem_3[i]]
            #     writer_b.writerow(data)

            # # compute truth edges
            # node_truth = node_attr['truth_particle']
            # node_volume_id = node_attr['volume_id']
            # node_in_volume_layer_id = node_attr['in_volume_layer_id']
            # for i in range(num_edges):
            #     for j in range(i):
            #         neighbor_node1 = edge_connections[i][0]
            #         neighbor_node2 = edge_connections[j][0]

            #         node1_truth = subGraph.nodes[neighbor_node1]['truth_particle']
            #         node2_truth = subGraph.nodes[neighbor_node2]['truth_particle']

            #         neighbour1_coords = subGraph.nodes[neighbor_node1]['xyzr']
            #         neighbour2_coords = subGraph.nodes[neighbor_node2]['xyzr']
            #         z_a, r_a = node_coords[2], node_coords[3]
            #         z_b, r_b = neighbour1_coords[2], neighbour1_coords[3]
            #         z_c, r_c = neighbour2_coords[2], neighbour2_coords[3]
            #         tau1 = (z_b - z_a)/(r_b - r_a)
            #         tau2 = (z_c - z_a)/(r_c - r_a)

            #         truth = 0
            #         # pairwise between edge 1 and edge 2
            #         if (node_truth == node1_truth) and (node1_truth == node2_truth) and (node_truth == node2_truth):
            #             truth = 1   # truth_edge_pairs[i][j] = 1
            #         data = [pairwise_distances[i][j], pairwise_distances_chi2[i][j], delta_a[i][j], delta_b[i][j], delta_tau[i][j], tau_average[i][j], theta_average[i][j], delta_thetas[i][j], xy_emp_var, zr_emp_var, xy_theta_var, zr_theta_var, truth, node_volume_id, node_in_volume_layer_id, num_edges, tau1, tau2]
            #         writer.writerow(data)

