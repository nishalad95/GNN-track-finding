import os, glob
import numpy as np
import networkx as nx
import argparse
from utilities import helper as h
import pprint
import math
import time
import csv

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
    tau1 = (z_b - z_a)/(r_b - r_a)
    tau2 = (z_c - z_a)/(r_c - r_a)
    residual = tau1 - tau2
    distance2 = residual**2 * inv_cov_delta_tau

    chi2 = distance1 + distance2
    return chi2

def calc_pairwise_distances_chi2(num_edges, edge_svs, edge_covs, node_coords, neighbour_coords):
    pairwise_distances_chi2 = np.zeros(shape=(num_edges, num_edges))
    for i in range(num_edges):
        for j in range(i):
            distance_chi2 = mahalanobis_distance(edge_svs[i], edge_covs[i], edge_svs[j], edge_covs[j], node_coords, neighbour_coords[i], neighbour_coords[j])
            pairwise_distances_chi2[i][j] = distance_chi2
    return pairwise_distances_chi2

# KL distance - takes into account covariances between the components 
# # If you were to use simple Euclidean distance, cov not taken into account
def KLDistance(mean1, cov1, mean2, cov2):
    inv1 = np.linalg.inv(cov1)
    inv2 = np.linalg.inv(cov2)
    trace = np.trace((cov1 - cov2) * (inv2 - inv1))
    return trace + (mean1 - mean2).T.dot(inv1 + inv2).dot(mean1 - mean2)

# inverse variance-weighting: multivariate case https://en.wikipedia.org/wiki/Inverse-variance_weighting#Multivariate_Case
def merge_states(mean1, cov1, mean2, cov2):
    inv1 = np.linalg.inv(cov1)
    inv2 = np.linalg.inv(cov2)
    sum_inv_covs = inv1 + inv2
    merged_cov = np.linalg.inv(sum_inv_covs)
    merged_mean = inv1.dot(mean1) + inv2.dot(mean2)
    merged_mean = merged_cov.dot(merged_mean)
    merged_inv_cov = np.linalg.inv(merged_cov)
    return merged_mean, merged_cov

def calc_dist_to_merged_state(num_edges, edge_svs, edge_covs, merged_mean, merged_cov):
    distances = []
    for i in range(num_edges):
        distance = KLDistance(edge_svs[i], edge_covs[i], merged_mean, merged_cov)
        distances.append(distance)
    return distances

def get_smallest_dist_idx(distances):
    if isinstance(distances, list):
        smallest_dist = np.min(distances)
        idx = distances.index(smallest_dist)
    else:
        nonzero_dist = distances[np.nonzero(distances)]
        smallest_dist = np.min(nonzero_dist)
        # row[0] & column[0] indicates neighbor_node indices with smallest distance
        row, column = np.where(distances==smallest_dist)
        idx = np.concatenate((row, column), axis=None)
    return smallest_dist, idx

def reset_reactivate(subGraphs):
    reset_subGraphs = []
    for subGraph in subGraphs:

        for (_,d) in subGraph.nodes(data=True):
            if "merged_state" in d.keys(): 
                del d["merged_state"]
                del d["merged_cov"]
                del d["merged_prior"]
            if "updated_track_states" in d.keys():
                del d["updated_track_states"]

        for component in nx.weakly_connected_components(subGraph):
            reset_subGraphs.append(subGraph.subgraph(component).copy())
    
    subGraphs = h.compute_track_state_estimates(reset_subGraphs)
    h.initialize_edge_activation(subGraphs)
    h.compute_prior_probabilities(subGraphs, 'track_state_estimates')
    h.compute_mixture_weights(subGraphs)

    return subGraphs


def cluster(inputDir, outputDir, track_state_key, chi2_threshold, KL_threshold, KL_lut, iteration_num, reactivate):
    # variable names
    subgraph_path = "_subgraph.gpickle"
    TRACK_STATE_KEY = track_state_key
    PRIOR = "prior"
    MERGED_STATE = "merged_state"
    MERGED_COVARIANCE = "merged_cov"
    MERGED_PRIOR = "merged_prior"
    
    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    # brute force approach to reset remaining network
    if reactivate:
        subGraphs = reset_reactivate(subGraphs)

    perc_correct_outliers_detected = 0
    total_outliers = 0
    nodes_with_merged_state = 0

    print("Total number of subgraphs to process in this iteration: ", len(subGraphs))
    total_number_of_nodes = 0
    total_number_of_edges = 0
    total_number_of_nodes_with_updated_state = 0

    for subGraph in subGraphs:
        total_number_of_nodes += len(subGraph.nodes())
        total_number_of_edges += len(subGraph.edges())

        edges_to_deactivate = []
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            if TRACK_STATE_KEY in node_attr.keys():
                total_number_of_nodes_with_updated_state += 1

                # if TRACK_STATE_KEY == "track_state_estimates":
                #     num_edges = h.query_node_degree_in_edges(subGraph, node_num) # node degree is dynamical between iterations, only check active edges
                # else:
                #     num_edges = len(node_attr[TRACK_STATE_KEY])

                track_state_estimates = node_attr[TRACK_STATE_KEY]
                num_edges = len(track_state_estimates)
                if (num_edges <= 2) or (num_edges >= 16): continue

                # convert attributes to arrays: "edge_state_vector" --> [a, b, c] used in merging states
                neighbors_to_deactivate = np.array([connection for connection in track_state_estimates.keys()])
                parabolic_edge_svs = np.array([component["edge_state_vector"] for component in track_state_estimates.values()])
                parabolic_edge_covs = np.array([component["edge_covariance"] for component in track_state_estimates.values()])
                parabolic_edge_covs = np.reshape(parabolic_edge_covs[:, :, np.newaxis], (num_edges, 3, 3))
                priors = np.array([component["prior"] for component in track_state_estimates.values()])

                # "joint_vector" --> [a, b, tau] used to cluster & merge states
                node_coords = node_attr['xyzr']
                all_neighbours_coords = np.array([value['xyzr'] for neighbour_num, value in track_state_estimates.items()])
                joint_edge_svs = np.array([component["joint_vector"] for component in track_state_estimates.values()])
                joint_edge_covs = np.array([component["joint_vector_covariance"] for component in track_state_estimates.values()])
                joint_edge_covs = np.reshape(joint_edge_covs[:, :, np.newaxis], (num_edges, 3, 3))
                # compute mahalanobis distances
                pairwise_distances_chi2 = calc_pairwise_distances_chi2(num_edges, joint_edge_svs, joint_edge_covs, node_coords, all_neighbours_coords)
                smallest_dist, idx = get_smallest_dist_idx(pairwise_distances_chi2) #[row_idx, column_idx]



                # # hyperparameter tuning:
                # # find out which pair of nodes had the smallest distance and find its ground truth info
                # n1, n2 = neighbors_to_deactivate[idx[0]], neighbors_to_deactivate[idx[1]]
                # n1_truth = subGraph.nodes[n1]['truth_particle']
                # n2_truth = subGraph.nodes[n2]['truth_particle']
                # if n1_truth != n2_truth:
                #     ground_truth = 0
                # else:
                #     ground_truth = 1
                # with open('chi2_dist_cluster_updates_states.csv', 'a') as c:
                #     writer = csv.writer(c)
                #     writer.writerow([smallest_dist, num_edges, ground_truth])
                


                # perform clustering
                if smallest_dist < chi2_threshold:

                    # merge parabolic states [a1, b1, c1] & [a2, b2, c2]
                    parabolic_merged_mean, parabolic_merged_cov = merge_states(parabolic_edge_svs[idx[0]], parabolic_edge_covs[idx[0]], parabolic_edge_svs[idx[1]], parabolic_edge_covs[idx[1]])
                    # merge joint states [a1, b1, tau1] & [a2, b2, tau2]
                    joint_merged_mean, joint_merged_cov = merge_states(joint_edge_svs[idx[0]], joint_edge_covs[idx[0]], joint_edge_svs[idx[1]], joint_edge_covs[idx[1]])
                    merged_prior = priors[idx[0]] + priors[idx[1]]

                    # update variables, keep the merged state information at the end
                    parabolic_edge_svs = np.delete(parabolic_edge_svs, idx, axis=0)
                    parabolic_edge_covs = np.delete(parabolic_edge_covs, idx, axis=0)
                    joint_edge_svs = np.delete(joint_edge_svs, idx, axis=0)
                    joint_edge_covs = np.delete(joint_edge_covs, idx, axis=0)
                    priors = np.delete(priors, idx)
                    neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx, axis=0)
                    num_edges = parabolic_edge_svs.shape[0]

                    # calc distances to the merged state
                    dist_to_merged_state = calc_dist_to_merged_state(num_edges, joint_edge_svs, joint_edge_covs, joint_merged_mean, joint_merged_cov)
                    smallest_dist, idx = get_smallest_dist_idx(dist_to_merged_state)
        
                    # hyperparameter tuning:
                    with open('kl_dist_cluster_updates_states.txt', 'a') as c:
                        c.write("%s\n" % smallest_dist)

                    # carry on merging one by one state, check for smallest distance
                    # if smallest distance is less than KL threshold, then merge
                    # recalc distances to merged state
                    while smallest_dist < KL_threshold:
                        # merge parabolic states [a_merged, b_merged, c_merged] & [a3, b3, c3]
                        parabolic_merged_mean, parabolic_merged_cov = merge_states(parabolic_edge_svs[idx], parabolic_edge_covs[idx], parabolic_merged_mean, parabolic_merged_cov)
                        # merge joint states [a_merged, b_merged, tau_merged] & [a3, b3, tau3]
                        joint_merged_mean, joint_merged_cov = merge_states(joint_edge_svs[idx], joint_edge_covs[idx], joint_merged_mean, joint_merged_cov)
                        merged_prior = priors[idx] + merged_prior

                        # update variables, keep the merged state at the end
                        parabolic_edge_svs = np.delete(parabolic_edge_svs, idx, axis=0)
                        parabolic_edge_covs = np.delete(parabolic_edge_covs, idx, axis=0)
                        joint_edge_svs = np.delete(joint_edge_svs, idx, axis=0)
                        joint_edge_covs = np.delete(joint_edge_covs, idx, axis=0)
                        priors = np.delete(priors, idx)
                        neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx, axis=0)
                        # print("check neighbours to deactivate:", neighbors_to_deactivate)
                        num_edges = parabolic_edge_svs.shape[0]

                        # if all edges have merged, break the loop
                        if len(neighbors_to_deactivate) == 0: break

                        # calc distances to the merged state
                        dist_to_merged_state = calc_dist_to_merged_state(num_edges, joint_edge_svs, joint_edge_covs, joint_merged_mean, joint_merged_cov)
                        smallest_dist, idx = get_smallest_dist_idx(dist_to_merged_state)

                    # store merged state as a node attribute
                    print("End of edge clusterising, saving merged state as node attribute")
                    subGraph.nodes[node_num][MERGED_STATE] = parabolic_merged_mean
                    subGraph.nodes[node_num][MERGED_COVARIANCE] = parabolic_merged_cov
                    subGraph.nodes[node_num][MERGED_PRIOR] = merged_prior
                    nodes_with_merged_state += 1

                    # append tuple of edge to deactivate
                    if len(neighbors_to_deactivate) > 0:
                        for neighbour_num in neighbors_to_deactivate:
                            # in edges, from neighbour to node 
                            # don't want to receive messages from the neighbour as the
                            # track state estimate given by that neighbour made no sense
                            edges_to_deactivate.append((neighbour_num, node_num))

                else:
                    # all edges are incompatible
                    print("No clusters found for node num: ", node_num, "vivl_id: ", node_attr['vivl_id'])
                    print("smallest chi2 distance: ", smallest_dist)

        # simultaneous deactivation of outlier edge connections
        # print("Deactivating outlier edges...", edges_to_deactivate)
        if len(edges_to_deactivate) > 0:
            for edge in edges_to_deactivate:
                neighbour_num = edge[0]
                node_num = edge[1]
                attrs = {(neighbour_num, node_num): {"activated": 0}}
                nx.set_edge_attributes(subGraph, attrs)

                node_truth_particle = subGraph.nodes[node_num]['truth_particle']
                neighbour_truth_particle = subGraph.nodes[neighbour_num]['truth_particle']
                if node_truth_particle != neighbour_truth_particle:
                    perc_correct_outliers_detected += 1
            total_outliers += len(edges_to_deactivate)
        
        # update the node degree as an attribute
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            degree = h.query_node_degree_in_edges(subGraph, node_num)
            subGraph.nodes[node_num]['degree'] = degree

    print("Statistics for this iteration:")
    print("Total number of nodes: ", total_number_of_nodes)
    print("Total number of edges: ", total_number_of_edges)
    print("Total number of nodes with updated states: ", total_number_of_nodes_with_updated_state)
    print("Total number of nodes with a new merged state: ", nodes_with_merged_state)

    print("numerator:", perc_correct_outliers_detected, "denominator:", total_outliers)
    if total_outliers != 0:
        perc_correct_outliers_detected = (perc_correct_outliers_detected / total_outliers) * 100
        print("Percentage of correct outliers detected:", perc_correct_outliers_detected)

    # reweight the mixture & calculate priors based on inward active edges
    if iteration_num == 1:
        h.compute_mixture_weights(subGraphs, TRACK_STATE_KEY)
    # else:
    #     h.reweight(subGraphs, 'updated_track_states')
    h.compute_prior_probabilities(subGraphs, TRACK_STATE_KEY)

    # title = "Filtered Graph outlier edge removal using clustering with KL distance measure"
    # h.plot_subgraphs(subGraphs, outputDir, node_labels=True, save_plot=True, title=title)
    # save networks
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)

    

def main():

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-o', '--output', help='output directory to save remaining network & track candidates')
    parser.add_argument('-d', '--dict', help='dictionary of track state estimates to use')
    parser.add_argument('-l', '--lut', help='lut file for KL distance acceptance region')
    parser.add_argument('-c', '--chi2', help='chi2 distance threshold')
    parser.add_argument('-k', '--kl', help='kl distance threshold')
    parser.add_argument('-t', '--iteration', help="iteration number")
    parser.add_argument('-r', '--reactivateall', default=False, type=bool)
    args = parser.parse_args()

    inputDir = args.input
    outputDir = args.output
    track_states_key = args.dict
    KL_lut = args.lut
    chi2_threshold = float(args.chi2)
    KL_threshold = float(args.kl)
    iteration_num = int(args.iteration)
    reactivate = args.reactivateall

    start = time.time()

    cluster(inputDir, outputDir, track_states_key, chi2_threshold, KL_threshold, KL_lut, iteration_num, reactivate)

    end = time.time()
    total_time = end - start
    print("Time taken in clustering.py cluster function: "+ str(total_time))

if __name__ == "__main__":
    main()