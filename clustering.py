import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from utils.utils import *
import pprint
import math

# KL distance - takes into account covariances between the components 
# If you were to use simple Euclidean distance, cov not taken into account
def KLDistance(mean1, cov1, inv1, mean2, cov2, inv2):
    trace = np.trace((cov1 - cov2) * (inv2 - inv1))
    return trace + (mean1 - mean2).T.dot(inv1 + inv2).dot(mean1 - mean2)

def merge_states(mean1, inv1, mean2, inv2):
    sum_inv_covs = inv1 + inv2
    merged_cov = np.linalg.inv(sum_inv_covs)
    merged_mean = inv1.dot(mean1) + inv2.dot(mean2)
    merged_mean = merged_cov.dot(merged_mean)
    return merged_mean, merged_cov, sum_inv_covs

def calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs):
    pairwise_distances = np.zeros(shape=(num_edges, num_edges))
    for i in range(num_edges):
        for j in range(i):
            distance = KLDistance(edge_svs[i], edge_covs[i], inv_covs[i], edge_svs[j], edge_covs[j], inv_covs[j])
            pairwise_distances[i][j] = distance
    return pairwise_distances

def get_smallest_dist_idx(pairwise_distances):
    nonzero_dist = pairwise_distances[np.nonzero(pairwise_distances)]
    smallest_dist = np.min(nonzero_dist)
    # row[0] & column[0] indicates neighbor_node indices with smallest distance
    row, column = np.where(pairwise_distances==smallest_dist)
    idx = np.concatenate((row, column), axis=None)
    return smallest_dist, idx

def load_lut(KL_lut):
    mapping = {}
    lut_file = open(KL_lut, "r")
    for line in lut_file.readlines():
        elements = line.split(" ")
        feature = float(elements[0]) 
        KL_thres = float(elements[2].split("\n")[0])
        mapping[feature] = KL_thres
    return mapping

def get_KL_upper_threshold(empvar_feature, distance, mapping):
    base = 0.05
    feature = math.ceil(float(empvar_feature)/base) - 1

    if float(feature) in mapping.keys():
        KL_thres = mapping[feature]
        if distance <= KL_thres: return KL_thres
        else: return 0
    return 0


def cluster(inputDir, outputDir, track_state_key, KL_lut):

    # variable names
    subgraph_path = "_subgraph.gpickle"
    TRACK_STATE_KEY = track_state_key
    EMPIRICAL_MEAN_VAR = "edge_gradient_mean_var"
    EDGE_STATE_VECTOR = "edge_state_vector"
    EDGE_COV = "edge_covariance"
    PRIOR = "prior"
    MERGED_STATE = "merged_state"
    MERGED_COVARIANCE = "merged_cov"
    MERGED_PRIOR = "merged_prior"

    # load predefined LUT: empirical variance: {upper bound emp var bin: KL_dist upper bound threshold}
    mapping = load_lut(KL_lut)
    
    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    perc_correct_outliers_detected = 0
    total_outliers = 0
    # clustering on edges using KL-distance threshold
    for subGraph in subGraphs:

        edges_to_deactivate = []
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]
            print("Processing node:", node_num)

            empvar = node_attr[EMPIRICAL_MEAN_VAR][1]
            num_edges = query_node_degree_in_edges(subGraph, node_num) # node degree is dynamical between iterations, only check active edges
            if num_edges <= 2: continue

            # convert attributes to arrays
            track_state_estimates = node_attr[TRACK_STATE_KEY]
            neighbors_to_deactivate = np.array([connection for connection in track_state_estimates.keys()])
            edge_svs = np.array([component[EDGE_STATE_VECTOR] for component in track_state_estimates.values()])
            edge_covs = np.array([component[EDGE_COV] for component in track_state_estimates.values()])
            edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 2, 2))
            inv_covs = np.linalg.inv(edge_covs)
            priors = np.array([component[PRIOR] for component in track_state_estimates.values()])


            # calculate pairwise distances between edge state vectors, find smallest distance & keep track of merged states
            pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
            smallest_dist, idx = get_smallest_dist_idx(pairwise_distances) #[row_idx, column_idx]

            # perform clustering, query LUT with degree/empvar & smallest pairwise distance
            KL_thres = get_KL_upper_threshold(empvar, smallest_dist, mapping)
            if smallest_dist < KL_thres:

                # merge states
                print("MERGING STATES & CLUSTERING")
                merged_mean, merged_cov, merged_inv_cov = merge_states(edge_svs[idx[0]], inv_covs[idx[0]], edge_svs[idx[1]], inv_covs[idx[1]])
                merged_prior = priors[idx[0]] + priors[idx[1]]

                # update variables, keep the merged state information at the end
                edge_svs = np.delete(edge_svs, idx, axis=0)
                edge_covs = np.delete(edge_covs, idx, axis=0)
                inv_covs = np.delete(inv_covs, idx, axis=0)
                priors = np.delete(priors, idx)
                neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx, axis=0)
                edge_svs = np.append(edge_svs, merged_mean.reshape(-1,2), axis=0)
                edge_covs = np.append(edge_covs, merged_cov.reshape(-1,2,2), axis=0)
                inv_covs = np.append(inv_covs, merged_inv_cov.reshape(-1,2,2), axis=0)
                priors = np.append(priors, merged_prior)
                num_edges = edge_svs.shape[0]

                # recalculate pairwise distances between edge state vectors, find smallest distance & keep track of merged states
                pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
                smallest_dist, idx = get_smallest_dist_idx(pairwise_distances)
                # if the merged state wasn't found in the smallest pairwise distance pair - new cluster - leave for further iterations
                if (idx[1] != len(pairwise_distances) - 1):
                    print("2nd cluster found! Ending clusterization here...")
                else:
                    while smallest_dist < KL_thres:
                        # merge states
                        print("continuing to merge...")
                        merged_mean, merged_cov, merged_inv_cov = merge_states(edge_svs[idx[0]], inv_covs[idx[0]], merged_mean, merged_inv_cov)
                        merged_prior = priors[idx[0]] + merged_prior
                        
                        # update variables, keep the merged state at the end
                        edge_svs = np.delete(edge_svs, idx, axis=0)
                        edge_covs = np.delete(edge_covs, idx, axis=0)
                        inv_covs = np.delete(inv_covs, idx, axis=0)
                        priors = np.delete(priors, idx)
                        neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx[0], axis=0)
                        edge_svs = np.append(edge_svs, merged_mean.reshape(-1,2), axis=0)
                        edge_covs = np.append(edge_covs, merged_cov.reshape(-1,2,2), axis=0)
                        inv_covs = np.append(inv_covs, merged_inv_cov.reshape(-1,2,2), axis=0)
                        priors = np.append(priors, merged_prior)
                        num_edges = edge_svs.shape[0]

                        # if all edges have merged, break the loop
                        if len(neighbors_to_deactivate) == 0: break

                        # recalculate pairwise distances & find smallest distance
                        pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
                        smallest_dist, idx = get_smallest_dist_idx(pairwise_distances)
                        # if the merged state wasn't found in the smallest pairwise distance pair - new cluster - leave for further iterations
                        if (idx[1] != len(pairwise_distances) - 1):
                            print("HERE 2nd cluster found! Ending clusterization here...")
                            break
                
                # store merged state as a node attribute
                print("End of edge clusterising, saving merged state as node attribute")
                subGraph.nodes[node_num][MERGED_STATE] = merged_mean
                subGraph.nodes[node_num][MERGED_COVARIANCE] = merged_cov
                subGraph.nodes[node_num][MERGED_PRIOR] = merged_prior


                # record which edge connections were identified as outliers
                if len(neighbors_to_deactivate) > 0:
                    for neighbour_num in neighbors_to_deactivate:
                        # in edges, from neighbour to node 
                        # don't want to receive messages from the neighbour as the
                        # track state estimate given by that neighbour made no sense
                        # edges_to_deactivate.append((neighbour_num, node_num))
                        edges_to_deactivate.append((node_num, neighbour_num))

            else:
                # all edges are incompatible
                print("NO CLUSTERS FOUND")

        # simultaneous deactivation of outlier edge connections
        print("Deactivating outlier edges...", edges_to_deactivate)
        if len(edges_to_deactivate) > 0:
            for edge in edges_to_deactivate:
                node_num = edge[0]
                neighbour_num = edge[1]
                attrs = {(node_num, neighbour_num): {"activated": 0}}
                nx.set_edge_attributes(subGraph, attrs)
                
                node_truth_particle = subGraph.nodes[node_num]['truth_particle']
                neighbour_truth_particle = subGraph.nodes[neighbour_num]['truth_particle']
                if node_truth_particle != neighbour_truth_particle:
                    perc_correct_outliers_detected += 1
            total_outliers += len(edges_to_deactivate)


    print("numerator:", perc_correct_outliers_detected)
    print("denominator:", total_outliers)
    if total_outliers != 0:
        perc_correct_outliers_detected = (perc_correct_outliers_detected / total_outliers) * 100
    print("PERC:", perc_correct_outliers_detected)


    # identify subgraphs
    # subGraphs = run_cca(subGraphs)
    # update network state: recompute priors based on active edges

    # compute priors for a node based on inward edges??
    compute_prior_probabilities(subGraphs, TRACK_STATE_KEY)
    title = "Filtered Graph outlier edge removal using clustering with KL distance measure"
    plot_save_subgraphs(subGraphs, outputDir, title)
    plot_subgraphs_merged_state(subGraphs, outputDir, title)

    # calibrate the KL threshold for clustering using MC truth
    # efficiency, score = compute_track_recon_eff(outputDir)
    # return efficiency, score

    # for i, s in enumerate(subGraphs):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")
    #     print("EDGE DATA:", s.edges.data(), "\n")

    

def main():

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-o', '--output', help='output directory to save remaining network & track candidates')
    parser.add_argument('-d', '--dict', help='dictionary of track state estimates to use')
    parser.add_argument('-l', '--lut', help='lut file for KL distance acceptance region')
    args = parser.parse_args()

    inputDir = args.input
    outputDir = args.output
    track_states_key = args.dict
    KL_lut = args.lut
    # efficiency = 0

    cluster(inputDir, outputDir, track_states_key, KL_lut)

    # while efficiency < 0.6:
    #     for f in glob.glob(outputDir + "*"):
    #         os.remove(f)
    #     efficiency, score = cluster(inputDir, outputDir, track_state_estimates, KL_thres)
    #     print("EFF:", efficiency, "score", score, "KL_thres", KL_thres)
    #     KL_thres *= 0.75


if __name__ == "__main__":
    main()