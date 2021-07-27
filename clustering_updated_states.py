import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from utils.utils import *
from KL_calibration import compute_track_recon_eff
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
            # pairwise_distances[j][i] = distance #TODO: don't need this line?
    return pairwise_distances

def get_smallest_dist_idx(pairwise_distances):
    nonzero_dist = pairwise_distances[np.nonzero(pairwise_distances)]
    print("NONZERO DISTANCES", nonzero_dist)
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


def cluster(inputDir, outputDir, track_state_estimates, KL_lut):

    # variable names
    subgraph_path = "_subgraph.gpickle"
    TRACK_STATE_ESTIMATES = 'updated_track_states'
    MERGED_STATE = "merged_state"
    MERGED_COVARIANCE = "merged_cov"
    MERGED_PRIOR = "merged_prior"

    # load LUT
    # degree of node: {degree: KL_dist upper bound threshold}
    # empirical variance: {upper bound emp var bin: KL_dist upper bound threshold}
    mapping = load_lut(KL_lut)
    
    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    # TODO: handling of more than 1 cluster?
    # k-means clustering on edges using KL-distance threshold
    for subGraph in subGraphs:

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            print("\nPROCESSING NODE NUM:", node_num)

            # KL_dist LUT variables
            empvar = node_attr['edge_gradient_mean_var'][1]
            # num_edges = node_attr['degree']
            
            pprint.pprint(node_attr)

            if 'updated_track_states' not in node_attr.keys(): continue
            if len(node_attr['updated_track_states']) == 1:
                print("Only 1 updated track state, clustering cannot be performed")
                print("leaving for further iterations")
                continue

            # convert attributes to arrays
            track_state_estimates = node_attr[TRACK_STATE_ESTIMATES]
            num_edges = len(track_state_estimates)
            # neighbor_nodes = np.array([connection[0] for connection in track_state_estimates.keys()])
            neighbors_to_deactivate = np.array([connection[0] for connection in track_state_estimates.keys()])
            # edge_weights = [connection[2]['activated'] for connection in subGraph.in_edges(node_num, data=True)]  # in_edges
            edge_svs = np.array([component['edge_state_vector'] for component in track_state_estimates.values()])
            edge_covs = np.array([component['edge_covariance'] for component in track_state_estimates.values()])
            edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 2, 2))
            inv_covs = np.linalg.inv(edge_covs)
            priors = np.array([component['prior'] for component in track_state_estimates.values()])

            # calculate pairwise distances between edge state vectors, find smallest distance & keep track of merged states
            pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
            print("PAIRWISE DISTANCES", pairwise_distances)
            smallest_dist, idx = get_smallest_dist_idx(pairwise_distances) #[row_idx, column_idx]
            # perform clustering, query LUT with degree/empvar & smallest pairwise distance
            KL_thres = get_KL_upper_threshold(empvar, smallest_dist, mapping)
            if smallest_dist < KL_thres:

                print("MERGING STATES & CLUSTERING")
                # merge states
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

                # if all edges have merged, break the loop
                if len(neighbors_to_deactivate) == 0: break

                # recalculate pairwise distances between edge state vectors, find smallest distance & keep track of merged states
                pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
                print("neighbours to deactivate:", neighbors_to_deactivate)
                print("PAIRWISE DISTANCES 2", pairwise_distances)
                smallest_dist, idx = get_smallest_dist_idx(pairwise_distances)
                # if the merged state wasn't found in the smallest pairwise distance pair - new cluster - leave for further iterations
                if (idx[1] != len(pairwise_distances) - 1):
                    print("2nd cluster found! Ending clusterization here...")
                else:
                    while smallest_dist < KL_thres:
                        # merge states
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

                # deactivate neighbor edges identified as outliers
                print("Neighbors to deactivate: ", neighbors_to_deactivate)
                if len(neighbors_to_deactivate) > 0:
                    print("Deactivating outlier edges...")
                    for n in neighbors_to_deactivate:
                        attrs = {(n, node_num): {"activated": 0}}
                        nx.set_edge_attributes(subGraph, attrs)

            else:
                # all edges are incompatible
                print("NO CLUSTERS FOUND")

        # check activation/deactivation of edges
        # print("--------------------")
        # print("EDGE DATA:", subGraph.edges.data(), "\n")
        # for node in subGraph.nodes(data=True):
            # pprint.pprint(node)



    # identify subgraphs
    # subGraphs = run_cca(subGraphs)
    # update network state: recompute priors based on active edges
    compute_prior_probabilities(subGraphs, 'updated_track_states')
    title = "Filtered Graph outlier edge removal using clustering with KL distance measure"
    plot_save_subgraphs(subGraphs, outputDir, title)
    plot_subgraphs_merged_state(subGraphs, outputDir, title)

    # printing node and edge attributes
    # for i, s in enumerate(subGraphs):
    #     if i<=1:
    #         print("-------------------")
    #         print("SUBGRAPH " + str(i))
    #         for node in s.nodes(data=True):
    #             if node[0] <= 100:
    #                 pprint.pprint(node)
    #         print("--------------------")
    #         print("EDGE DATA:", s.edges.data(), "\n")

    # calibrate the KL threshold for clustering using MC truth
    # efficiency, score = compute_track_recon_eff(outputDir)
    # return efficiency, score
    

def main():

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-o', '--output', help='output directory to save remaining network & track candidates')
    parser.add_argument('-d', '--dict', help='dictionary of track state estimates to use')
    parser.add_argument('-l', '--lut', help='lut file for KL distance acceptance region')
    args = parser.parse_args()

    inputDir = args.input
    outputDir = args.output
    track_state_estimates = args.dict
    KL_lut = args.lut
    # efficiency = 0

    cluster(inputDir, outputDir, track_state_estimates, KL_lut)

    # while efficiency < 0.6:
    #     for f in glob.glob(outputDir + "*"):
    #         os.remove(f)
    #     efficiency, score = cluster(inputDir, outputDir, track_state_estimates, KL_thres)
    #     print("EFF:", efficiency, "score", score, "KL_thres", KL_thres)
    #     KL_thres *= 0.75


if __name__ == "__main__":
    main()