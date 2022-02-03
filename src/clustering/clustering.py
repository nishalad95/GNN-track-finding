import os, glob
import numpy as np
import networkx as nx
import argparse
from utils.utils import *
from modules.GNN_Measurement import *
from modules.HitPairPredictor import *
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

def calc_dist_to_merged_state(num_edges, edge_svs, edge_covs, inv_covs, merged_mean, merged_cov, merged_inv_cov):
    distances = []
    for i in range(num_edges):
        distance = KLDistance(edge_svs[i], edge_covs[i], inv_covs[i], merged_mean, merged_cov, merged_inv_cov)
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

def reset_reactivate(subGraphs, sigma0):
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
    
    subGraphs = compute_track_state_estimates(reset_subGraphs, sigma0)
    initialize_edge_activation(subGraphs)
    compute_prior_probabilities(subGraphs, 'track_state_estimates')
    compute_mixture_weights(subGraphs)

    return subGraphs


def cluster(inputDir, outputDir, track_state_key, KL_lut, sigma0, reactivate):

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

    # brute force approach to reset remaining network
    if reactivate:
        print("Resetting & reactivating all edges in remaining network")
        subGraphs = reset_reactivate(subGraphs, sigma0)


    # clustering on edges using KL-distance threshold
    perc_correct_outliers_detected = 0
    total_outliers = 0
    for subGraph in subGraphs:

        edges_to_deactivate = []
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

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
                merged_mean, merged_cov, merged_inv_cov = merge_states(edge_svs[idx[0]], inv_covs[idx[0]], edge_svs[idx[1]], inv_covs[idx[1]])
                merged_prior = priors[idx[0]] + priors[idx[1]]

                # update variables, keep the merged state information at the end
                edge_svs = np.delete(edge_svs, idx, axis=0)
                edge_covs = np.delete(edge_covs, idx, axis=0)
                inv_covs = np.delete(inv_covs, idx, axis=0)
                priors = np.delete(priors, idx)
                neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx, axis=0)
                # print("check neighbours to deactivate:", neighbors_to_deactivate)
                num_edges = edge_svs.shape[0]

                # calc distances to the merged state
                dist_to_merged_state = calc_dist_to_merged_state(num_edges, edge_svs, edge_covs, inv_covs, 
                                                                    merged_mean, merged_cov, merged_inv_cov)
                smallest_dist, idx = get_smallest_dist_idx(dist_to_merged_state)
                
                
                # carry on merging one by one state, check for smallest distance
                # if smallest distance is less than KL threshold, then merge
                # recalc distances to merged state
                while smallest_dist < KL_thres:
                    # merge states
                    merged_mean, merged_cov, merged_inv_cov = merge_states(edge_svs[idx], inv_covs[idx], merged_mean, merged_inv_cov)
                    merged_prior = priors[idx] + merged_prior

                    # update variables, keep the merged state at the end
                    edge_svs = np.delete(edge_svs, idx, axis=0)
                    edge_covs = np.delete(edge_covs, idx, axis=0)
                    inv_covs = np.delete(inv_covs, idx, axis=0)
                    priors = np.delete(priors, idx)
                    neighbors_to_deactivate = np.delete(neighbors_to_deactivate, idx, axis=0)
                    # print("check neighbours to deactivate:", neighbors_to_deactivate)
                    num_edges = edge_svs.shape[0]

                    # if all edges have merged, break the loop
                    if len(neighbors_to_deactivate) == 0: break

                    # calc distances to the merged state
                    dist_to_merged_state = calc_dist_to_merged_state(num_edges, edge_svs, edge_covs, inv_covs, 
                                                                        merged_mean, merged_cov, merged_inv_cov)
                    smallest_dist, idx = get_smallest_dist_idx(dist_to_merged_state)


                # store merged state as a node attribute
                # print("End of edge clusterising, saving merged state as node attribute")
                subGraph.nodes[node_num][MERGED_STATE] = merged_mean
                subGraph.nodes[node_num][MERGED_COVARIANCE] = merged_cov
                subGraph.nodes[node_num][MERGED_PRIOR] = merged_prior

                # print("Outliers found:", neighbors_to_deactivate)

                if len(neighbors_to_deactivate) > 0:
                    for neighbour_num in neighbors_to_deactivate:
                        # in edges, from neighbour to node 
                        # don't want to receive messages from the neighbour as the
                        # track state estimate given by that neighbour made no sense
                        edges_to_deactivate.append((neighbour_num, node_num))


            else:
                # all edges are incompatible
                print("No clusters found for node num: ", node_num)

        # simultaneous deactivation of outlier edge connections
        print("Deactivating outlier edges...", edges_to_deactivate)
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

    print("numerator:", perc_correct_outliers_detected, "denominator:", total_outliers)
    if total_outliers != 0:
        perc_correct_outliers_detected = (perc_correct_outliers_detected / total_outliers) * 100
        print("Percentage of correct outliers detected:", perc_correct_outliers_detected)


    # reweight the mixture based on inward active edges
    compute_mixture_weights(subGraphs)
    # compute priors for a node based on inward edges
    compute_prior_probabilities(subGraphs, TRACK_STATE_KEY)
  
    title = "Filtered Graph outlier edge removal using clustering with KL distance measure"
    plot_save_subgraphs(subGraphs, outputDir, title)
    plot_subgraphs_merged_state(subGraphs, outputDir, title)

    for i, s in enumerate(subGraphs):
        print("-------------------")
        print("SUBGRAPH " + str(i))
        print("-------------------")
        print("EDGE DATA:")
        for connection in s.edges.data():
            print(connection)
        print("-------------------")
        for node in s.nodes(data=True):
            pprint.pprint(node)
        print("--------------------")

    

def main():

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-o', '--output', help='output directory to save remaining network & track candidates')
    parser.add_argument('-d', '--dict', help='dictionary of track state estimates to use')
    parser.add_argument('-l', '--lut', help='lut file for KL distance acceptance region')
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-r', '--reactivateall', default=False, type=bool)
    args = parser.parse_args()

    inputDir = args.input
    outputDir = args.output
    track_states_key = args.dict
    KL_lut = args.lut
    sigma0 = float(args.error)
    reactivate = args.reactivateall

    cluster(inputDir, outputDir, track_states_key, KL_lut, sigma0, reactivate)



if __name__ == "__main__":
    main()