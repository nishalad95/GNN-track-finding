import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import argparse
from utils import *
from KL_calibration import compute_track_recon_eff

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
            pairwise_distances[j][i] = distance
    return pairwise_distances


def cluster(inputDir, outputDir, track_state_estimates, KL_thres):

    # variable names
    subgraph_path = "_subgraph.gpickle"
    TRACK_STATE_ESTIMATES = track_state_estimates
    EDGE_STATE_VECTOR = "edge_state_vector"
    EDGE_COVARIANCE = "edge_covariance"
    MASKED_EDGES = "masked_edges"
    MERGED_STATE = "merged_state"
    MERGED_COVARIANCE = "merged_cov"
    all_merged = np.array([-1])


    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    # TODO: change all occurences of 'while' to 'for' and use glob in other files when reading in subgraph data
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    # k-means clustering on edges using KL-distance threshold
    for subGraph in subGraphs:

        for node in subGraph.nodes(data=True):
            
            num_edges = node[1]['degree']
            if num_edges <= 2: continue
            masked_edges = np.zeros(num_edges)

            # convert graph attributes to arrays
            track_state_estimates = node[1][TRACK_STATE_ESTIMATES]
            node_state_estimates = pd.DataFrame(track_state_estimates)
            orig_connected_node_nums = node_state_estimates.columns.values
            connected_node_nums = node_state_estimates.columns.values
            edge_svs = np.vstack(node_state_estimates.loc[EDGE_STATE_VECTOR].to_numpy())
            edge_covs = np.vstack(node_state_estimates.loc[EDGE_COVARIANCE].to_numpy())
            edge_covs = np.reshape(edge_covs[:, :, np.newaxis], (num_edges, 2, 2))
            inv_covs = np.linalg.inv(edge_covs)
            
            # print("ORIGINAL VALUES")
            # print("node num:\n", node)
            # print("degree before:", subGraph.degree([node[0]]))

            # print("connected_node_nums\n", connected_node_nums)
            # print("num edges:", num_edges)

            # print("ORIGINAL VALUES")
            # print("edge_sv\n", edge_svs)

            # calculate pairwise distances between edge state vectors
            pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)

            # find the smallest distance & keep track of masked edges
            nonzero_dist = pairwise_distances[np.nonzero(pairwise_distances)]
            smallest_dist = np.min(nonzero_dist)
            idx, _ = np.where(pairwise_distances==smallest_dist) # idx[0] & idx[1] indicates edge indices with smallest distance
            masked_edges = np.empty(shape=(0, 0), dtype=int)

            # perform clustering
            if smallest_dist < KL_thres:
                while smallest_dist < KL_thres:
                    # merge states
                    merged_mean, merged_cov, merged_inv_cov = merge_states(edge_svs[idx[0]], inv_covs[idx[0]], 
                                                                            edge_svs[idx[1]], inv_covs[idx[1]])
                    # update variables
                    edge_svs = np.delete(edge_svs, idx, axis=0)
                    edge_covs = np.delete(edge_covs, idx, axis=0)
                    inv_covs = np.delete(inv_covs, idx, axis=0)
                    edge_svs = np.append(edge_svs, merged_mean.reshape(-1,2), axis=0)
                    edge_covs = np.append(edge_covs, merged_cov.reshape(-1,2,2), axis=0)
                    inv_covs = np.append(inv_covs, merged_inv_cov.reshape(-1,2,2), axis=0)
                    num_edges = edge_svs.shape[0]

                    # keep track of masked edges
                    masked_edges = np.append(masked_edges, connected_node_nums[idx])
                    # print("Masked edges: ", masked_edges)
                    subGraph.nodes[node[0]][MASKED_EDGES] = masked_edges
                    connected_node_nums = np.delete(connected_node_nums, idx, axis=0)
                    connected_node_nums = np.append(connected_node_nums, -1)
                    
                    # store merged state as a node attribute
                    subGraph.nodes[node[0]][MERGED_STATE] = merged_mean
                    subGraph.nodes[node[0]][MERGED_COVARIANCE] = merged_cov
                    # print("node num:\n", node)
                    
                    # if all edges have collapsed, break the loop
                    if (connected_node_nums == all_merged).all(): break
                    
                    # recalculate pairwise distances & find smallest distance
                    pairwise_distances = calc_pairwise_distances(num_edges, edge_svs, edge_covs, inv_covs)
                    nonzero_dist = pairwise_distances[np.nonzero(pairwise_distances)]
                    smallest_dist = np.min(nonzero_dist)
                    idx, _ = np.where(pairwise_distances==smallest_dist)
                
                # print("END OF CLUSTERING")

                # remove any outlier edges from node attributes
                outliers = np.setdiff1d(orig_connected_node_nums, masked_edges)
                # print("outlier edges:", outliers)
                # print("subgraph edges: \n", subGraph.edges())
                if len(outliers) > 0:
                    for outlier in outliers: 
                        track_state_estimates.pop(outlier) # remove attribute
                        subGraph.remove_edges_from([(node[0], outlier), (outlier, node[0])]) # remove bidirectional edge
                    node[1][TRACK_STATE_ESTIMATES] = track_state_estimates  # update track state estimates
                    subGraph.nodes[node[0]]['degree'] = int(subGraph.degree(node[0]) / 2)   # update node attribute
            # else:
            #     # all edges are incompatible
            #     print("NO CLUSTERS FOUND")

            # print("AFTER CLUSTERING VALUES")
            # print("node num:\n", node)
            # print("degree after:", subGraph.degree(node[0]))  


    # Identify subgraphs by running CCA & updating track state estimates, plot & save
    run_cca(subGraphs, outputDir)

    # calibrate the KL threshold for clustering using MC truth
    efficiency, score = compute_track_recon_eff(outputDir)
    return efficiency, score
    

def main():

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-o', '--output', help='output directory to save remaining network & track candidates')
    parser.add_argument('-d', '--dict', help='dictionary of track state estimates to use')
    args = parser.parse_args()

    inputDir = args.input
    outputDir = args.output
    track_state_estimates = args.dict
    KL_thres = 50
    efficiency = 0

    while efficiency < 0.6:
        for f in glob.glob(outputDir + "*"):
            os.remove(f)
        efficiency, score = cluster(inputDir, outputDir, track_state_estimates, KL_thres)
        print("EFF:", efficiency, "score", score, "KL_thres", KL_thres)
        KL_thres *= 0.75




if __name__ == "__main__":
    main()