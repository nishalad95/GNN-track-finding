import os, glob
import numpy as np
import math
import networkx as nx
from filterpy.kalman import KalmanFilter
from filterpy import common
import argparse
from utils.utils import *
import pprint


def calculate_norm_factor_lr_layers(subGraph, node, updated_track_states):
    
    # split track state estimates into LHS & RHS
    node_x_layer = node[1]['GNN_Measurement'].x
    # track_state_estimates = node[1]['track_state_estimates']
    left_nodes, right_nodes = [], []
    left_coords, right_coords = [], []
    
    print("EDGE DATA:", subGraph.edges.data(), "\n")
    print("updated track state estimates:\n")
    pprint.pprint(updated_track_states)

    for edge_tuple, v in updated_track_states.items():
        # neighbour_x_layer = v['coord_Measurement'][0]

        central_node_num = edge_tuple[1]
        neighbour_node_num = edge_tuple[0]

        neighbour_x_layer = subGraph.nodes[neighbour_node_num]['GNN_Measurement'].x

        # print("edge_tuple", edge_tuple)
        # print("central_node_num,", central_node_num)
        # print("neighbour_node_num,", neighbour_node_num)

        # only calculate for activated edges
        # if subGraph[neighbour_node_num][central_node_num]['activated'] == 1:
        if neighbour_x_layer < node_x_layer: 
            left_nodes.append(edge_tuple)
            left_coords.append(neighbour_x_layer)
        else: 
            right_nodes.append(edge_tuple)
            right_coords.append(neighbour_x_layer)
    
    # store norm factor as node attribute
    left_norm = len(list(set(left_coords)))
    for lf_tuple in left_nodes:
        updated_track_states[lf_tuple]['lr_layer_norm'] = 1 / left_norm

    right_norm = len(list(set(right_coords)))
    for rn_tuple in right_nodes:
        updated_track_states[rn_tuple]['lr_layer_norm'] = 1 / right_norm

    # print("NODE:\n")
    # pprint.pprint(node)
    # print("TRACK STATE EST\n", track_state_estimates)


def perform_KF_updated_state(subGraph, F, H, inv_S, extrp_state, extrp_cov, chi2, sigma0, neighbour_num, node_num):
    # calc beta: measurement likelihood
    norm_factor = math.pow(2 * math.pi * np.abs(inv_S), -0.5)
    likelihood = norm_factor * np.exp(-0.5 * chi2)
    
    # initialize KF
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = extrp_state  # X state vector
    f.F = F # F state transition matrix
    f.H = H # H measurement matrix
    f.P = extrp_cov
    f.R = sigma0**2
    f.Q = 0.

    # perform KF update & save data
    f.predict()
    f.update(extrp_state[0])
    updated_state, updated_cov = f.x_post, f.P_post

    updated_state_dict = {  'edge_state_vector': updated_state, 
                            'edge_covariance': updated_cov,
                            'likelihood': likelihood,
                            'prior': subGraph.nodes[node_num]['track_state_estimates'][(neighbour_num, node_num)]['prior'],
                            # this is the previous mixture weight at this point, it will get updated later
                            'mixture_weight': subGraph.nodes[node_num]['track_state_estimates'][(neighbour_num, node_num)]['mixture_weight'],
                        }
    return updated_state_dict




def extrapolate(subGraphs, reweight_threshold, outputDir):

    H = np.array([[1.,0.]])

    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            print("\nProcessing node num: ", node[0])
            node_num = node[0]

            # extrapolate each neighbour node state & perform KF update
            updated_track_states = {}
            node_x = node[1]['GNN_Measurement'].x
            node_y = node[1]['GNN_Measurement'].y
            for neighbour_num in subGraph.neighbors(node_num):

                print("neighbour_num", neighbour_num)
                
                # print("neighbour", neighbour)
                neighbour_dict = subGraph.nodes[neighbour_num]
                
                if 'merged_state' in neighbour_dict:
                    # clustering was executed for that neighbour node
                    print("merge state exists...")
                    
                    # get merged state vector from neighbouring node
                    merged_state = neighbour_dict['merged_state']
                    merged_cov = neighbour_dict['merged_cov']
                    sigma0 = neighbour_dict['GNN_Measurement'].sigma0

                    # extrapolate the merged state & covariance
                    neighbour_x = neighbour_dict['coord_Measurement'][0]
                    dx = node_x - neighbour_x
                    F = np.array([ [1, dx], [0, 1] ])
                    extrp_state = F.dot(merged_state)
                    extrp_cov = F.dot(merged_cov).dot(F.T)

                    # calc chi2 between measurement of node and extrapolated track state
                    residual = node_y - H.dot(extrp_state)       # compute the residual
                    S = H.dot(extrp_cov).dot(H.T) + sigma0**2    # covariance of residual (denominator of kalman gain)
                    inv_S = np.linalg.inv(S)
                    chi2 = residual.T.dot(inv_S).dot(residual)
                    chi2_cut = 5 * S[0][0]
                    # print("chi2:", chi2)

                    if chi2 < chi2_cut:
                        print("chi2 OK, updating state dict")
                        # this node is OK --> perform KF update
                        updated_state_dict = perform_KF_updated_state(subGraph, F, H, inv_S, extrp_state, extrp_cov, chi2, sigma0, neighbour_num, node_num)
                        updated_track_states[(neighbour_num, node_num)] = updated_state_dict
                    
                    else:
                        print("chi2 not OK, deactivating connection")
                        # deactivate the edge straight away if chi2 distance too large
                        subGraph[neighbour_num][node_num]["activated"] = 0
                
                else:
                    print("no merge state, finding smallest chi2...")
                    #Find the state with the smallest chi2 distance and extrapolate that?
                    track_state_estimates = neighbour_dict['track_state_estimates']
                    
                    F_array, H_array, inv_S_array = [], [], []
                    chi2_array, chi2_cut_array, extrp_state_array, extrp_cov_array = [], [], [], []
                    
                    for edge_tuple, attr in track_state_estimates.items():
                        
                        edge_state = attr['edge_state_vector']
                        edge_cov = attr['edge_covariance'].reshape(2, 2)
                        sigma0 = neighbour_dict['GNN_Measurement'].sigma0

                        # extrapolate the merged state & covariance
                        neighbour_x = neighbour_dict['coord_Measurement'][0]
                        dx = node_x - neighbour_x
                        F = np.array([ [1, dx], [0, 1] ])
                        extrp_state = F.dot(edge_state)
                        extrp_cov = F.dot(edge_cov).dot(F.T)

                        # calc chi2 between measurement of node and extrapolated track state
                        residual = node_y - H.dot(extrp_state)       # compute the residual
                        S = H.dot(extrp_cov).dot(H.T) + sigma0**2    # covariance of residual (denominator of kalman gain)
                        inv_S = np.linalg.inv(S)
                        chi2 = residual.T.dot(inv_S).dot(residual)
                        chi2_cut = 5 * S[0][0]

                        # append to arrays
                        F_array.append(F)
                        H_array.append(H)
                        inv_S_array.append(inv_S)
                        chi2_array.append(chi2)
                        chi2_cut_array.append(chi2_cut)
                        extrp_state_array.append(extrp_state)
                        extrp_cov_array.append(extrp_cov)

                    idx = np.argmin(chi2_array)
                    F = F_array[idx]
                    H = H_array[idx]
                    inv_S = inv_S_array[idx]
                    chi2 = chi2_array[idx]
                    chi2_cut = chi2_cut_array[idx]
                    extrp_state = extrp_state_array[idx]
                    extrp_cov = extrp_cov_array[idx]

                    print("smallest chi2 found")

                    if chi2 < chi2_cut:
                        print("chi2 OK updating track state")
                        # this node is OK --> perform KF update
                        updated_state_dict = perform_KF_updated_state(subGraph, F, H, inv_S, extrp_state, extrp_cov, chi2, sigma0, edge_tuple[1], node_num)
                        updated_track_states[(neighbour_num, node_num)] = updated_state_dict
                    
                    else:
                        print("chi2 not OK, deactivating connection")
                        # deactivate the edge straight away if chi2 distance too large
                        subGraph[neighbour_num][node_num]["activated"] = 0

            # store the updated track states
            print("Finished and storing updating track states @ node\n")
            subGraph.nodes[node_num]['updated_track_states'] = updated_track_states

    print("\n Processed all nodes")
    print("recomputing priors...")
    # recalculate priors
    compute_prior_probabilities(subGraphs, 'updated_track_states')

    # reweight
    print("\nStarting reweighting...")
    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            if 'updated_track_states' in node_attr.keys():
                updated_track_states = node_attr['updated_track_states']
                
                print("calculating norm factor for LR layers")
                calculate_norm_factor_lr_layers(subGraph, node, updated_track_states)
                
                # compute reweight denominator
                reweight_denom = 0
                for edge_tuple, updated_state_dict in updated_track_states.items():
                    reweight_denom += (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'])
                
                # compute reweight
                for edge_tuple, updated_state_dict in updated_track_states.items():
                    
                    reweight = (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'] * updated_state_dict['prior']) / reweight_denom
                    reweight /= updated_state_dict['lr_layer_norm']
                    print("REWEIGHT:", reweight)
                    
                    updated_state_dict['mixture_weight'] = reweight
                    
                    # reactivate/deactivate
                    if reweight < reweight_threshold:
                        subGraph[edge_tuple[0]][edge_tuple[1]]["activated"] = 0
                        print("deactivating edge: (", edge_tuple[0], ",", edge_tuple[1], ")")
                    else:
                        subGraph[edge_tuple[0]][edge_tuple[1]]["activated"] = 1
                        print("activating edge: (", edge_tuple[0], ",", edge_tuple[1], ")")
                
                # store the updated track states as node attr
                subGraph.nodes[node_num]['updated_track_states'] = updated_track_states
        
    title = "Subgraphs after extrapolation of merged states"
    plot_save_subgraphs(subGraphs, outputDir, title)




def main():

    reweight_threshold = 0.05
    subgraph_path = "_subgraph.gpickle"

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--inputDir', help='input directory of outlier removal')
    parser.add_argument('-o', '--outputDir', help='output directory for updated states')
    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir

    # read in subgraph data
    subGraphs = []
    filenames = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)
        filenames.append(file)

    
    extrapolate(subGraphs, reweight_threshold, outputDir)





if __name__ == "__main__":
    main()