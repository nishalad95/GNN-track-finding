import os, glob
from filterpy.kalman.kalman_filter import update
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



def reweight(subGraphs, track_state_estimates_key):
    print("\nStarting reweighting...")
    reweight_threshold = 0.05

    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            if track_state_estimates_key in node_attr.keys():
                updated_track_states = node_attr[track_state_estimates_key]
                
                print("calculating norm factor for LR layers")
                calculate_norm_factor_lr_layers(subGraph, node, updated_track_states)
                
                # compute reweight denominator
                reweight_denom = 0
                for edge_tuple, updated_state_dict in updated_track_states.items():
                    reweight_denom += (updated_state_dict['previous_mixture_weight'] * updated_state_dict['likelihood'])
                
                # compute reweight
                for edge_tuple, updated_state_dict in updated_track_states.items():
                    
                    reweight = (updated_state_dict['previous_mixture_weight'] * updated_state_dict['likelihood'] * updated_state_dict['prior']) / reweight_denom
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
                subGraph.nodes[node_num][track_state_estimates_key] = updated_track_states
            
            print("NODE:\n")
            pprint.pprint(node)



def extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr):
    node_x = node_attr['GNN_Measurement'].x
    neighbour_x = neighbour_attr['GNN_Measurement'].x
    neighbour_y = neighbour_attr['GNN_Measurement'].y

    # get merged state vector from central node
    merged_state = node_attr['merged_state']
    merged_cov = node_attr['merged_cov']
    sigma0 = node_attr['GNN_Measurement'].sigma0

    # extrapolate the merged state from the central node to the neighbour node
    print("extrapolating merged state...")
    dx = neighbour_x - node_x
    F = np.array([ [1, dx], [0, 1] ])
    extrp_state = F.dot(merged_state)
    extrp_cov = F.dot(merged_cov).dot(F.T)

    # validate the extrapolated state against the measurement at the neighbour node
    # calc chi2 distance between measurement at neighbour node and extrapolated track state
    H = np.array([[1.,0.]])
    residual = neighbour_y - H.dot(extrp_state)     # compute the residual
    S = H.dot(extrp_cov).dot(H.T) + sigma0**2       # covariance of residual (denominator of kalman gain)
    inv_S = np.linalg.inv(S)
    chi2 = residual.T.dot(inv_S).dot(residual)
    chi2_cut = 5 * S[0][0]
    # print("chi2:", chi2)
    # print("chi2_cut:", chi2_cut)

    # validate chi2 distance
    if chi2 < chi2_cut:
        # print("chi2 OK, performing KF update...")

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
        # print("updated_state", updated_state, "updated_cov", updated_cov)

        return { 'edge_state_vector': updated_state, 
                 'edge_covariance': updated_cov,
                 'likelihood': likelihood,
                 'prior': subGraph.nodes[neighbour_num]['track_state_estimates'][(node_num, neighbour_num)]['prior'], # previous prior
                 'previous_mixture_weight': subGraph.nodes[neighbour_num]['track_state_estimates'][(node_num, neighbour_num)]['mixture_weight']
                }

    else:
        print("chi2 distance too large, deactivating edge connection...")
        subGraph[node_num][neighbour_num]["activated"] = 0
        return None




def message_passing(subGraphs):
    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        # distribution of 'merged state' from nodes with this attribute to its neighbours
        for node in subGraph.nodes(data=True):
            print("")
            node_num = node[0]
            node_attr = node[1]
            if 'merged_state' in node_attr.keys():
                print("Performing message passing for node: ", node_num)
                for neighbour_num in subGraph.neighbors(node_num):
                    neighbour_attr = subGraph.nodes[neighbour_num]
                    updated_state_dict = extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr)
                    if updated_state_dict != None:
                        # store the updated track states
                        if 'updated_track_states' not in neighbour_attr:
                            subGraph.nodes[neighbour_num]['updated_track_states'] = {(node_num, neighbour_num) : updated_state_dict}
                        else:
                            subGraph.nodes[neighbour_num]['updated_track_states'][(node_num, neighbour_num)] = updated_state_dict
                 
                        # print("updated neighbour dict: \n")
                        # pprint.pprint(subGraph.nodes[neighbour_num])

            else: print("no merged state found, leaving for further iterations")




def main():

    # reweight_threshold = 0.05
    subgraph_path = "_subgraph.gpickle"

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--inputDir', help='input directory of outlier removal')
    parser.add_argument('-o', '--outputDir', help='output directory for updated states')
    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir

    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    message_passing(subGraphs)

    # update network state: recompute priors based on active edges
    compute_prior_probabilities(subGraphs, 'updated_track_states')
    reweight(subGraphs, 'updated_track_states')

    title = "Subgraphs after iteration 2: message passing, extrapolation \n& validation of merged state, formation of updated state"
    plot_save_subgraphs(subGraphs, outputDir, title)
    plot_subgraphs_merged_state(subGraphs, outputDir, title)



if __name__ == "__main__":
    main()