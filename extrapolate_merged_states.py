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


def calculate_side_norm_factor(subGraph, node, updated_track_states):
    
    # split track state estimates into LHS & RHS
    node_num = node[0]
    node_attr = node[1]
    node_x_layer = node_attr['GNN_Measurement'].x
    left_nodes, right_nodes = [], []
    left_coords, right_coords = [], []

    for neighbour_num, _ in updated_track_states.items():
        neighbour_x_layer = subGraph.nodes[neighbour_num]['GNN_Measurement'].x

        # only calculate for activated edges
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            if neighbour_x_layer < node_x_layer: 
                left_nodes.append(neighbour_num)
                left_coords.append(neighbour_x_layer)
            else: 
                right_nodes.append(neighbour_num)
                right_coords.append(neighbour_x_layer)
    
    # store norm factor as node attribute
    left_norm = len(list(set(left_coords)))
    for left_neighbour in left_nodes:
        updated_track_states[left_neighbour]['side'] = "left"
        updated_track_states[left_neighbour]['lr_layer_norm'] = 1
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            updated_track_states[left_neighbour]['lr_layer_norm'] = left_norm

    right_norm = len(list(set(right_coords)))
    for right_neighbour in right_nodes:
        updated_track_states[right_neighbour]['side'] = "right"
        updated_track_states[right_neighbour]['lr_layer_norm'] = 1
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            updated_track_states[right_neighbour]['lr_layer_norm'] = right_norm



def reweight(subGraphs, track_state_estimates_key):
    reweight_threshold = 0.1

    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            if track_state_estimates_key in node_attr.keys():
                print("\nReweighting node:", node_num)
                updated_track_states = node_attr[track_state_estimates_key]
                
                calculate_side_norm_factor(subGraph, node, updated_track_states)
                
                # compute reweight denominator
                reweight_denom = 0
                for neighbour_num, updated_state_dict in updated_track_states.items():
                    if subGraph[neighbour_num][node_num]['activated'] == 1:
                        reweight_denom += (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'])
                
                # compute reweight
                for neighbour_num, updated_state_dict in updated_track_states.items():
                    if subGraph[neighbour_num][node_num]['activated'] == 1:
                        reweight = (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'] * updated_state_dict['prior']) / reweight_denom
                        reweight /= updated_state_dict['lr_layer_norm']
                        print("REWEIGHT:", reweight)
                        print("side:", updated_state_dict['side'])  
                        updated_state_dict['mixture_weight'] = reweight

                        # add as edge attribute
                        subGraph[neighbour_num][node_num]['mixture_weight'] = reweight
                    
                        # reactivate/deactivate
                        if reweight < reweight_threshold:
                            subGraph[neighbour_num][node_num]['activated'] = 0
                            print("deactivating edge: (", node_num, ",", neighbour_num, ")")
                        else:
                            subGraph[neighbour_num][node_num]['activated'] = 1
                            print("reactivating edge: (", node_num, ",", neighbour_num, ")")




def extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, chi2CutFactor):
    node_x = node_attr['GNN_Measurement'].x
    node_y = node_attr['GNN_Measurement'].y
    neighbour_x = neighbour_attr['GNN_Measurement'].x
    neighbour_y = neighbour_attr['GNN_Measurement'].y
    merged_state = node_attr['merged_state']
    merged_cov = node_attr['merged_cov']
    sigma0 = node_attr['GNN_Measurement'].sigma0

    # extrapolate the merged state from the central node to the neighbour node & storing at the neighbur node
    print("extrapolating merged state for", node_num, "to", neighbour_num)
    dx = neighbour_x - node_x
    F = np.array([ [1, dx], [0, 1] ])
    extrp_state = F.dot(merged_state)
    extrp_cov = F.dot(merged_cov).dot(F.T)

    # validate the extrapolated state against the measurement at the neighbour node
    # calc chi2 distance between measurement at neighbour node and extrapolated track state
    H = np.array([[1.,0.]])
    residual = neighbour_y - H.dot(extrp_state)     # compute the residual
    S = H.dot(extrp_cov).dot(H.T) + sigma0**2       # covariance of residual (denominator of kalman gain)
    # S = sigma0**2 - H.dot(extrp_cov).dot(H.T)
    inv_S = np.linalg.inv(S)
    chi2 = residual.T.dot(inv_S).dot(residual)
    chi2_cut = chi2CutFactor * S[0][0]
    # print("chi2:", chi2)
    # print("chi2_cut:", chi2_cut)

    # validate chi2 distance
    if chi2 < chi2_cut:
        print("chi2 OK, performing KF update...")

        # calc beta: measurement likelihood
        factor = 2 * math.pi * np.abs(S)
        norm_factor = math.pow(factor, -0.5)
        likelihood = norm_factor * np.exp(-0.5 * chi2)
        print("likelihood: ", likelihood)

        # initialize KF
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.x = extrp_state                   # X state vector
        f.F = F                             # F state transition matrix
        f.H = H                             # H measurement matrix
        f.P = extrp_cov
        f.R = sigma0**2
        f.Q = 0.
        z = neighbour_y                     # "sensor reading"

        # perform KF update & save data
        f.predict()
        f.update(z)
        updated_state, updated_cov = f.x_post, f.P_post

        return { 'coord_Measurement': (node_x, node_y),
                 'edge_state_vector': updated_state, 
                 'edge_covariance': updated_cov,
                 'likelihood': likelihood,
                 'prior': subGraph.nodes[node_num]['track_state_estimates'][neighbour_num]['prior'],                   # previous prior
                 'mixture_weight': subGraph.nodes[node_num]['track_state_estimates'][neighbour_num]['mixture_weight']  # previous weight
                }

    else:
        print("chi2 distance too large")
        print("DEACTIVATING edge connection ( ", node_num, ", ", neighbour_num, " )")
        subGraph[node_num][neighbour_num]["activated"] = 0
        return None



def message_passing(subGraphs, chi2CutFactor):
    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        # distribution of 'merged state' from nodes with this attribute to its neighbours
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]
            print("\nProcessing node: ", node_num)

            if 'merged_state' in node_attr.keys():
                print("Performing message passing")
                for neighbour_num in subGraph.neighbors(node_num):

                    # extrapolating outwards, from node to neighbour
                    if subGraph[node_num][neighbour_num]["activated"] == 1:
                        neighbour_attr = subGraph.nodes[neighbour_num]
                        updated_state_dict = extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, chi2CutFactor)
                        if updated_state_dict != None:
                            # store the updated track states at the neighbour node
                            if 'updated_track_states' not in neighbour_attr:
                                subGraph.nodes[neighbour_num]['updated_track_states'] = {node_num : updated_state_dict}
                            else:
                                subGraph.nodes[neighbour_num]['updated_track_states'][node_num] = updated_state_dict
                    else:
                        print("edge", node_num, neighbour_num, "not activated, message not transmitted")

            else: print("No merged state found, leaving for further iterations")


# use single component updated state as 'merged' state
def convert_single_updated_state(subGraphs):
    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]
            if 'updated_track_states' in node_attr.keys():
                updated_track_states = node_attr['updated_track_states']
                if len(updated_track_states) == 1:
                    # print("Single updated state found for node:", node_num, "\nconverting to 'merged'")
                    single_state_data = list(updated_track_states.values())[0]
                    subGraph.nodes[node_num]['merged_state'] = single_state_data['edge_state_vector']
                    subGraph.nodes[node_num]['merged_cov'] = single_state_data['edge_covariance']
                    subGraph.nodes[node_num]['merged_prior'] = single_state_data['prior']


def main():

    subgraph_path = "_subgraph.gpickle"

    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--inputDir', help='input directory of outlier removal')
    parser.add_argument('-o', '--outputDir', help='output directory for updated states')
    parser.add_argument('-c', '--chi2CutFactor', help='chi2 cut factor for threshold')
    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir
    chi2CutFactor = float(args.chi2CutFactor)

    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        # print("File:", file)
        subGraphs.append(sub)


    # distribute merged state to neighbours, extrapolate, validation & create new updated state(s)
    message_passing(subGraphs, chi2CutFactor)
    convert_single_updated_state(subGraphs)

    compute_prior_probabilities(subGraphs, 'updated_track_states')
    reweight(subGraphs, 'updated_track_states')
    compute_prior_probabilities(subGraphs, 'updated_track_states')

    # for i, s in enumerate(subGraphs):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")
    #     print("EDGE DATA:", s.edges.data(), "\n")

    title = "Subgraphs after iteration 2: message passing, extrapolation \n& validation of merged state, formation of updated state"
    plot_save_subgraphs(subGraphs, outputDir, title)
    plot_subgraphs_merged_state(subGraphs, outputDir, title)



if __name__ == "__main__":
    main()