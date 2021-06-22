import os, glob
import numpy as np
import math
import networkx as nx
from filterpy.kalman import KalmanFilter
from filterpy import common
import argparse
from utils import plot_save_subgraphs

reweight_threshold = 0.1
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


def calculate_weights_left_right_layers(node):
    
    # split track state estimates into LHS & RHS
    node_x_layer = node[1]['GNN_Measurement'].x
    track_state_estimates = node[1]['track_state_estimates']
    left_nodes, right_nodes = [], []
    left_coords, right_coords = [], []
    for node_num, v in track_state_estimates.items():
        neighbour_x_layer = v['coord_Measurement'][0]
        if neighbour_x_layer < node_x_layer: 
            left_nodes.append(node_num)
            left_coords.append(neighbour_x_layer)
        else: 
            right_nodes.append(node_num)
            right_coords.append(neighbour_x_layer)
    
    # update prior probabilities based on no. of layers on LHS/RHS of track
    left_norm = len(list(set(left_coords)))
    for lf in left_nodes:
        track_state_estimates[lf]['weight'] = track_state_estimates[lf]['prior'] / left_norm

    right_norm = len(list(set(right_coords)))
    for rn in right_nodes:
        track_state_estimates[rn]['weight'] = track_state_estimates[rn]['prior'] / right_norm

    print("NODE:", node)
    # print("TRACK STATE EST", track_state_estimates)


# extrapolate track state estimates
for subGraph in subGraphs:
    if len(subGraph.nodes()) == 1: continue

    for node in subGraph.nodes(data=True):
        print("\nProcessing node num: ", node[0])
        
        calculate_weights_left_right_layers(node)

        # extrapolate each neighbour node state & perform KF update
        updated_track_states = {}
        deactivated_neighbour_edges = []
        node_x = node[1]['coord_Measurement'][0]
        node_y = node[1]['coord_Measurement'][1]
        for neighbour in subGraph.neighbors(node[0]):
            
            print("neighbour", neighbour)
            neighbour_dict = subGraph.nodes[neighbour]
            # print("neighbour_dict: ", neighbour_dict)
            
            if 'merged_state' in neighbour_dict:
                # clustering was executed for that neighbour node
                
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

                # compute the residual
                H = np.array([[1.,0.]])
                residual = node_y - H.dot(extrp_state)

                # covariance of residual (denominator of kalman gain)
                S = H.dot(extrp_cov).dot(H.T) + sigma0**2

                # chi2 between measurement of node and extrapolated track state
                inv_S = np.linalg.inv(S)
                chi2 = residual.T.dot(inv_S).dot(residual)
                print("chi2:", chi2)

                # if chi2 distance < 3*sigma then perform KF update
                if chi2 < 3 * S[0][0]:
                    # this edge is OK
             
                    # measurement likelihood stored in network
                    norm_factor = math.pow(2 * math.pi * np.abs(inv_S), -0.5)
                    likelihood = norm_factor * np.exp(-0.5 * chi2)
                    
                    # initialize KF
                    f = KalmanFilter(dim_x=2, dim_z=1)
                    f.x = extrp_state  # X state vector
                    f.F = F # F state transition matrix
                    f.H = H # H measurement matrix
                    f.P = extrp_cov  # P: covariance #TODO: check if use of extrp_cov correct here
                    f.R = sigma0**2
                    f.Q = 0.

                    # perform KF update & save data
                    saver = common.Saver(f)
                    f.predict()
                    f.update(extrp_state[0])
                    saver.save()

                    updated_state = np.array(saver['x'])[0]
                    # print("updated cov", f.P)
                    # print("x state after KF update", updated_state)

                    updated_state_dict = {  'edge_state_vector': updated_state, 
                                            'edge_covariance': f.P,
                                            'likelihood': likelihood,
                                            'prior': subGraph.nodes[node[0]]['track_state_estimates'][neighbour]['prior'],
                                            'weight': subGraph.nodes[node[0]]['track_state_estimates'][neighbour]['weight']
                                        }
                    updated_track_states[neighbour] = updated_state_dict
                
                else:
                    deactivated_neighbour_edges.append(neighbour)
        
        # reweight based on likelihood
        reweight_denom = 0
        for neighbour, updated_state_dict in updated_track_states.items():
            reweight_denom += (updated_state_dict['weight'] * updated_state_dict['likelihood'])
        
        for neighbour, updated_state_dict in updated_track_states.items():
            reweight = (updated_state_dict['weight'] * updated_state_dict['likelihood'] * updated_state_dict['prior']) / reweight_denom
            print("REWEIGHT:", reweight)
            updated_state_dict['reweight'] = reweight
            if reweight < reweight_threshold:
                deactivated_neighbour_edges.append(neighbour)
        
        # store the updated track states
        subGraph.nodes[node[0]]['updated_track_state_estimates'] = updated_track_states

        print("\n\nNODE: updated states after reweighting: ", updated_track_states, "\n\n")
        print("End of processing node num: \n", node, "\n\n")

        deactivated_neighbour_edges = list(set(deactivated_neighbour_edges))
        print("deactivated_neighbour_edges:", deactivated_neighbour_edges)
        # remove deactivated_neighbour_edges which was calculated from low chi2 dist
        # remove low_weighted_neighbours edges - components with v small weightings
        for neighbour in deactivated_neighbour_edges:
            subGraph.remove_edges_from([(node[0], neighbour), (neighbour, node[0])]) # remove bidirectional edge


title = "Subgraphs after extrapolation of merged states"
plot_save_subgraphs(subGraphs, outputDir, title)

# save the networks
# for i, sub in enumerate(subGraphs):
#     name = filenames[i].split("/")[3]
#     nx.write_gpickle(subGraph, outputDir + name)