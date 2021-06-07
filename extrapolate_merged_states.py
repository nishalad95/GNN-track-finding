import os, glob
import numpy as np
import math
import networkx as nx
from filterpy.kalman import KalmanFilter
from filterpy import common
import argparse


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


# second iteration update the merged track state estimates
for subGraph in subGraphs:
    if len(subGraph.nodes()) == 1: continue

    for node in subGraph.nodes(data=True):
        print("Processing node num: ", node[0])
        
        # calculate prior probablities for each neighbour
        prior_prob = {}
        for neighbour in subGraph.neighbors(node[0]):
            print("here", neighbour, subGraph.nodes[neighbour]['coord_Measurement'][0])
            x_layer = subGraph.nodes[neighbour]['coord_Measurement'][0]
            if x_layer not in prior_prob:
                prior_prob[x_layer] = [neighbour]
            else:
                prior_prob[x_layer].append(neighbour)
        
        # assign prior probabilities to node attributes
        for x_layer, neighbours in prior_prob.items():
            prior = 1/len(neighbours)
            for n in neighbours:
                subGraph.nodes[node[0]]['track_state_estimates'][n]['prior'] = prior
                

        # extrapolate each neighbour node state & perform KF update
        updated_track_states = {}
        for neighbour in subGraph.neighbors(node[0]):
            print("neighbour", neighbour)
            
            neighbour_dict = subGraph.nodes[neighbour] 
            if 'merged_state' in neighbour_dict:
                
                # get merged state vector from neighbouring node
                merged_state = neighbour_dict['merged_state']
                merged_cov = neighbour_dict['merged_cov']
                sigma0 = neighbour_dict['GNN_Measurement'].sigma0

                # extrapolate the merged state & covariance
                node_x = node[1]['coord_Measurement'][0]
                node_y = node[1]['coord_Measurement'][1]
                neighbour_x = neighbour_dict['coord_Measurement'][0]
                dx = node_x - neighbour_x
                F = np.array([ [1, dx], [0, 1] ])
                extrp_state = F.dot(merged_state)
                extrp_cov = F.dot(merged_cov).dot(F.T)

                # residual
                H = np.array([[1.,0.]])
                residual = node_y - H.dot(extrp_state)

                # covariance of residual (denominator of kalman gain)
                S = H.dot(extrp_cov).dot(H.T) + sigma0**2

                # chi2 between measurement of node and extrapolated track state
                inv_S = np.linalg.inv(S)
                chi2 = residual.T.dot(inv_S).dot(residual)

                print("Extrapolated state", extrp_state)
                print("Extrapolated cov", extrp_cov)
                print("residual", residual)
                print("Cov of residual S", S)
                print("chi2", chi2)

                # if chi2 distance OK then perform KF update
                if chi2 < 3 * S[0][0]:

                    # edge is OK
                    subGraph.nodes[node[0]]['track_state_estimates'][neighbour]['activated_edge'] = True

                    # measurement likelihood stored in network
                    norm_factor = math.pow(2 * math.pi * np.abs(inv_S), -0.5)
                    likelihood = norm_factor * np.exp(-0.5 * chi2)
                    subGraph.nodes[node[0]]['track_state_estimates'][neighbour]['likelihood'] = likelihood
                    
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

                    print("updated cov", f.P)

                    updated_state = np.array(saver['x'])[0]
                    print("x state after KF update", updated_state)
                    updated_state_dict = {'edge_state_vector': updated_state, 'edge_covariance': f.P}
                    updated_track_states[neighbour] = updated_state_dict
                
                else:
                    # deactivate this edge and this node becomes isolated
                    subGraph.nodes[node[0]]['track_state_estimates'][neighbour]['activated_edge'] = False
                


        print("NODE: updated states: ", updated_track_states, "\n")
        subGraph.nodes[node[0]]['updated_track_state_estimates'] = updated_track_states
        print("End of processing node num: ", node, "\n\n")

        #TODO: 
        # calc reweighted gaussian mixture model for valid edges & store the state
        # loop through all track_state_estimates key for that node and calc reweights
        # if activated_edge in dictionary and activated_edge == True, then calc reweights


# save the networks
for i, sub in enumerate(subGraphs):
    name = filenames[i].split("/")[3]
    nx.write_gpickle(subGraph, outputDir + name)