import os, glob
import numpy as np
import networkx as nx
from scipy.stats import chisquare
from numpy.linalg import inv
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
        
        updated_track_states = {}

        for neighbour in subGraph.neighbors(node[0]):
            print("neighbour", neighbour)
            
            neighbour_dict = subGraph.nodes[neighbour] 
            if 'merged_state' in neighbour_dict:
                
                # get merged state vector from neighbouring nodes
                merged_state = neighbour_dict['merged_state']
                merged_cov = neighbour_dict['merged_cov']

                # extrapolate the merged state
                node_x = node[1]['coord_Measurement'][0]
                node_y = node[1]['coord_Measurement'][1]
                neighbour_x = neighbour_dict['coord_Measurement'][0]
                dx = node_x - neighbour_x
                extrp_state = merged_state[0] + merged_state[1] * dx
                
                # extrapolate the merged covariance
                F = np.array([ [1, dx], [0, 1] ])
                extrp_cov = F.dot(merged_cov).dot(F.T)

                # calc chi2 between measurement at node and extrapolated track state
                chi2, _ = chisquare([node_y], [extrp_state])
                print("merged_cov", merged_cov)
                print("chisq", chi2)
                print("Extrapolated cov", extrp_cov)
                #TODO: sigma???
                sigma = extrp_cov[0][0]

                # if ch2 distance OK then perform KF update
                if chi2 < sigma:
                    # initialize KF
                    f = KalmanFilter(dim_x=2, dim_z=1)
                    f.x = [extrp_state, merged_state[1]]  # X state vector
                    f.F = F # F state transition matrix
                    
                    f.H = np.array([[1.,0.]]) # H measurement matrix
                    f.P = extrp_cov  # P: covariance

                    f.R = sigma**2
                    f.Q = 0.

                    print("f.x", f.x)
                    # save data for kf filter
                    saver = common.Saver(f)
                    f.predict()
                    f.update(extrp_state)
                    saver.save()

                    print("updated cov", f.P)

                    updated_state = np.array(saver['x'])[0]
                    print("x state after KF update", updated_state)
                    updated_state_dict = {'edge_state_vector': updated_state, 'edge_covariance': f.P}
                    updated_track_states[neighbour] = updated_state_dict

        print("NODE: updated states: ", updated_track_states, "\n")
        subGraph.nodes[node[0]]['updated_track_state_estimates'] = updated_track_states
        print("End of processing node num: ", node, "\n\n")


# save the networks
for i, sub in enumerate(subGraphs):
    name = filenames[i].split("/")[3]
    nx.write_gpickle(subGraph, outputDir + name)