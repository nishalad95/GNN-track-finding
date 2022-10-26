from filterpy.kalman import *
from filterpy import common
from scipy.stats import distributions, chisquare, chi2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
import collections
import random
from utilities import helper as h
# from community_detection import community_detection
import pprint
from collections import Counter
from itertools import combinations
import itertools
from math import *
from more_itertools import locate


COMMUNITY_DETECTION = False

def run_community_detection(candidate, fragment):
    # when == 1 potential, need to remember to append the graph to the 'remaining' list
    # > 1 potential track
    #TODO: coordinates & community detection method needs to be updated
    valid_communities, vc_coords = community_detection(candidate, fragment)
    if len(valid_communities) > 0:
        print("found communities via community detection")
        for vc, vcc in zip(valid_communities, vc_coords):
            pval = KF_track_fit_xy(sigma0, sigma_ms, vcc)
            if pval >= track_acceptance:
                print("Good KF fit, P value:", pval, "(x,y,z,r):", vcc)
                extracted.append(vc)
                extracted_pvals.append(pval)
                candidate_to_remove_from_subGraph.append(vc)
            else:
                print("pval too small,", pval, "leave for further processing")


def compute_3d_distance(coord1, coord2):
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )


def angle_trunc(a, p2):
    if p2[1] > 0.0:
        a = (2*np.pi) - a
    return a


# get angle to the positive x axis in radians
def getAngleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle = np.abs(atan2(deltaY, deltaX))
    return angle_trunc(angle, p2)


def rotate_track(coords, separation_3d_threshold):
    # coords are ordered from outermost to innermost -> use innermost edge
    p1 = coords[-1]
    p2 = coords[-2]

    # if nodes p1 and p2 are too close, use the next node
    distance = compute_3d_distance(p1, p2)
    if distance < separation_3d_threshold:
        p2 = coords[-3]

    # rotate counter clockwise, first edge to be parallel with x axis
    angle = getAngleBetweenPoints(p1, p2)
    rotated_coords = []
    for c in coords:
        x, y, z, r = c[0], c[1], c[2], c[3]
        x_new = x * np.cos(angle) - y * np.sin(angle)    # x_new = xcos(angle) - ysin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)    # y_new = xsin(angle) + ycos(angle) 
        rotated_coords.append((x_new, y_new, z, r)) 
    return rotated_coords


def KF_predict_update(f, obs_y):
    # KF predict and update
    chi2_dists = []
    saver = common.Saver(f)
    for measurement in obs_y[1:]:
        f.predict()
        f.update(measurement)
        saver.save()

        # update
        updated_state, updated_cov = f.x_post, f.P_post
        residual = measurement - f.H.dot(updated_state) 
        S = f.H.dot(updated_cov).dot(f.H.T) + f.R
        inv_S = np.linalg.inv(S)

        # chi2 distance
        chi2_dist = residual.T.dot(inv_S).dot(residual)
        chi2_dists.append(chi2_dist)
    
    # chi2 probability distribution
    total_chi2 = sum(chi2_dists)                    # chi squared statistic
    dof = len(obs_y) - 2                            # (no. of measurements * 1D) - no. of track params
    pval = distributions.chi2.sf(total_chi2, dof)
    print("P value for KF track fit: ", pval)
    return pval


def KF_track_fit_xy(sigma_ms, coords):
    # KF applied from outermost point to innermost point
    obs_x = [c[0] for c in coords]
    obs_y = [c[1] for c in coords]
    yf = obs_y[0]
    dx = coords[1][0] - coords[0][0]    # dx = x1 - x0

    # variables for F; state transition matrix
    alpha = 0.1 #1.0                                                   # OU parameter
    e1 = np.exp(-np.abs(dx) * alpha)
    f1 = (1.0 - e1) / alpha
    g1 = (np.abs(dx) - f1) / alpha
    
    # variables for Q process noise matrix
    sigma0 = 0.1                                                    # IMPORTANT This is different in model setup!
    sigma_ou = 0.00001                                              # 10^-5
    sw2 = sigma_ou**2                                               # OU parameter 
    st2 = sigma_ms**2                                               # process noise representing multiple scattering
    dx2 = dx**2
    dxw2 = dx2 * sw2
    Q02 = 0.5*dxw2
    Q01 = dx*(st2 + Q02)
    Q12 = dx*sw2

    # 3D KF: initialize at outermost layer
    f = KalmanFilter(dim_x=3, dim_z=1)
    f.x = np.array([yf, 0., 0.])                                    # X state vector [yf, dy/dx, w] = [coordinate, track inclination, integrated OU]

    f.F = np.array([[1.,    dx,     g1], 
                    [0.,    1.,     f1],
                    [0.,    0.,     e1]])                           # F state transition matrix, extrapolation Jacobian - linear & OU
    
    f.H = np.array([[1., 0., 0.]])                                  # H measurement matrix
    f.P = np.array([[sigma0**2,  0., 0.],     
                    [0.,         1., 0.],
                    [0.,         0., 1.]])                          # P covariance
    
    f.R = sigma0**2                                                 # R measuremnt noise
    f.Q = np.array([[dx2*(st2 + 0.25*dxw2), Q01,        Q02], 
                    [Q01,                   st2 + dxw2, Q12],
                    [Q02,                   Q12,        sw2]])      # Q process uncertainty/noise, OU model

    pval = KF_predict_update(f, obs_y)
    return pval



def KF_track_fit_zr(sigma_ms, coords):
    # KF applied from outermost point to innermost point
    obs_x = [c[2] for c in coords]
    obs_y = [c[3] for c in coords]
    yf = obs_y[0]
    dx = coords[1][2] - coords[0][2]    # dx = z1 - z0

    sigma0 = 0.1                                                    # IMPORTANT This is different in model setup!
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([yf, 0.])                                # X state vector [yf, dy/dx, w] = [coordinate, track inclination, integrated OU]

    f.F = np.array([[1.,    dx], 
                    [0.,    1.]])                           # F state transition matrix, extrapolation Jacobian - linear & OU
    
    f.H = np.array([[1., 0.]])                              # H measurement matrix
    f.P = np.array([[sigma0**2,  0.],     
                    [0.,         1000.]])                   # P covariance
    
    f.R = sigma0**2                                         # R measuremnt noise
    f.Q = sigma_ms                                          # Q process uncertainty/noise, OU model

    pval = KF_predict_update(f, obs_y)
    return pval



def CCA(subCopy):
    edges_to_remove = []
    for edge in subCopy.edges():
        if subCopy[edge[0]][edge[1]]['activated'] == 0: edges_to_remove.append(edge)  
    
    potential_tracks = []
    if len(edges_to_remove) > 0 :
        for edge in edges_to_remove: subCopy.remove_edge(edge[0], edge[1])

        for component in nx.weakly_connected_components(subCopy):
            potential_tracks.append(subCopy.subgraph(component).copy())
    else: 
        potential_tracks.append(subCopy)

    return potential_tracks


def main():
    
    # parse command line args
    parser = argparse.ArgumentParser(description='extract track candidates')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-c', '--candidates', help='output directory to save track candidates')
    parser.add_argument('-r', '--remain', help='output directory to save remaining network')
    parser.add_argument('-f', '--fragments', help='output directory to save track fragments')
    parser.add_argument('-p', '--pval', help='chi-squared track candidate acceptance level')
    parser.add_argument('-s', '--separation_3d_threshold', help="3d distance cut between close proximity nodes, used in node merging")
    
    # parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-m', '--sigma_ms', help="uncertainty due to multiple scattering, process noise")
    parser.add_argument('-n', '--numhits', help="minimum number of hits for good track candidate")
    args = parser.parse_args()

    # set variables
    inputDir = args.input
    candidatesDir = args.candidates
    remainingDir = args.remain
    fragmentsDir = args.fragments
    track_acceptance = float(args.pval)
    # sigma0 = float(args.error)
    sigma_ms = float(args.sigma_ms)
    subgraph_path = "_subgraph.gpickle"
    fragment = int(args.numhits)
    separation_3d_threshold = float(args.separation_3d_threshold)
    
    # get iteration num
    inputDir_list = inputDir.split("/")
    iterationDir = filter(lambda x: x.startswith('iteration_'), inputDir_list)
    for i in iterationDir: iteration_num = i.split("_")[-1]

    # read in subgraph data
    subGraphs = []
    i = 0
    path = inputDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        subGraphs.append(sub)
        i += 1
        path = inputDir + str(i) + subgraph_path

    print("Intial total no. of subgraphs:", len(subGraphs))

    extracted = []
    extracted_pvals = []
    extracted_pvals_zr = []
    remaining = []
    fragments = []
    for i, subGraph in enumerate(subGraphs):
        
        subCopy = subGraph.copy()
        potential_tracks = CCA(subCopy)     # remove any deactive edges & identify subgraphs
        print("\nProcessing subGraph: ", i, "\nNum. of potential tracks: ", len(potential_tracks))
        print("node numbers: ", subGraph.nodes())

        candidate_to_remove_from_subGraph = []
        for n, candidate in enumerate(potential_tracks):
            print("Processing candidate: ", n)
            print("node numbers in candidate: ", candidate.nodes())

            #TODO: need to check for holes?

            # check for track fragments
            if candidate.number_of_nodes() >= fragment:
                
                candidate_to_assess = candidate
                # check for 1 hit per layer - use volume_id & in_volume_layer_id
                vivl_id_values = nx.get_node_attributes(candidate_to_assess,'vivl_id').values()

                if len(vivl_id_values) == len(set(vivl_id_values)): 
                    # good candidate
                    print("no duplicates volume_ids & in_volume_layer_ids for this candidate")

                    # sort the candidates by radius r largest to smallest (4th element in this tuple of tuples: (node_num, (x,y,z,r)) )
                    nodes_coords_tuples = list(nx.get_node_attributes(candidate_to_assess, 'xyzr').items())
                    sorted_nodes_coords_tuples = sorted(nodes_coords_tuples, reverse=True, key=lambda item: item[1][3])
                    # check if the sorted nodes are connected
                    all_connected = True
                    for j in range(len(sorted_nodes_coords_tuples) - 1):
                        node1 = sorted_nodes_coords_tuples[j][0]
                        node2 = sorted_nodes_coords_tuples[j+1][0]
                        if not candidate_to_assess.has_edge(node1, node2) and not candidate_to_assess.has_edge(node2, node1):
                            all_connected = False
                    
                    if all_connected:
                        coords = [element[1] for element in sorted_nodes_coords_tuples]
                        # rotate the track such that innermost edge parallel to x-axis - r&z components are left unchanged
                        coords = rotate_track(coords, separation_3d_threshold)
                        # apply KF track fit - TODO: parallelize these 2 KF track fits
                        pval = KF_track_fit_xy(sigma_ms, coords)
                        pval_zr = KF_track_fit_zr(sigma_ms, coords)
                        if (pval >= track_acceptance) and (pval_zr >= track_acceptance):
                            print("Good KF fit, p-value:", pval, "\n(x,y,z,r):", coords)
                            extracted.append(candidate)
                            extracted_pvals.append(pval)
                            extracted_pvals_zr.append(pval_zr)
                            candidate_to_remove_from_subGraph.append(candidate)
                            
                        else:
                            print("p-value too small, leave for further processing, pval_xy: " + str(pval) + " pval_zr: " + str(pval_zr))
                    else:
                        print("Candidate not accepted, not connceted in order")

                else: 
                    print("Bad candidate, > 1 hit per layer, will pass through community detection")
                    # TODO: community detection?
                    if COMMUNITY_DETECTION:
                        run_community_detection(candidate, fragment)
            else:
                print("Too few nodes, track fragment")

        # remove good candidates from subGraph & save remaining network
        for good_candidate in candidate_to_remove_from_subGraph:
            nodes = good_candidate.nodes()
            subGraph.remove_nodes_from(nodes)
        if (len(subGraph.nodes()) <= 3) and (len(subGraph.nodes()) > 0): 
            fragments.append(subGraph)
        elif len(subGraph.nodes()) >= 4:
            remaining.append(subGraph)

    
    # attach iteration number & color to good extracted tracks
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])]
    for subGraph in extracted:
        subGraph.graph["iteration"] = iteration_num
        subGraph.graph["color"] = color[0]

    print("\nNumber of extracted candidates during this iteration:", len(extracted))
    print("Number of remaining subGraphs to be further processed:", len(remaining))
    print("Number of track fragments found:", len(fragments))
    # load all extracted candidates, from previous iterations
    i = 0
    path = candidatesDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        extracted.append(sub)
        i += 1
        path = candidatesDir + str(i) + subgraph_path

    print("Total number of extracted candidates:", len(extracted))

    # plot and save all extracted candidates from previous and this iteration
    h.plot_save_subgraphs_iterations(extracted, extracted_pvals, extracted_pvals_zr, candidatesDir, "Extracted candidates", node_labels=True, save_plot=True)
    # plot and save the remaining subgraphs to be further processed
    h.plot_subgraphs(remaining, remainingDir, node_labels=True, save_plot=True, title="Remaining candidates")
    # save remaining and track fragments
    for i, sub in enumerate(remaining):
        h.save_network(remainingDir, i, sub)
    for i, sub in enumerate(fragments):
        h.save_network(fragmentsDir, i, sub)
    

    # TODO: plot the distribution of edge weightings within the extracted candidates




if __name__ == "__main__":
    main()