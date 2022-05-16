from filterpy.kalman import *
from filterpy import common
from scipy.stats import distributions, chisquare, chi2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import argparse
import collections
import random
from utilities import helper as h
from community_detection import community_detection
import pprint
from collections import Counter
from itertools import combinations
import itertools
from math import *


COMMUNITY_DETECTION = False

# def run_community_detection():
    # when == 1 potential, need to remember to append the graph to the 'remaining' list
    # > 1 potential track
    #     #TODO: coordinates & community detection method needs to be updated
    #     valid_communities, vc_coords = community_detection(candidate, fragment)
    #     if len(valid_communities) > 0:
    #         print("found communities via community detection")
    #         for vc, vcc in zip(valid_communities, vc_coords):
    #             pval = KF_track_fit(sigma0, sigma_ms, vcc)
    #             if pval >= track_acceptance:
    #                 print("Good KF fit, P value:", pval, "(x,y,z,r):", vcc)
    #                 extracted.append(vc)
    #                 extracted_pvals.append(pval)
    #                 candidate_to_remove_from_subGraph.append(vc)
    #             else:
    #                 print("pval too small,", pval, "leave for further processing")


def compute_3d_distance(coord1, coord2):
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )


def get_midpoint_coords(xyzr_coords):
    x1, y1, z1 = xyzr_coords[0][0], xyzr_coords[0][1], xyzr_coords[0][2]
    x2, y2, z2 = xyzr_coords[1][0], xyzr_coords[1][1], xyzr_coords[1][2]
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2
    zm = (z1 + z2) / 2
    rm = np.sqrt(xm**2 + ym**2)
    return xm, ym, zm, rm


def check_close_proximity_nodes(subGraph, separation_3d_threshold):
    # determine if a subgraph contains between 1 - 3 layers with 2 close nodes
    node_in_volume_layer_id_dict = nx.get_node_attributes(subGraph, 'in_volume_layer_id')
    # node_module_id_dict = nx.get_node_attributes(subGraph, 'module_id')
    counts = Counter(node_in_volume_layer_id_dict.values())
    num_nodes_in_same_layer = 2
    layer_counts_dict = dict((k, v) for k, v in dict(counts).items() if int(v) == num_nodes_in_same_layer)
    num_layers_with_multiple_nodes = len(layer_counts_dict)     # currently every layer with 2 nodes in it (num_nodes_in_same_layer)

    if 1 <= num_layers_with_multiple_nodes <= 3:
        # get node indexes which are in the same layer
        for layer_id, count in layer_counts_dict.items():
            node_idx = [node for node in node_in_volume_layer_id_dict.keys() if layer_id == node_in_volume_layer_id_dict[node]]
            print("nodes that are close together:\n", node_idx)
            
            # currently only merging with 2 nodes in close proximity
            if len(node_idx) == num_nodes_in_same_layer:

                # check that these 2 nodes have a common node in their neighbourhood
                node1 = node_idx[0]
                node2 = node_idx[1]
                node1_edges = subGraph.edges(node1)
                node2_edges = subGraph.edges(node2)
                node1_edges = list(itertools.chain.from_iterable(node1_edges))
                node2_edges = list(itertools.chain.from_iterable(node2_edges))
                node1_edges = filter(lambda val: val != node1, node1_edges)
                node2_edges = filter(lambda val: val != node2, node2_edges)
                common_nodes = list(set(node1_edges).intersection(node2_edges))

                if len(common_nodes) != 0:
                    # compute pairwise separation in 3D space of all nodes in close proximity in this layer
                    xyzr_coords = [subGraph.nodes[n]['xyzr'] for n in node_idx]
                    separation = [compute_3d_distance(a, b) for a, b in combinations(xyzr_coords, 2)]

                    # node merging if separation is below threhold
                    if all(i <= separation_3d_threshold for i in separation):
                        # replace for node 1, delete node 2 - use midpoint coordinates
                        xm, ym, zm, rm = get_midpoint_coords(xyzr_coords)
                        node1 = node_idx[0]
                        node2 = node_idx[1]
                        subGraph.nodes[node1]['xyzr'] = (xm, ym, zm, rm)
                        subGraph.nodes[node1]['xy'] = (xm, ym)
                        subGraph.nodes[node1]['zr'] = (zm, rm)
                        subGraph.nodes[node1]['GNN_Measurement'].x = xm
                        subGraph.nodes[node1]['GNN_Measurement'].y = ym
                        subGraph.nodes[node1]['GNN_Measurement'].z = zm
                        subGraph.nodes[node1]['GNN_Measurement'].r = rm
                        # merge module ids
                        node1_module_id = subGraph.nodes[node1]['module_id']
                        node2_module_id = subGraph.nodes[node2]['module_id']
                        subGraph.nodes[node1]['module_id'] = np.concatenate((node1_module_id, node2_module_id))
                        # merge particle & hit ids
                        dict1 = subGraph.nodes[node1]['hit_dissociation']
                        dict2 = subGraph.nodes[node2]['hit_dissociation']
                        new_hit_ids = np.concatenate((dict1['hit_id'], dict2['hit_id']))
                        new_particle_ids = dict1['particle_id'] + dict2['particle_id']
                        subGraph.nodes[node1]['hit_dissociation'] = {'hit_id' : new_hit_ids,
                                                                     'particle_id' : new_particle_ids}
                        subGraph.remove_node(node2)
    return subGraph


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a


# get angle to the positive x axis in radians
def getAngleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    return angle_trunc(atan2(deltaY, deltaX))


def rotate_track(coords, separation_3d_threshold):
    # coords are ordered from outermost to innermost -> use innermost edge
    p1 = coords[-1]
    p2 = coords[-2]

    # if nodes p1 and p2 are too close, use the next node
    distance = compute_3d_distance(p1, p2)
    if distance < separation_3d_threshold:
        p2 = coords[-3]

    # rotate counter clockwise, first edge to be parallel with x axis
    angle = 2*pi - getAngleBetweenPoints(p1, p2)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_coords = []
    for c in coords:
        x, y = c[0], c[1]
        x_new = x * np.cos(angle) - y * np.sin(angle)    # x_new = xcos(angle) - ysin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)    # y_new = xsin(angle) + ycos(angle) 
        rotated_coords.append((x_new, y_new)) 
    return rotated_coords


def KF_track_fit(sigma0, sigma_ms, coords):
    obs_x = [c[0] for c in coords]
    obs_y = [c[1] for c in coords]
    yf = obs_y[0]
    dx = coords[1][0] - coords[0][0]

    # variables for F; state transition matrix
    alpha = 0.1                                                    # OU parameter
    e1 = np.exp(-np.abs(dx) * alpha)
    f1 = (1.0 - e1) / alpha
    g1 = (np.abs(dx) - f1) / alpha
    
    # variables for Q process noise matrix
    # sigma_ou = 0.0001                                               # 10^-4: TODO needs tuning
    sigma_ou = 0.00001
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
    print("P value: ", pval)

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
    parser.add_argument('-p', '--pval', help='chi-squared track candidate acceptance level')
    parser.add_argument('-s', '--separation_3d_threshold', help="3d distance cut between close proximity nodes, used in node merging")
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-m', '--sigma_ms', help="uncertainty due to multiple scattering, process noise")
    parser.add_argument('-n', '--numhits', help="minimum number of hits for good track candidate")
    args = parser.parse_args()

    # set variables
    inputDir = args.input
    candidatesDir = args.candidates
    remainingDir = args.remain
    track_acceptance = float(args.pval)
    sigma0 = float(args.error)
    sigma_ms = float(args.sigma_ms)
    subgraph_path = "_subgraph.gpickle"
    pvalsfile_path = "_pvals.csv"
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
    remaining = []
    for i, subGraph in enumerate(subGraphs):
        
        subCopy = subGraph.copy()
        potential_tracks = CCA(subCopy)     # remove any deactive edges & identify subgraphs
        print("\nProcessing subGraph: ", i, "\nNum. of potential tracks: ", len(potential_tracks))

        candidate_to_remove_from_subGraph = []
        for n, candidate in enumerate(potential_tracks):
            print("Processing candidate: ", n)

            #TODO: need to check for holes?

            # check for track fragments
            if len(candidate.nodes()) >= fragment:
                # check for close proximity nodes - merge where appropriate
                candidate = check_close_proximity_nodes(candidate, separation_3d_threshold)
                # check for 1 hit per layer - use volume_id & in_volume_layer_id
                vivl_id_values = nx.get_node_attributes(candidate,'vivl_id').values()
                if len(vivl_id_values) == len(set(vivl_id_values)): 
                    # good candidate
                    print("no duplicates volume_ids & in_volume_layer_ids for this candidate")
                    # rotate the track such that innermost edge parallel to x-axis
                    coords = list(nx.get_node_attributes(candidate, 'xyzr').values())
                    coords = sorted(coords, reverse=True, key=lambda xyzr: xyzr[3])
                    coords = rotate_track(coords, separation_3d_threshold)
                    # apply KF track fit
                    pval = KF_track_fit(sigma0, sigma_ms, coords)
                    if pval >= track_acceptance:
                        print("Good KF fit, p-value:", pval, "\n(x,y,z,r):", coords)
                        extracted.append(candidate)
                        extracted_pvals.append(pval)
                        candidate_to_remove_from_subGraph.append(candidate)
                    else:
                        print("p-value too small:", pval, "leave for further processing")
                else: 
                    print("Bad candidate, > 1 hit per layer, will pass through community detection")
                    # TODO: community detection?
                    # if COMMUNITY_DETECTION:
                    #     run_community_detection()
            else:
                print("Too few nodes, track fragment")

        # remove good candidates from subGraph & save remaining network
        for good_candidate in candidate_to_remove_from_subGraph:
            nodes = good_candidate.nodes()
            subGraph.remove_nodes_from(nodes)
        if len(subGraph.nodes()) > 0: remaining.append(subGraph)

    
    # attach iteration number & color to good extracted tracks
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])]
    for subGraph in extracted:
        subGraph.graph["iteration"] = iteration_num
        subGraph.graph["color"] = color[0]

    print("\nNumber of extracted candidates during this iteration:", len(extracted))
    print("Number of remaining subGraphs to be further processed:", len(remaining))
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
    h.plot_save_subgraphs_iterations(extracted, extracted_pvals, candidatesDir, "Extracted candidates", node_labels=True, save_plot=True)
    # plot and save the remaining subgraphs to be further processed
    h.plot_subgraphs(remaining, remainingDir, node_labels=True, save_plot=True, title="Remaining candidates")
    for i, sub in enumerate(remaining):
        h.save_network(remainingDir, i, sub)

    # TODO: plot the distribution of edge weightings within the extracted candidates




if __name__ == "__main__":
    main()