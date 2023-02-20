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

# def run_community_detection(candidate, fragment):
#     # when == 1 potential, need to remember to append the graph to the 'remaining' list
#     # > 1 potential track
#     #TODO: coordinates & community detection method needs to be updated
#     valid_communities, vc_coords = community_detection(candidate, fragment)
#     if len(valid_communities) > 0:
#         print("found communities via community detection")
#         for vc, vcc in zip(valid_communities, vc_coords):
#             pval = KF_track_fit_xy(sigma0, sigma_ms, vcc)
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


def check_close_proximity_nodes(subgraph, threshold_distance):
    # get the freq distribution of the vivl_ids
    vivl_id_dict = nx.get_node_attributes(subgraph, "vivl_id")
    module_id_dict = nx.get_node_attributes(subgraph, "module_id")
    node_nums = list(vivl_id_dict.keys())
    vivl_ids = list(vivl_id_dict.values())
    vivl_ids_freq = {x:vivl_ids.count(x) for x in vivl_ids}
    freq_count = list(vivl_ids_freq.values())

    copied_subgraph = None
    # TODO:
    # # scenario 1)
    # # check that there are exactly 2 nodes per layer in all layers
    # if not any(count != 2 for count in freq_count):
    #     # print("Expect 2 nodes in each layer")
    #     # print("Here! subgraph: ",str(i))
    #     # TODO: execute track splitting
    #     print("TODO: execute track splitting")

    # scenario 2)
    # check there is 1 node per layer in all layers, apart from 2 or fewer layers with 2 nodes
    if 2 in freq_count:
        # check there were only 2 or fewer occurences of the frequency '2' (2 nodes per layer)
        freq_count_remove_2 = list(filter(lambda x: x!= 2, freq_count))
        if len(freq_count) - len(freq_count_remove_2) <= 2:
            # check that all other values are equal to 1
            if not any(count != 1 for count in freq_count_remove_2):
                # Get duplicate nodes in same layer & compute their 3d separation distance
                duplicated_vivl_ids = list(set([tup for tup in vivl_ids if vivl_ids.count(tup) > 1]))
                
                # there could be more than 1 duplicated element
                copied_subgraph = subgraph.copy()
                for dup in duplicated_vivl_ids:
                    # get the indexes which they appear at, and hence get the node numbers
                    indexes_of_repeated_items = list(locate(vivl_ids, lambda x: x == dup))
                    nodes_of_interest = [node_nums[idx] for idx in indexes_of_repeated_items]
                    # check if only 2 nodes are presented for each duplicated item
                    if len(nodes_of_interest) == 2:
                        # compute the distance between the nodes
                        node1 = nodes_of_interest[0]
                        node2 = nodes_of_interest[1]
                        node1_coords = copied_subgraph.nodes[node1]['xyzr']
                        node2_coords = copied_subgraph.nodes[node2]['xyzr']
                        distance = np.sqrt( (node1_coords[0] - node2_coords[0])**2 +
                                            (node1_coords[1] - node2_coords[1])**2 +
                                            (node1_coords[2] - node2_coords[2])**2 )
                        if distance <= threshold_distance:
                            # merge the 2 nodes together and update the copied_subgraph
                            # the copied_subgraph contains the candidate with the merged nodes - only to be used in the KF
                            nodes_to_merge = (node1_coords, node2_coords)
                            xm, ym, zm, rm = get_midpoint_coords(nodes_to_merge)
                            print("merging of nodes possible")
                            print("distance separation of close proximity nodes: ", distance)
                            
                            # change subgraph attributes
                            copied_subgraph.nodes[node1]['GNN_Measurement'].x = xm
                            copied_subgraph.nodes[node1]['GNN_Measurement'].y = ym
                            copied_subgraph.nodes[node1]['GNN_Measurement'].z = zm
                            copied_subgraph.nodes[node1]['GNN_Measurement'].r = rm
                            copied_subgraph.nodes[node1]['xy'] = (xm, ym)
                            copied_subgraph.nodes[node1]['zr'] = (zm, rm)
                            copied_subgraph.nodes[node1]['xyzr'] = (xm, ym, zm, rm)
                            node1_module_id = copied_subgraph.nodes[node1]['module_id']
                            node2_module_id = copied_subgraph.nodes[node2]['module_id']
                            copied_subgraph.nodes[node1]['module_id'] = np.concatenate((node1_module_id, node2_module_id))
                            dict1 = copied_subgraph.nodes[node1]['hit_dissociation']
                            dict2 = copied_subgraph.nodes[node2]['hit_dissociation']
                            new_hit_ids = np.concatenate((dict1['hit_id'], dict2['hit_id']))
                            new_particle_ids = dict1['particle_id'] + dict2['particle_id']
                            copied_subgraph.nodes[node1]['hit_dissociation'] = {'hit_id' : new_hit_ids,
                                                                                'particle_id' : new_particle_ids}
                            # remove the other node
                            copied_subgraph.remove_node(node2)

                        else:
                            print("merging of nodes nodes not possible, too large distance between them, distance: ", distance)
                            copied_subgraph = None
                            break
                    else:
                        print("merging not possible, more than 2 nodes close together")
                        copied_subgraph = None
                        break
            else:
                print("More than 1 node per layer, cannot process subgraph")
        else:
            print("More than 1 layer with 2 nodes, cannot process subgraph")
    else:
        print("Cannot process subgraph, leaving for further iterations")
    
    return subgraph, copied_subgraph



def angle_trunc(a, p2):
    if p2 > 0.0:
        a = (2*np.pi) - a
    return a
    

# get angle to the positive x axis in radians
def getAngleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    angle_xy = atan2(deltaY, deltaX)
    deltaZ = p2[2] - p1[2]
    deltaR = p2[3] - p1[3]
    angle_zr = atan2(deltaZ, deltaR)
    return angle_xy, angle_zr


def rotate_track(coords, separation_3d_threshold):
    # coords are ordered from outermost to innermost -> use innermost edge
    p1 = coords[-1]
    p2 = coords[-2]

    # if nodes p1 and p2 are too close, use the next node
    distance = compute_3d_distance(p1, p2)
    if distance < separation_3d_threshold:
        p2 = coords[-3]

    # rotate such that the first edge is parallel with x axis
    angle_xy, angle_zr = getAngleBetweenPoints(p1, p2)

    rotated_coords = []
    for c in coords:
        x, y, z, r = c[0], c[1], c[2], c[3]
        x_new = x * np.cos(angle_xy) + y * np.sin(angle_xy)  
        y_new = -x * np.sin(angle_xy) + y * np.cos(angle_xy) 
        r_new = r * np.cos(angle_zr) + r * np.sin(angle_zr) 
        z_new = -z * np.sin(angle_zr) + z * np.cos(angle_zr) 
        rotated_coords.append((x_new, y_new, z_new, r_new)) 
    return rotated_coords


# Calculate the unknowns of the equation y = ax^2 + bx + c
def calc_parabola_params(x1, y1, x2, y2, x3, y3):
    '''
    Adapted and modifed to get unknowns for defining a parabola:
    http://stackoverflow.com/questions/717762/how-to-calculate-the-vertex-of-a-parabola-given-three-points
    '''
    denom = (x1-x2) * (x1-x3) * (x2-x3)
    a     = ((x3 * (y2-y1)) + (x2 * (y1-y3)) + (x1 * (y3-y2))) / denom
    b     = ((x3**2 * (y1-y2)) + (x2**2 * (y3-y1)) + (x1**2 * (y2-y3))) / denom
    return a, b


# KF track fit in xy plane and zr plane
def KF_track_fit_moliere(sigma0, coords):
    # initialize 3D Kalman Filter for xy plane
    yf = coords[0][1]                               # observed y (coords is a list: [(x, y, z, r), (), ...])
    f = KalmanFilter(dim_x=3, dim_z=1)
    f.x = np.array([yf, 0., 0.])                    # X state vector [yf, dy/dx, w] = [coordinate, track inclination, integrated OU]
    f.H = np.array([[1., 0., 0.]])                  # H measurement matrix
    f.P = np.array([[sigma0**2,  0., 0.],     
                    [0.,         1., 0.],
                    [0.,         0., 1.]])          # P covariance
    f.R = sigma0**2                                 # R measuremnt noise
    saver = common.Saver(f)
    chi2_dists = []

    # initialize 2D KF for zr plane
    zf = coords[0][3]                               # observed z (coords is a list: [(x, y, z, r), (), ...])
    g = KalmanFilter(dim_x=2, dim_z=1)
    g.x = np.array([zf, 0.])                   
    g.H = np.array([[1., 0.]])                      # H measurement matrix
    g.P = np.array([[sigma0**2,  0.],     
                    [0.,         1000.]])           # P covariance
    g.R = sigma0**2                                 # R measuremnt noise
    g_saver = common.Saver(g)
    g_chi2_dists = []

    # track following & fit: process all coords in track candidate
    for i in range(len(coords)-1):
        # calculate parabolic parameters using 3 coords: origin, current node & next node
        x1, y1 = .0, .0
        x2, y2 = coords[i][0], coords[i][1]
        x3, y3 = coords[i+1][0], coords[i+1][1]
        a, b = calc_parabola_params(x1, y1, x2, y2, x3, y3)

        # calculation of kappa & radius of curvature ( r = sqrt(x^2 + y^2) )
        z2, r2 = coords[i][2], coords[i][3]
        z3, r3 = coords[i+1][2], coords[i+1][3]
        dr = r3 - r2
        dz = z3 - z2
        hyp = np.sqrt(dr**2 + dz**2)
        sin_t = np.abs(dr) / hyp
        tan_t = np.abs(dr) / np.abs(dz)
        kappa = (2*a) / (1 + ((2 * a * x3) + b)**2)**1.5

        # Moliere Theory - Highland formula multiple scattering error
        var_ms = sin_t * ((13.6 * 1e-3 * np.sqrt(0.02) * kappa) / 0.3)**2
        if np.abs(z3) >= 600.0: 
            # endcap - orientation of detector layers are vertical
            var_ms = var_ms * tan_t
        
        # variables for F; state transition matrix
        dx = x3 - x2
        alpha = 0.1                                     # OU parameter
        e1 = np.exp(-np.abs(dx) * alpha)
        f1 = (1.0 - e1) / alpha
        g1 = (np.abs(dx) - f1) / alpha

        # variables for Q process noise matrix
        sigma_ou = 0.00001                              # 10^-5
        sw2 = sigma_ou**2                               # OU parameter 
        st2 = var_ms                                    # process noise representing Moliere multiple scattering
        dx2 = dx**2
        dxw2 = dx2 * sw2
        Q02 = 0.5*dxw2
        Q01 = dx*(st2 + Q02)
        Q12 = dx*sw2

        # F state transition matrix, extrapolation Jacobian - linear & OU
        f.F = np.array([[1.,    dx,     g1], 
                        [0.,    1.,     f1],
                        [0.,    0.,     e1]])
        
        # Q process uncertainty/noise, OU model
        f.Q = np.array([[dx2*(st2 + 0.25*dxw2), Q01,        Q02], 
                      [Q01,                     st2 + dxw2, Q12],
                      [Q02,                     Q12,        sw2]])

        # KF predict and update
        measurement = coords[i+1][1] # observed y
        f.predict()
        f.update(measurement)
        saver.save()

        # update & calculate chi2 distance
        updated_state, updated_cov = f.x_post, f.P_post
        residual = measurement - f.H.dot(updated_state) 
        S = f.H.dot(updated_cov).dot(f.H.T) + f.R
        inv_S = np.linalg.inv(S)
        chi2_dist = residual.T.dot(inv_S).dot(residual)
        chi2_dists.append(chi2_dist)

        # KF for zr plane
        g.F = np.array([[1.,    dz], 
                        [0.,    1.]])                      # F state transition matrix, extrapolation Jacobian - linear & OU
    
        g.Q = var_ms                                       # Q process uncertainty/noise, OU model

        # KF predict and update
        measurement = coords[i+1][3] # observed r
        g.predict()
        g.update(measurement)
        g_saver.save()

        # update & calculate chi2 distance
        updated_state, updated_cov = g.x_post, g.P_post
        residual = measurement - g.H.dot(updated_state) 
        S = g.H.dot(updated_cov).dot(g.H.T) + g.R
        inv_S = np.linalg.inv(S)
        chi2_dist = residual.T.dot(inv_S).dot(residual)
        g_chi2_dists.append(chi2_dist)

    # chi2 probability distribution
    total_chi2 = sum(chi2_dists)                        # chi squared statistic
    dof = len(coords) - 2                               # (no. of measurements * 1D) - no. of track params
    pval = distributions.chi2.sf(total_chi2, dof)

    # chi2 probability distribution
    total_chi2 = sum(g_chi2_dists)                      # chi squared statistic
    pval_zr = distributions.chi2.sf(total_chi2, dof)
    print("P values for KF track fit in xy and zr planes: ", pval, pval_zr)
    
    return pval, pval_zr



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
    parser.add_argument('-t', '--threshold_distance_node_merging', help="threshold_distance_node_merging")
    parser.add_argument('-e', '--error', help="sigma0 rms of track position measurements")
    parser.add_argument('-n', '--numhits', help="minimum number of hits for good track candidate")
    parser.add_argument('-a', '--iteration', help="iteration number of algorithm")
    args = parser.parse_args()

    # set variables
    inputDir = args.input
    candidatesDir = args.candidates
    remainingDir = args.remain
    fragmentsDir = args.fragments
    track_acceptance = float(args.pval)
    sigma0 = float(args.error)
    subgraph_path = "_subgraph.gpickle"
    fragment = int(args.numhits)
    separation_3d_threshold = float(args.separation_3d_threshold)
    threshold_distance_node_merging = float(args.threshold_distance_node_merging)
    # get iteration num
    iteration_num = str(args.iteration)

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

        candidate_to_remove_from_subGraph = []
        for n, candidate in enumerate(potential_tracks):
            print("Processing candidate: ", n)

            #TODO: need to check for holes?

            # check for track fragments
            if candidate.number_of_nodes() >= fragment:
                
                # check for close proximity nodes based on their distance & common module_id values - merge where appropriate
                # if merged_candidate is not None, then we need to use merged_candidate's coords in the KF, but extract the original candidate
                candidate, merged_candidate = check_close_proximity_nodes(candidate, threshold_distance_node_merging)
                
                candidate_to_assess = candidate
                if merged_candidate is not None: candidate_to_assess = merged_candidate

                # check for 1 hit per layer - use volume_id & in_volume_layer_id
                vivl_id_values = nx.get_node_attributes(candidate_to_assess,'vivl_id').values()

                if (len(vivl_id_values) == len(set(vivl_id_values))) and (len(set(vivl_id_values)) >= fragment): 
                    # good candidate
                    # print("no duplicates volume_ids & in_volume_layer_ids for this candidate")

                    # sort the candidates by radius largest to smallest, tuples: node_num, (x,y,z,r)
                    nodes_coords_tuples = list(nx.get_node_attributes(candidate_to_assess, 'xyzr').items())
                    sorted_nodes_coords_tuples = sorted(nodes_coords_tuples, reverse=True, key=lambda item: item[1][3])
                    coords = [element[1] for element in sorted_nodes_coords_tuples]
                    # rotate the track such that innermost edge parallel to x-axis - r&z components are left unchanged
                    coords = rotate_track(coords, separation_3d_threshold)

                    # KF track fit - Moliere theory multiple scattering
                    pval, pval_zr = KF_track_fit_moliere(sigma0, coords)
                    if (pval >= track_acceptance) and (pval_zr >= track_acceptance):
                        print("Good KF fit, p-value:", pval, "\n(x,y,z,r):", coords)
                        extracted.append(candidate)
                        extracted_pvals.append(pval)
                        extracted_pvals_zr.append(pval_zr)
                        candidate_to_remove_from_subGraph.append(candidate) 
                    else:
                        print("p-value too small, leave for further processing, pval_xy: " + str(pval) + " pval_zr: " + str(pval_zr))

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
        num_nodes_in_subgraph = len(subGraph.nodes())
        if (num_nodes_in_subgraph < fragment) and (num_nodes_in_subgraph > 0): 
            fragments.append(subGraph)
        elif num_nodes_in_subgraph >= fragment:
            remaining.append(subGraph)

    
    # attach iteration number to good extracted tracks
    for subGraph in extracted:
        subGraph.graph["iteration"] = iteration_num

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

    # save p-value information
    pvals_df = pd.DataFrame({'pvals_xy' : extracted_pvals, 'pvals_zr' : extracted_pvals_zr}) 
    pvals_df.to_csv(candidatesDir + 'pvals.csv')

    # save extracted tracks
    for i, sub in enumerate(extracted):
        h.save_network(candidatesDir, i, sub) # save network to serialized form

    # # plot and save all extracted candidates from previous and this iteration
    # h.plot_save_subgraphs_iterations(extracted, candidatesDir, "Extracted candidates", node_labels=True, save_plot=True)

    # # plot and save the remaining subgraphs to be further processed
    # h.plot_subgraphs(remaining, remainingDir, node_labels=True, save_plot=True, title="Remaining candidates")
    
    # save remaining and track fragments
    for i, sub in enumerate(remaining):
        h.save_network(remainingDir, i, sub)
    for i, sub in enumerate(fragments):
        h.save_network(fragmentsDir, i, sub)



if __name__ == "__main__":
    main()