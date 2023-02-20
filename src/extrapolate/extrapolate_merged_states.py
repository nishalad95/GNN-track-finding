import os, glob
from filterpy.kalman.kalman_filter import update
import numpy as np
import math
from math import atan2
import networkx as nx
from filterpy.kalman import KalmanFilter
from filterpy import common
import argparse
from utilities import helper as h
import pprint


# inverse variance-weighting: multivariate case https://en.wikipedia.org/wiki/Inverse-variance_weighting#Multivariate_Case
def merge_states(mean1, cov1, mean2, cov2):
    inv1 = np.linalg.inv(cov1)
    inv2 = np.linalg.inv(cov2)
    sum_inv_covs = inv1 + inv2
    merged_cov = np.linalg.inv(sum_inv_covs)
    merged_mean = inv1.dot(mean1) + inv2.dot(mean2)
    merged_mean = merged_cov.dot(merged_mean)
    merged_inv_cov = np.linalg.inv(merged_cov)
    return merged_mean, merged_cov


def extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, 
                        chi2CutFactor, state_to_extrapolate, state_cov, is_merged_state=False):

    # global coordinates of node and neighbour
    node_x = node_attr['GNN_Measurement'].x
    node_y = node_attr['GNN_Measurement'].y
    node_z = node_attr['GNN_Measurement'].z
    node_r = node_attr['GNN_Measurement'].r
    neighbour_x = neighbour_attr['GNN_Measurement'].x
    neighbour_y = neighbour_attr['GNN_Measurement'].y
    neighbour_z = neighbour_attr['GNN_Measurement'].z
    neighbour_r = neighbour_attr['GNN_Measurement'].r

    # compute the angle of rotation
    angle_of_rotation_C = atan2(node_y, node_x)
    angle_of_rotation_C_deg = angle_of_rotation_C * 180 / np.pi

    # Change coordinate systems! Transform coordinate axis nodeA into coord axis of nodeC
    # calculation of x_A and y_A (coordinates x and y of node A in c.s. of node C)
    nodeA_trans_x = node_attr['translation'][0]         # global translation
    nodeA_trans_y = node_attr['translation'][1]
    nodeA_x = node_x                                    # global coordinates
    nodeA_y = node_y
    nodeC_trans_x = neighbour_attr['translation'][0]    # global translation
    nodeC_trans_y = neighbour_attr['translation'][1]
    x_A = (neighbour_x - node_x)*np.cos(angle_of_rotation_C) + (neighbour_y - node_y)*np.sin(angle_of_rotation_C)
    y_A = -(neighbour_x - node_x)*np.sin(angle_of_rotation_C) + (neighbour_y - node_y)*np.cos(angle_of_rotation_C)
    # print("transformed coordinates of nodeA in c.s. of nodeC (x,y): ", x_A, y_A)
 
    # parabolic track state (and parameteres) at node A
    merged_state = state_to_extrapolate
    a, b, c = merged_state[0], merged_state[1], merged_state[2]
    phi = atan2( (node_x*neighbour_y) - (node_y*neighbour_x), (node_x*neighbour_x) + (node_y*neighbour_y) )
    phi_deg = phi * 180 / np.pi

    # calculation of track position/parameters in the target c.s. (nodeC)
    x_prime = x_A + (c * np.sin(phi))     # x_prime = x_A + scos(phi) + (as**2 + bs + c)*sin(phi), when s=0
    Vx_prime = np.cos(phi) + (b * np.sin(phi))
    Ax_prime = a * np.sin(phi)

    # s* substitution - from solution of quadratic equation
    s_star = (- x_prime * ((2 * Vx_prime**2) + (Ax_prime * x_prime))) / (2 * Vx_prime**3)

    # compute y', Vy' and Ay': transformed parameters in the c.s. of nodeC
    y_prime = y_A - (s_star * np.sin(phi)) + ((a*s_star**2 + b*s_star + c)*np.cos(phi))
    Vy_prime = - np.sin(phi) + (b * np.cos(phi))    # taking the terms with s from y_prime equation
    Ay_prime = a * np.cos(phi)                      # taking the terms with s**2 from y_prime equation

    # compute new (extrapolated) parabolic parameters at nodeC
    x_c = 0     # condition
    y_c = y_prime + (Vy_prime * s_star) + (0.5 * Ay_prime * s_star**2)  # y_c = parabolic parmeter c and nodeC
    b_c = (Vy_prime + (s_star * Ay_prime)) / (Vx_prime + (s_star * Ax_prime))
    a_c = Ay_prime

    # partial s* as a function of parabolic parameters a, b, c
    numer = x_A + c*np.sin(phi)
    denom = np.cos(phi) + b*np.sin(phi)
    ds_da = - (np.sin(phi) * numer**2 ) / denom**3
    ds_db = ((np.sin(phi) * numer) * (1 + ((3 * a * np.sin(phi) * numer) / denom**2))) / denom**2
    ds_dc = - np.sin(phi) * (1 + ((2 * a * np.sin(phi) * numer) / denom**2)) / denom

    # partial a' : da'/da, da'/db, da'dc
    denom = np.cos(phi) + ((2*a + b) * np.sin(phi))
    da_da = (1 / denom**3) * (1 - ((6 * a * np.sin(phi)) * (s_star + a * ds_da) / denom))
    da_db = (-3 * a * np.sin(phi) * ((2 * a * ds_db) + 1)) / denom**4
    da_dc = (-6 * np.sin(phi) * ds_dc * a**2) / denom**4

    # partial b' : db'/da, db'/db, db'dc
    denom = np.cos(phi) + ((2*a*s_star + b) * np.sin(phi))
    bracket = np.cos(phi) - ((np.sin(phi) * (-np.sin(phi) + ((2*a*s_star + b)*np.cos(phi))) ) / denom )
    db_da = (2 * (s_star + a * ds_da) * bracket) / denom
    db_db = ((1 + (2 * a * ds_da)) * bracket) / denom
    db_dc = (2 * a * ds_dc * bracket) / denom

    # partial c' : dc'/da, dc'/db, dc'dc
    bracket = (np.cos(phi) * (2*a + b)) - np.sin(phi)
    dc_da = (ds_da * bracket) + (s_star**2 * np.cos(phi))
    dc_db = (ds_db * bracket) + (s_star * np.cos(phi))
    dc_dc = (ds_dc * bracket) + np.cos(phi)

    # extrapolation Jacobian
    F = np.array([[da_da,    da_db,     da_dc], 
                  [db_da,    db_db,     db_dc],
                  [dc_da,    dc_db,     dc_dc]])      # F state transition matrix, extrapolation Jacobian

    merged_cov = state_cov
    sigma0 = node_attr['GNN_Measurement'].sigma0
    extrp_state = F.dot(merged_state)
    extrp_cov = F.dot(merged_cov).dot(F.T)

    # validate the extrapolated state against the measurement at the neighbour node
    # calc chi2 distance between measurement at neighbour node and extrapolated track state
    H = np.array([[0., 0., 1.]])
    # residual = neighbour_y - H.dot(extrp_state)               # compute the residual
    neighbour_y_in_cs_nodeC = .0                                # measurement is always zero in c.s. of neighbour
    residual = neighbour_y_in_cs_nodeC - H.dot(extrp_state)     # compute the residual
    S = H.dot(extrp_cov).dot(H.T) + sigma0**2                   # covariance of residual (denominator of kalman gain)
    inv_S = np.linalg.inv(S)
    chi2 = residual.T.dot(inv_S).dot(residual)
    chi2_cut = chi2CutFactor

    # save chi2 distance data - USED FOR TUNING THE CHI2 CUT FACTOR
    # truth, chi2 distance, chi2_cut
    # node_truth = subGraph.nodes[node_num]["truth_particle"]
    # neighbour_truth = subGraph.nodes[neighbour_num]["truth_particle"]
    # truth = 0
    # if node_truth == neighbour_truth: truth = 1
    # line = str(truth) + " " + str(chi2) + " " + str(chi2_cut) + "\n"
    # with open('chi2_data_in_extrapolation.csv', 'a') as f:
    #     f.write(line)

    # validate chi2 distance
    if chi2 <= chi2_cut:
        # calculate beta: measurement likelihood
        factor = 2 * math.pi * np.abs(S)
        norm_factor = math.pow(factor, -0.5)
        likelihood = norm_factor * np.exp(-0.5 * chi2)

        # Moliere Theory - Highland formula multiple scattering
        # calculation of hypotenuse using a pair of hits (track segment)   
        dr = node_r - neighbour_r
        dz = node_z - neighbour_z
        hyp = np.sqrt(dr**2 + dz**2)
        sin_t = np.abs(dr) / hyp
        tan_t = np.abs(dr) / np.abs(dz)
        # kappa and radius of curvature
        kappa = (2*a) / (1 + ((2*a*neighbour_x) + b)**2)**1.5
        var_ms = sin_t * ((13.6 * 1e-3 * np.sqrt(0.02) * kappa) / 0.3)**2
        if np.abs(neighbour_z) >= 600.0: 
            # endcap - orientation of detector layers are vertical
            var_ms = var_ms * tan_t

        # initialize KF
        f = KalmanFilter(dim_x=3, dim_z=1)
        f.x = extrp_state                   # X state vector
        f.F = F                             # F state transition matrix
        f.H = H                             # H measurement matrix
        f.P = extrp_cov
        f.R = sigma0**2
        f.Q = np.array([[0.,     0.,       0.], 
                        [0.,     var_ms,   0.],
                        [0.,     0.,       0.]])      # Q process uncertainty/noise

        # z = neighbour_y                     # "sensor reading"
        z = neighbour_y_in_cs_nodeC         # "sensor reading"

        # perform KF update & save data
        f.predict()
        f.update(z)
        updated_state, updated_cov = f.x_post, f.P_post

        # form tau and variance in tau
        tau = dz / dr
        # default error for barrel located node
        sigma_r = 0.1
        sigma_z = 0.5
        # if node in endcap, then reduce z error: used in calc of error in tau = dz/dr
        if np.abs(node_z) >= 600.0: 
            sigma_z = 0.1
            sigma_r = 0.5
        del_dz = sigma_z
        del_dr = sigma_r
        # error in tau
        del_tau = tau * np.sqrt((del_dz / dz)**2 + (del_dr / dr)**2)
        if dz == 0:
            del_tau = tau * np.sqrt((del_dr / dr)**2)
        variance_tau = del_tau**2

        # form the joint vector state and joint vector covariance
        joint_vector = [updated_state[0], updated_state[1], tau]
        joint_vector_covariance = updated_cov
        joint_vector_covariance[:, 2] = 0.0
        joint_vector_covariance[2, :] = 0.0
        joint_vector_covariance[2, 2] = variance_tau + var_ms

        return { 'xy': (node_x, node_y),
                 'zr': (node_z, node_r),
                 'xyzr': (node_x, node_y, node_z, node_r),
                 'edge_state_vector': updated_state, 
                 'edge_covariance': updated_cov,
                #  'joint_vector': joint_vector,
                #  'joint_vector_covariance': joint_vector_covariance,
                 'joint_vector': updated_state,
                 'joint_vector_covariance': updated_cov,
                 'likelihood': likelihood,
                 # previous weight - gets updated at the end of extrapolation
                 'mixture_weight': subGraph.nodes[node_num]['track_state_estimates'][neighbour_num]['mixture_weight']
                }, chi2, 0, 0

    else:
        perc_correct_outliers_detected = 0
        total_outliers = 0
        if is_merged_state:
            print("chi2 distance too large")
            print("DEACTIVATING edge connection ( ", node_num, ", ", neighbour_num, " )")
            subGraph[node_num][neighbour_num]["activated"] = 0

            # precision in outlier masking
            node_truth_particle = subGraph.nodes[node_num]['truth_particle']
            neighbour_truth_particle = subGraph.nodes[neighbour_num]['truth_particle']
            if node_truth_particle != neighbour_truth_particle:
                perc_correct_outliers_detected += 1
            total_outliers += 1

        return None, chi2, perc_correct_outliers_detected, total_outliers



def message_passing(subGraphs, chi2CutFactor):
    number_of_nodes_with_merged_state = 0
    perc_correct_outliers_detected = 0
    total_outliers = 0

    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        # extrapolating outwards, from node to neighbour
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]
            # print("\nProcessing node: ", node_num)

            # CASE 1: message passing 'merged' (parabolic state)
            if 'merged_state' in node_attr.keys():
                number_of_nodes_with_merged_state += 1
                state_to_extrapolate = node_attr['merged_state']
                state_cov = node_attr['merged_cov']
                # extrapolate merged state to all neighbours
                for neighbour_num in subGraph.neighbors(node_num):
                    if subGraph[node_num][neighbour_num]["activated"] == 1:
                        neighbour_attr = subGraph.nodes[neighbour_num]
                        updated_state_dict, chi2, numerator, denominator = extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, 
                                                                chi2CutFactor, state_to_extrapolate, state_cov, is_merged_state=True)
                        
                        perc_correct_outliers_detected += numerator
                        total_outliers += denominator

                        if updated_state_dict != None:
                            # store the updated track states at the neighbour node
                            if 'updated_track_states' not in neighbour_attr:
                                subGraph.nodes[neighbour_num]['updated_track_states'] = {node_num : updated_state_dict}
                            else:
                                stored_dict = subGraph.nodes[neighbour_num]['updated_track_states']
                                subGraph.nodes[neighbour_num]['updated_track_states'][node_num] = updated_state_dict
                        else:
                            print("node number: ", node_num, "chi2 distance was too large: ", chi2, "leaving for further iterations")
                    else:
                        print("edge", node_num, neighbour_num, "not activated, message not transmitted")
            # CASE 2: extrapolate all states (parabolic) and find the 2 states with the smallest chi2 distance to the neighbour --> merge these into a new updated state
            # else:
            #     # extrapolate all parabolic states to all neighbours --> manually form an updated track state
            #     for neighbour_num in subGraph.neighbors(node_num):
            #         if subGraph[node_num][neighbour_num]["activated"] == 1:
            #             neighbour_attr = subGraph.nodes[neighbour_num]

            #             # extrapolate all track states (parabolic) 
            #             all_chi2s = []
            #             all_updated_state_dicts = []
            #             all_states_to_extrapolate = node_attr['track_state_estimates']
            #             all_states_list = list(node_attr['track_state_estimates'].values())
            #             if len(all_states_list) >= 2:
            #                 for s in all_states_to_extrapolate.values():
            #                     state_to_extrapolate = s['edge_state_vector']
            #                     state_cov = s['edge_covariance']
            #                     updated_state_dict, chi2 = extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, 
            #                                                                 chi2CutFactor, state_to_extrapolate, state_cov)
            #                     all_updated_state_dicts.append(updated_state_dict)
            #                     all_chi2s.append(chi2)
                            
            #                 # find the 2 smallest chi2 values and merge their states
            #                 sorted_chi2s = sorted(all_chi2s)
            #                 smallest_chi2_1 = sorted_chi2s[0]
            #                 smallest_chi2_2 = sorted_chi2s[1]
            #                 if (smallest_chi2_1 < 0.25 and smallest_chi2_2 < 0.25):
            #                     idx1 = all_chi2s.index(smallest_chi2_1)
            #                     idx2 = all_chi2s.index(smallest_chi2_2)
            #                     state1 = all_states_list[idx1]
            #                     state2 = all_states_list[idx2]
            #                     mean1, mean2 = state1["edge_state_vector"], state2["edge_state_vector"]
            #                     cov1, cov2 = state1["edge_covariance"], state2["edge_covariance"]
            #                     merged_state, merged_cov = merge_states(mean1, cov1, mean2, cov2)
            #                     state1["edge_state_vector"] = merged_state
            #                     state1["edge_covariance"] = merged_cov

            #                     updated_state_dict = state1
            #                     # store the updated track states at the neighbour node
            #                     if 'updated_track_states' not in neighbour_attr:
            #                         subGraph.nodes[neighbour_num]['updated_track_states'] = {node_num : updated_state_dict}
            #                     else:
            #                         stored_dict = subGraph.nodes[neighbour_num]['updated_track_states']
            #                         subGraph.nodes[neighbour_num]['updated_track_states'][node_num] = updated_state_dict

    print("Total number of nodes with merged state: ", number_of_nodes_with_merged_state)
    print("numerator:", perc_correct_outliers_detected, "denominator:", total_outliers)
    if total_outliers != 0:
        perc_correct_outliers_detected = (perc_correct_outliers_detected / total_outliers) * 100
        print("Percentage of correct outliers detected:", perc_correct_outliers_detected)          


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
        subGraphs.append(sub)

    print("Statistics before extrapolation:")
    print("Total number of subgraphs to be processed: ", len(subGraphs))
    total_num_nodes = 0
    total_num_edges = 0
    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        total_num_nodes += len(subGraph.nodes())
        total_num_edges += len(subGraph.edges())
    print("total number of nodes to be processed: ", total_num_nodes)
    print("total number of edges to be processed: ", total_num_edges)

    print("Beginning message passing...")
    # distribute merged state to neighbours, extrapolate, validation & create new updated state(s)
    message_passing(subGraphs, chi2CutFactor)

    h.compute_prior_probabilities(subGraphs, 'updated_track_states')
    h.reweight(subGraphs, 'updated_track_states')

    # TODO: Check with and without this during the next iteration
    # RECALCULATE priors and mixture weights again using same reweight threshold as edges are deactivated!
    h.compute_prior_probabilities(subGraphs, 'updated_track_states')
    h.reweight(subGraphs, 'updated_track_states')

    # update the node degree as an attribute - number of active updated track states
    for subGraph in subGraphs:
        for node in subGraph.nodes(data=True):
            node_num = node[0]
            degree = h.query_node_degree_in_edges(subGraph, node_num)
            subGraph.nodes[node_num]['degree'] = degree
            node_attr = node[1]

    # for i, s in enumerate(subGraphs):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")
        # print("EDGE DATA:", s.edges.data(), "\n")

    # save networks
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)



if __name__ == "__main__":
    main()