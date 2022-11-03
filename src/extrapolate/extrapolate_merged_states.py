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


def extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, chi2CutFactor, sigma_ms):
    print("\n Extrapolate validate:")
    print("Processing node num: ", node_num)

    # global coordinates of node and neighbour
    node_x = node_attr['GNN_Measurement'].x
    node_y = node_attr['GNN_Measurement'].y
    neighbour_x = neighbour_attr['GNN_Measurement'].x
    neighbour_y = neighbour_attr['GNN_Measurement'].y
    print("global nodeA x,y: ", node_x, node_y)
    print("global target nodeC x,y: ", neighbour_x, neighbour_y)

    # compute the angle of rotation
    angle_of_rotation_C = atan2(node_y, node_x)
    angle_of_rotation_C_deg = angle_of_rotation_C * 180 / np.pi
    print("phi in rad: ", angle_of_rotation_C)
    print("phi in deg:", angle_of_rotation_C_deg)

    # Change coordinate systems! Transform coordinate axis nodeA into coord axis of nodeC
    # calculation of x_A and y_A (coordinates x and y of node A in c.s. of node C)
    nodeA_trans_x = node_attr['translation'][0]         # global translation
    nodeA_trans_y = node_attr['translation'][1]
    nodeA_x = node_attr['GNN_Measurement'].x        # global coordinates
    nodeA_y = node_attr['GNN_Measurement'].y
    nodeC_trans_x = neighbour_attr['translation'][0]    # global translation
    nodeC_trans_y = neighbour_attr['translation'][1]
    x_A = (neighbour_x - node_x)*np.cos(angle_of_rotation_C) + (neighbour_y - node_y)*np.sin(angle_of_rotation_C)
    y_A = -(neighbour_x - node_x)*np.sin(angle_of_rotation_C) + (neighbour_y - node_y)*np.cos(angle_of_rotation_C)
    print("transformed coordinates of nodeA in c.s. of nodeC (x,y): ", x_A, y_A)
 
    # parabolic track state (and parameteres) at node A
    merged_state = node_attr['merged_state']
    a, b, c = merged_state[0], merged_state[1], merged_state[2]
    print("original parabolic parameters: ", a, b, c)

    phi = atan2( (node_x*neighbour_y) - (node_y*neighbour_x), (node_x*neighbour_x) + (node_y*neighbour_y) )
    phi_deg = phi * 180 / np.pi
    print("global phi version 2 between node A and node C (relative angle):")
    print("phi_version_2 in rad: ", phi)
    print("phi_version_2 in deg:", phi_deg)

    # calculation of track position/parameters in the target c.s. (nodeC)
    x_prime = x_A + (c * np.sin(phi))     # x_prime = x_A + scos(phi) + (as**2 + bs + c)*sin(phi), when s=0
    Vx_prime = np.cos(phi) + (b * np.sin(phi))
    Ax_prime = a * np.sin(phi)
    print("x_prime, Vx_prime, Ax_prime: ", x_prime, Vx_prime, Ax_prime)

    # s* substitution - from solution of quadratic equation
    s_star = (- x_prime * ((2 * Vx_prime**2) + (Ax_prime * x_prime))) / (2 * Vx_prime**3)
    print("step size: ", s_star)
    with open('s_star.csv', 'a') as f:
        f.write(str(s_star) + "\n")

    # compute y', Vy' and Ay': transformed parameters in the c.s. of nodeC
    y_prime = y_A - (s_star * np.sin(phi)) + ((a*s_star**2 + b*s_star + c)*np.cos(phi))
    # TODO check this stage:
    Vy_prime = - np.sin(phi) + (b * np.cos(phi))    # taking the terms with s from y_prime equation
    Ay_prime = a * np.cos(phi)                      # taking the terms with s**2 from y_prime equation
    print("parameter a: ", a)
    print("np.cos(phi): ", np.cos(phi))
    print("y_prime, Vy_prime, Ay_prime: ", y_prime, Vy_prime, Ay_prime)

    # compute new (extrapolated) parabolic parameters at nodeC
    x_c = 0     # condition
    y_c = y_prime + (Vy_prime * s_star) + (0.5 * Ay_prime * s_star**2)  # y_c = parabolic parmeter c and nodeC
    b_c = (Vy_prime + (s_star * Ay_prime)) / (Vx_prime + (s_star * Ax_prime))
    a_c = Ay_prime

    # calculating the Jacobian
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

    # extrapolation jacobian
    F = np.array([[da_da,    da_db,     da_dc], 
                  [db_da,    db_db,     db_dc],
                  [dc_da,    dc_db,     dc_dc]])      # F state transition matrix, extrapolation Jacobian

    merged_cov = node_attr['merged_cov']
    sigma0 = node_attr['GNN_Measurement'].sigma0
    extrp_state = F.dot(merged_state)
    extrp_cov = F.dot(merged_cov).dot(F.T)

    print("extrapolated ac, bc, yc from computation: ", a_c, b_c, y_c)
    print("extrapolated track state from F matrix (derivations): ", extrp_state)
    print("extrapolated covariance: \n", extrp_cov)
    print("Jacobian: \n", F)

    # validate the extrapolated state against the measurement at the neighbour node
    # calc chi2 distance between measurement at neighbour node and extrapolated track state
    H = np.array([[0., 0., 1.]])
    print("H vector: ", H)
    # residual = neighbour_y - H.dot(extrp_state)     # compute the residual
    neighbour_y_in_cs_nodeC = .0                      # measurement is always zero in c.s. of neighbour
    residual = neighbour_y_in_cs_nodeC - H.dot(extrp_state)              # compute the residual
    S = H.dot(extrp_cov).dot(H.T) + sigma0**2       # covariance of residual (denominator of kalman gain)
    inv_S = np.linalg.inv(S)
    chi2 = residual.T.dot(inv_S).dot(residual)
    chi2_cut = chi2CutFactor

    print("residual: ", residual)
    print("S matrix: ", S)
    print("inverse S: ", inv_S)

    # save chi2 distance data - USED FOR TUNING THE CHI2 CUT FACTOR
    # truth, chi2 distance, chi2_cut
    node_truth = subGraph.nodes[node_num]["truth_particle"]
    neighbour_truth = subGraph.nodes[neighbour_num]["truth_particle"]
    truth = 0
    if node_truth == neighbour_truth: truth = 1
    line = str(truth) + " " + str(chi2) + " " + str(chi2_cut) + "\n"
    with open('chi2_data_sigma_0.1.csv', 'a') as f:
        f.write(line)

    print("chi2 cut: ", chi2_cut)
    print("chi2 distance: ", chi2)

    # validate chi2 distance
    if chi2 < chi2_cut:
        print("chi2 OK, performing KF update...")

        # calc beta: measurement likelihood
        factor = 2 * math.pi * np.abs(S)
        norm_factor = math.pow(factor, -0.5)
        likelihood = norm_factor * np.exp(-0.5 * chi2)
        print("likelihood: ", likelihood)

        # initialize KF
        f = KalmanFilter(dim_x=3, dim_z=1)
        f.x = extrp_state                   # X state vector
        f.F = F                             # F state transition matrix
        f.H = H                             # H measurement matrix
        f.P = extrp_cov
        f.R = sigma0**2
        f.Q = np.array([[0.,     0.,            0.], 
                        [0.,     sigma_ms**2,   0.],
                        [0.,     0.,            0.]])      # Q process uncertainty/noise
        # z = neighbour_y                     # "sensor reading"
        z = neighbour_y_in_cs_nodeC         # "sensor reading"

        # perform KF update & save data
        f.predict()
        f.update(z)
        updated_state, updated_cov = f.x_post, f.P_post

        return { 'xy': (node_x, node_y),
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



def message_passing(subGraphs, chi2CutFactor, sigma_ms):
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
                        updated_state_dict = extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, chi2CutFactor, sigma_ms)
                        if updated_state_dict != None:
                            # store the updated track states at the neighbour node
                            if 'updated_track_states' not in neighbour_attr:
                                subGraph.nodes[neighbour_num]['updated_track_states'] = {node_num : updated_state_dict}
                            else:
                                subGraph.nodes[neighbour_num]['updated_track_states'][node_num] = updated_state_dict
                    else:
                        print("edge", node_num, neighbour_num, "not activated, message not transmitted")

            else: 
                # TODO: extrapolate all track state estimates?
                print("No merged state found, leaving for further iterations")


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
    parser.add_argument('-m', '--sigma_ms', help="uncertainty due to multiple scattering, process noise")
    args = parser.parse_args()
    inputDir = args.inputDir
    outputDir = args.outputDir
    chi2CutFactor = float(args.chi2CutFactor)
    sigma_ms = float(args.sigma_ms)                # process error - due to multiple scattering

    # read in subgraph data
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)


    # distribute merged state to neighbours, extrapolate, validation & create new updated state(s)
    message_passing(subGraphs, chi2CutFactor, sigma_ms)
    convert_single_updated_state(subGraphs)

    h.compute_prior_probabilities(subGraphs, 'updated_track_states')
    h.reweight(subGraphs, 'updated_track_states')
    h.compute_prior_probabilities(subGraphs, 'updated_track_states')

    # for i, s in enumerate(subGraphs):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")
    #     print("EDGE DATA:", s.edges.data(), "\n")

    title = "Subgraphs after iteration 2: message passing, extrapolation \n& validation of merged state, formation of updated state"
    h.plot_subgraphs(subGraphs, outputDir, node_labels=True, save_plot=True, title=title)
    # save networks
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)



if __name__ == "__main__":
    main()