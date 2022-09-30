import os, glob
from filterpy.kalman.kalman_filter import update
import numpy as np
import math
import networkx as nx
from filterpy.kalman import KalmanFilter
from filterpy import common
import argparse
from utilities import helper as h
import pprint



def extrapolate_validate(subGraph, node_num, node_attr, neighbour_num, neighbour_attr, chi2CutFactor, sigma_ms):
    # 1) transformation of parametric t_vector from surface bound to surface free plane
    # 2) extrapolation of the tvector 
    # 3) calculation of new extrapolated parabolic parameters
    # transform the t_vector from the node coord sys to its neighbour node coord sys
    nodeA_angle = node_attr['angle_of_rotation']
    nodeA_trans = node_attr['translation']
    tvector = node_attr['track_state_estimates'][neighbour_num]['t_vector']
    nodeB_angle = neighbour_attr['angle_of_rotation']
    nodeB_trans = neighbour_attr['translation']
    print("6 component tvector: ", tvector)
    tvector_coords = tvector[:2]
    x_a = tvector_coords[0]
    tvector_vel = tvector[2:4]
    tvector_acc = tvector[4:]
    # convert node coord sys to global
    # undo the translation of nodeA's tvector - affects only the first 2 components
    print("translation to apply: ", nodeA_trans)
    print("tvector_coords: ", tvector_coords)
    print("tvector_vel: ", tvector_vel)
    print("tvector_acc: ", tvector_acc)
    nodeA_trans = np.array([nodeA_trans[0], nodeA_trans[1]]) # x and y components of translation
    tvector_coords = tvector_coords + nodeA_trans
    print("translated coordinates: ", tvector_coords)
    # undo the rotation from nodeA's coord sys back to global, rotate clockwise
    # ( x_new = xcos(angle) + ysin(angle) and y_new = -xsin(angle) + ycos(angle) )
    rotation_matrix_to_global = np.array([[np.cos(nodeA_angle), np.sin(nodeA_angle)], 
                                          [-1*np.sin(nodeA_angle), np.cos(nodeA_angle)]])
    tvector_coords = rotation_matrix_to_global.dot(tvector_coords)
    tvector_vel = rotation_matrix_to_global.dot(tvector_vel)
    tvector_acc = rotation_matrix_to_global.dot(tvector_acc)
    # convert from global coord system to nodeB coord sys
    # apply the rotation from global coord system to nodeB
    # ( x_new = xcos(angle) - ysin(angle) and y_new = xsin(angle) + ycos(angle) )
    rotation_matrix_to_nodeB = np.array([[np.cos(nodeB_angle), -1*np.sin(nodeB_angle)], 
                                          [np.sin(nodeB_angle), np.cos(nodeB_angle)]])
    tvector_coords = rotation_matrix_to_nodeB.dot(tvector_coords)
    tvector_vel = rotation_matrix_to_nodeB.dot(tvector_vel)
    tvector_acc = rotation_matrix_to_nodeB.dot(tvector_acc)
    print("transformed vel: ", tvector_vel)
    print("transformed acc: ", tvector_acc)
    # apply the translation - affects only the first 2 components
    nodeB_trans = np.array([nodeB_trans[0], nodeB_trans[1]]) # x and y components of translation
    tvector_coords = tvector_coords - nodeB_trans
    # now the node's t_vector will be in the neighbour's coordinate system
    transformed_vector = np.array([tvector_coords[0], tvector_coords[1], 
                                   tvector_vel[0], tvector_vel[1], 
                                   tvector_acc[0], tvector_acc[1]])
    print("transformed vector: ", transformed_vector)

    # compute the step size for extrapolation - calculated from solving quadratic equation
    xt = transformed_vector[0]
    Vxt = transformed_vector[2]
    Axt = transformed_vector[4]
    print("transformed xt, Vxt and Axt: ", xt, Vxt, Axt)
    step = (-xt / Vxt) - ( (Axt * xt**2) / (2 * Vxt**3) )
    
    print("step: ", step)
    # temporary - for checking the step size
    with open('step.txt', 'a') as the_file:
        the_file.write(str(step) + '\n')

    # compute parabolic parameters
    yt = transformed_vector[1]
    Vyt = transformed_vector[3]
    Ayt = transformed_vector[5]
    xc = 0
    yc = yt + (Vyt * step) + (0.5 * Ayt * step**2)
    bc = (Vyt + (step * Ayt)) / (Vxt + (step * Axt))
    ac = Ayt

    # parabolic track state
    merged_state = node_attr['merged_state']
    a, b, c = merged_state[0], merged_state[1], merged_state[2]

    # partial s* as a function of parabolic parameters a, b, c
    theta = # TODO : calculate new theta angle between nodeA and nodeC
    numer = x_a + c*np.sin(theta)
    denom = np.cos(theta) + b*np.sin(theta)
    ds_da = - (np.sin(theta) * numer**2 ) / denom**3
    ds_db = ((np.sin(theta) * numer) * (1 + ((3 * a * np.sin(theta) * numer) / denom**2))) / denom**2
    ds_dc = - np.sin(theta) * (1 + ((2 * a * np.sin(theta) * numer) / denom**2)) / denom

    # partial a' : da'/da, da'/db, da'dc
    denom = np.cos(theta) + ((2*a + b) * np.sin(theta))
    da_da = (1 / denom**3) * (1 - ((6 * a * np.sin(theta)) * (step + a * ds_da) / denom))
    da_db = (-3 * a * np.sin(theta) * ((2 * a * ds_db) + 1)) / denom**4
    da_dc = (-6 * np.sin(theta) * ds_dc * a**2) / denom**4

    # partial b' : db'/da, db'/db, db'dc


    # partial c' : dc'/da, dc'/db, dc'dc


    # extrapolation jacobian
    F = np.array([[da_da,    da_db,     da_dc], 
                  [db_da,    db_db,     db_dc],
                  [dc_da,    dc_db,     dc_dc]])      # F state transition matrix, extrapolation Jacobian






    


    # OLD extrapolation technique...
    node_x = node_attr['GNN_Measurement'].x
    node_y = node_attr['GNN_Measurement'].y
    neighbour_x = neighbour_attr['GNN_Measurement'].x
    neighbour_y = neighbour_attr['GNN_Measurement'].y
    merged_state = node_attr['merged_state']
    merged_cov = node_attr['merged_cov']
    # TODO: pass this in as hyperparameter
    sigma0 = node_attr['GNN_Measurement'].sigma0

    # extrapolate the merged state from the central node to the neighbour node & storing at the neighbur node
    print("extrapolating merged state for", node_num, "to", neighbour_num)
    dx = neighbour_x - node_x
    
    # variables for F; state transition matrix
    alpha = 0.1                                 # OU parameter TODO: this value needs to be tuned
    e1 = np.exp(-np.abs(dx) * alpha)
    f1 = (1.0 - e1) / alpha
    g1 = (np.abs(dx) - f1) / alpha
    # variables for Q process noise matrix
    sigma_ou = 0.0                              # for extrapolation sigma_ou should be zero
    sw2 = sigma_ou**2                           # OU parameter 
    st2 = sigma_ms**2                           # process noise representing multiple scattering
    dx2 = dx**2
    dxw2 = dx2 * sw2
    Q02 = 0.5*dxw2
    Q01 = dx*(st2 + Q02)
    Q12 = dx*sw2

    F = np.array([[1.,    dx,     g1], 
                  [0.,    1.,     f1],
                  [0.,    0.,     e1]])      # F state transition matrix, extrapolation Jacobian - linear & OU
    extrp_state = F.dot(merged_state)
    extrp_cov = F.dot(merged_cov).dot(F.T)

    # validate the extrapolated state against the measurement at the neighbour node
    # calc chi2 distance between measurement at neighbour node and extrapolated track state
    H = np.array([[1., 0., 0.]])
    residual = neighbour_y - H.dot(extrp_state)     # compute the residual
    S = H.dot(extrp_cov).dot(H.T) + sigma0**2       # covariance of residual (denominator of kalman gain)
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
    # with open('chi2_data_sigma_0.1.csv', 'a') as f:
    #     f.write(line)

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
        f.Q = np.array([[dx2*(st2 + 0.25*dxw2), Q01,        Q02], 
                        [Q01,                   st2 + dxw2, Q12],
                        [Q02,                   Q12,        sw2]])      # Q process uncertainty/noise, OU model
        z = neighbour_y                     # "sensor reading"

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

    for i, s in enumerate(subGraphs):
        print("-------------------")
        print("SUBGRAPH " + str(i))
        for node in s.nodes(data=True):
            pprint.pprint(node)
        print("--------------------")
        print("EDGE DATA:", s.edges.data(), "\n")

    title = "Subgraphs after iteration 2: message passing, extrapolation \n& validation of merged state, formation of updated state"
    h.plot_subgraphs(subGraphs, outputDir, node_labels=True, save_plot=True, title=title)
    # save networks
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)



if __name__ == "__main__":
    main()