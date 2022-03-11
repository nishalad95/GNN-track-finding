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

COMMUNITY_DETECTION = False

def KF_track_fit(sigma0, sigma_ms, coords):
    
    print("KF fit coords: \n", coords)
    obs_x = [c[0] for c in coords]
    obs_y = [c[1] for c in coords]
    yf = obs_y[0]
    dx = coords[1][0] - coords[0][0]

    # 2D KF: initialize at outermost layer
    # f = KalmanFilter(dim_x=2, dim_z=1)
    # f.x = np.array([yf, 0.])                  # X state vector
    # f.F = np.array([[1.,dx], [0.,1.]])        # F state transition matrix
    # f.H = np.array([[1.,0.]])                 # H measurement matrix
    # f.P = np.array([[sigma0**2,    0.],
    #                 [0.,         1000.]])     # P covariance
    # f.R = sigma0**2                           # R measurement noise
    # f.Q = sigma_ms**2                         # Q process uncertainty/noise, due to multiple scattering

    # variables for F; state transition matrix
    alpha = 0.1                                                     # OU parameter TODO: this value needs to be tuned
    e1 = np.exp(-np.abs(dx) * alpha)
    f1 = (1.0 - e1) / alpha
    g1 = (np.abs(dx) - f1) / alpha
    # variables for Q process noise matrix
    sigma_ou = 0.0001                                               # 10^-4
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

        print("DEBUGGING:\n updated_state:\n", updated_state, "\nupdated_cov:\n", updated_cov)

        # chi2 distance
        chi2_dist = residual.T.dot(inv_S).dot(residual)
        chi2_dists.append(chi2_dist)
    
    # chi2 probability distribution
    total_chi2 = sum(chi2_dists)                    # chi squared statistic
    dof = len(obs_y) - 2                            # (no. of measurements * 1D) - no. of track params
    pval = distributions.chi2.sf(total_chi2, dof)
    print("P value: ", pval)

    # save pvalue to file
    # print("SAVING P-VAL")
    # file_object = open('pvals.txt', 'a')
    # file_object.write(str(pval) + "\n")
    # file_object.close()

    # plot the smoothed tracks
    # x_state = np.array(saver['x'])
    # y_a = x_state[:, 0] # y_a = y_b + t_b(x_a - x_b)
    # plt.scatter(obs_x[1:], y_a, alpha=0.5, label="KF")
    # plt.scatter(obs_x, obs_y, alpha=0.5, label="Measurement")
    # plt.legend()
    # plt.show()

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
        
        print("\nProcessing subGraph: ", i)
        subCopy = subGraph.copy()
        potential_tracks = CCA(subCopy)     # remove any deactive edges & identify subgraphs
        print("No. of potential tracks: ", len(potential_tracks))

        if len(potential_tracks) == 1:
            candidate = potential_tracks[0]
            
            # check for track fragments
            if len(candidate.nodes()) < fragment : 
                print("Too few nodes, track fragment")
                remaining.append(subGraph)
                continue

            # check for > 1 hit per layer - use volume_id & in_volume_layer_id
            good_candidate = True
            vivl_id_values = nx.get_node_attributes(candidate,'vivl_id').values()

            #debugging
            # print("DEBUG: vivl_id_values")
            # print("single subgraph")
            # print(vivl_id_values)

            if len(vivl_id_values) == len(set(vivl_id_values)): 
                print("no duplicates volume_ids & in_volume_layer_ids for this candidate")
            else: 
                print("Bad candidate, > 1 hit per layer, will process through community detection")
                good_candidate = False
                
                #debugging
                # print("-----------------------------")
                # print("DEBUGGING SUBGRAPH")
                # print("EDGE DATA:")
                # for connection in subGraph.edges.data():
                #     print(connection)
                # print("-------------------")
                # for n in subGraph.nodes(data=True):
                #     pprint.pprint(n)
                # print("-------------------------------")
                # prefix = str(i) + "_"
                # h.plot_subgraphs([subGraph], prefix, node_labels=True, save_plot=True, title="Extracted candidates")
            

            #TODO: need to check for holes?

            # At this stage we know candidate has 1 hit per layer
            # coords = list(nx.get_node_attributes(candidate,'xy').values())
            coords = list(nx.get_node_attributes(candidate, 'xyzr').values())
            # sort according to decreasing radius value
            coords = sorted(coords, reverse=True, key=lambda xyzr: xyzr[3])
    

            if good_candidate:
                pval = KF_track_fit(sigma0, sigma_ms, coords)
                if pval >= track_acceptance:
                    print("Good KF fit, P value:", pval, "(x,y,z,r):", coords)
                    extracted.append(candidate)
                    extracted_pvals.append(pval)
                else:
                    print("pval too small,", pval, "leave for further processing")
                    remaining.append(subGraph)
            else:
                if COMMUNITY_DETECTION:
                    print("Run community detection...")
                    #TODO: coordinates & community detection method needs to be updated
                    valid_communities, vc_coords = community_detection(candidate, fragment)
                    if len(valid_communities) > 0:
                        print("found communities via community detection")
                        for vc, vcc in zip(valid_communities, vc_coords):
                            pval = KF_track_fit(sigma0, sigma_ms, vcc)
                            if pval >= track_acceptance:
                                print("Good KF fit, P value:", pval, "(x,y,z,r):", vcc)
                                extracted.append(vc)
                                extracted_pvals.append(pval)
                                good_nodes = vc.nodes()
                                subGraph.remove_nodes_from(good_nodes)
                            else:
                                print("pval too small,", pval, "leave for further processing")
                remaining.append(subGraph)

        else:

            candidate_to_remove_from_subGraph = []
            for n, candidate in enumerate(potential_tracks):
                print("Processing sub:", n)

                # check for track fragments
                if len(candidate.nodes()) < fragment : 
                    print("Too few nodes, track fragment")
                    continue
                

                # check for > 1 hit per layer - use volume_id & in_volume_layer_id
                good_candidate = True
                vivl_id_values = nx.get_node_attributes(candidate,'vivl_id').values()
                
                # print("DEBUG: vivl_id_values")
                # print("multiple subgraphs")
                # print(vivl_id_values)
                
                if len(vivl_id_values) == len(set(vivl_id_values)): 
                    print("no duplicates volume_ids & in_volume_layer_ids for this candidate")
                else: 
                    print("Bad candidate, > 1 hit per layer, will process through community detection")
                    good_candidate = False

                    #debugging
                    # print("-----------------------------")
                    # print("DEBUGGING SUBGRAPH")
                    # print("EDGE DATA:")
                    # for connection in subGraph.edges.data():
                    #     print(connection)
                    # print("-------------------")
                    # for n in subGraph.nodes(data=True):
                    #     pprint.pprint(n)
                    # print("-------------------------------")
                    # prefix = str(i) + "_multiple_"
                    # h.plot_subgraphs([subGraph], prefix, node_labels=True, save_plot=True, title="Extracted candidates")

                #TODO: need to check for holes?

                # At this stage we know candidate has 1 hit per layer
                # coords = list(nx.get_node_attributes(candidate,'xy').values())
                coords = list(nx.get_node_attributes(candidate, 'xyzr').values())
                # sort according to decreasing radius value
                coords = sorted(coords, reverse=True, key=lambda xyzr: xyzr[3])

                
                if good_candidate:
                    pval = KF_track_fit(sigma0, sigma_ms, coords)
                    if pval >= track_acceptance:
                        print("Good KF fit, P value:", pval, "(x,y,z,r):", coords)
                        extracted.append(candidate)
                        extracted_pvals.append(pval)
                        candidate_to_remove_from_subGraph.append(candidate)
                    else:
                        print("pval too small,", pval, "leave for further processing")
                else:
                    if COMMUNITY_DETECTION:
                        print("Run community detection...")
                        #TODO: coordinates & community detection method needs to be updated
                        valid_communities, vc_coords = community_detection(candidate, fragment)
                        if len(valid_communities) > 0:
                            print("found communities via community detection")
                            for vc, vcc in zip(valid_communities, vc_coords):
                                pval = KF_track_fit(sigma0, sigma_ms, vcc)
                                if pval >= track_acceptance:
                                    print("Good KF fit, P value:", pval, "(x,y,z,r):", vcc)
                                    extracted.append(vc)
                                    extracted_pvals.append(pval)
                                    candidate_to_remove_from_subGraph.append(vc)
                                else:
                                    print("pval too small,", pval, "leave for further processing")

            
            # remove good candidates & save remaining network
            for good_candidate in candidate_to_remove_from_subGraph:
                nodes = good_candidate.nodes()
                subGraph.remove_nodes_from(nodes)
            remaining.append(subGraph)

    
    # attach iteration number & color to good extracted tracks
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])]
    for subGraph in extracted:
        subGraph.graph["iteration"] = iteration_num
        subGraph.graph["color"] = color[0]

    print("\nNumber of extracted candidates during this iteration:", len(extracted))
    print("Number of remaining subGraphs to be further processed:", len(remaining))
    # plot all extracted candidates, from previous iterations
    # TODO: this can be replaced with the helper function "save_network"
    i = 0
    path = candidatesDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        extracted.append(sub)
        i += 1
        path = candidatesDir + str(i) + subgraph_path

    print("Total number of extracted candidates:", len(extracted))

    # plot_save_subgraphs_iterations(extracted, extracted_pvals, candidatesDir, "Extracted candidates")
    h.plot_save_subgraphs_iterations(extracted, extracted_pvals, candidatesDir, "Extracted candidates", node_labels=True, save_plot=True)
    h.plot_subgraphs(remaining, remainingDir, node_labels=True, save_plot=True, title="Extracted candidates")

    # save remaining subgraphs to be further processed
    for i, sub in enumerate(remaining):
        h.save_network(remainingDir, i, sub)


    # plot the distribution of edge weightings within the extracted candidates




if __name__ == "__main__":
    main()