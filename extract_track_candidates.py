from filterpy.kalman import *
from filterpy import common
from scipy.stats import distributions, chisquare, chi2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import argparse
from utils.utils import *
from community_detection import community_detection


def KF_track_fit(sigma0, coords):
    
    obs_x = [c[0] for c in coords]
    obs_y = [c[1] for c in coords]
    yf = obs_y[0]
    # obs_y = obs_y[1:]
    dx = coords[1][0] - coords[0][0]

    # initialize KF at outermost layer
    f = KalmanFilter(dim_x=2, dim_z=1)
    f.x = np.array([yf, 0.])                # X state vector 
    f.F = np.array([[1.,dx], [0.,1.]])      # F state transition matrix
    f.H = np.array([[1.,0.]])               # H measurement matrix
    f.P = np.array([[sigma0**2,    0.],
                    [0.,         1000.]])   # P covariance
    f.R = sigma0**2
    f.Q = 0.

    # KF predict and update
    chi2_dists = []
    saver = common.Saver(f)
    for measurement in obs_y[1:]:
        f.predict()
        f.update(measurement)
        saver.save()

        # calculate chi2 distance
        updated_state, updated_cov = f.x_post, f.P_post
        residual = measurement - f.H.dot(updated_state) 
        S = f.H.dot(updated_cov).dot(f.H.T) + f.R
        inv_S = np.linalg.inv(S)
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
    parser = argparse.ArgumentParser(description='edge outlier removal')
    parser.add_argument('-i', '--input', help='input directory of outlier removal')
    parser.add_argument('-c', '--candidates', help='output directory to save track candidates')
    parser.add_argument('-r', '--remain', help='output directory to save remaining network')
    parser.add_argument('-p', '--pval', help='chi-squared track candidate acceptance level')
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-n', '--numhits', help="minimum number of hits for good track candidate")
    args = parser.parse_args()

    inputDir = args.input
    candidatesDir = args.candidates
    remainingDir = args.remain
    track_acceptance = float(args.pval)
    sigma0 = float(args.error)
    subgraph_path = "_subgraph.gpickle"
    fragment = int(args.numhits)
    iteration_num = inputDir.split("/")[1][-1]

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
    remaining = []
    for i, subGraph in enumerate(subGraphs):
        
        print("\nProcessing subGraph: ", i)
        subCopy = subGraph.copy()
        potential_tracks = CCA(subCopy)     # remove any deactive edges & identify subgraphs
        print("No. of potential tracks: ", len(potential_tracks))

        if len(potential_tracks) == 1:
            candidate = potential_tracks[0]
            
            # check for track fragments
            if len(candidate.nodes()) <= fragment : 
                print("Too few nodes, track fragment")
                remaining.append(subGraph)
                continue

            # check for 1 hit per layer
            coords = list(nx.get_node_attributes(candidate,'coord_Measurement').values())
            coords = sorted(coords, reverse=True, key=lambda x: x[0])
            good_candidate = True
            for j in range(0, len(coords)-1):
                if (np.abs(coords[j+1][0] - coords[j][0]) != 1):
                    print("Track", i, "bad candidate, not 1 hit per layer, will process through community detection")
                    good_candidate = False
                    break
            
            if good_candidate:
                pval = KF_track_fit(sigma0, coords)
                if pval >= track_acceptance:
                    print("Good KF fit, P value:", pval, "first coord:", coords[0])
                    extracted.append(candidate)
                else:
                    print("pval too small,", pval, "leave for further processing")
                    remaining.append(subGraph)
            else:
                print("Run community detection...")
                valid_communities, vc_coords = community_detection(candidate, fragment)
                if len(valid_communities) > 0:
                    print("found communities via community detection")
                    for vc, vcc in zip(valid_communities, vc_coords):
                        pval = KF_track_fit(sigma0, vcc)
                        if pval >= track_acceptance:
                            print("Good KF fit, P value:", pval, "first coord:", vcc[0])
                            extracted.append(vc)
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
                if len(candidate.nodes()) <= fragment : 
                    print("Too few nodes, track fragment")
                    continue

                # check for 1 hit per layer
                coords = list(nx.get_node_attributes(candidate,'coord_Measurement').values())
                coords = sorted(coords, reverse=True, key=lambda x: x[0])
                good_candidate = True
                for j in range(0, len(coords)-1):
                    if (np.abs(coords[j+1][0] - coords[j][0]) != 1):
                        print("Track", i, "bad candidate, not 1 hit per layer")
                        good_candidate = False
                        break
                
                if good_candidate:
                    pval = KF_track_fit(sigma0, coords)
                    if pval >= track_acceptance:
                        print("Good KF fit, P value:", pval, "first coord:", coords[0])
                        extracted.append(candidate)
                        candidate_to_remove_from_subGraph.append(candidate)
                    else:
                        print("pval too small,", pval, "leave for further processing")
                else:
                    print("Run community detection...")   
                    valid_communities, vc_coords = community_detection(candidate, fragment)
                    if len(valid_communities) > 0:
                        print("found communities via community detection")
                        for vc, vcc in zip(valid_communities, vc_coords):
                            pval = KF_track_fit(sigma0, vcc)
                            if pval >= track_acceptance:
                                print("Good KF fit, P value:", pval, "first coord:", vcc[0])
                                extracted.append(vc)
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

    print("Number of extracted candidates during this iteration:", len(extracted))
    # plot all extracted candidates, from previous iterations
    i = 0
    path = candidatesDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        extracted.append(sub)
        i += 1
        path = candidatesDir + str(i) + subgraph_path
    print("Total number of extracted candidates:", len(extracted))
    plot_save_subgraphs_iterations(extracted, candidatesDir, "Extracted candidates")
    # plot_save_subgraphs(extracted, candidatesDir, "Extracted candidates")
    plot_save_subgraphs(remaining, remainingDir, "Remaining network")




if __name__ == "__main__":
    main()