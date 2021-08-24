from filterpy.kalman import *
from filterpy import common
from scipy.stats import distributions
from scipy.stats import chi2 as chi_2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import argparse
from utils.utils import *


def KF_track_fit(sigma0, coords):
    obs_x = [c[0] for c in coords]
    obs_y = [c[1] for c in coords]
    yf = obs_y[0]
    dx = coords[1][0] - coords[0][0]
    # dx = obs_x[1] - obs_x[0]

    print("y measurements:\n", obs_y)
    print("dx:", dx)

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
    for measurement in obs_y:
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
    print("chi2 distances:", chi2_dists)
    print("total chi2 dist", total_chi2)
    print("P value: ", pval)

    # plot the smoothed tracks
    x_state = np.array(saver['x'])
    y_a = x_state[:, 0] # y_a = y_b + t_b(x_a - x_b)
    plt.scatter(obs_x, y_a, alpha=0.5, label="KF")
    plt.scatter(obs_x, obs_y, alpha=0.5, label="Measurement")
    plt.legend()
    plt.show()

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
    parser.add_argument('-cs', '--chisq', help='chi-squared track candidate acceptance level')
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    args = parser.parse_args()

    inputDir = args.input
    candidatesDir = args.candidates
    remainingDir = args.remain
    track_acceptance = float(args.chisq)
    sigma0 = float(args.error)
    subgraph_path = "_subgraph.gpickle"
    fragment = 4

    # read in subgraph data
    subGraphs = []
    i = 0
    path = inputDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        subGraphs.append(sub)
        i += 1
        path = inputDir + str(i) + subgraph_path

    print("no. of subgraphs:", len(subGraphs))

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
            print("coord:", coords[0])
            good_candidate = True
            for j in range(0, len(coords)-1):
                if (np.abs(coords[j+1][0] - coords[j][0]) != 1):
                    print("Track", i, "bad candidate, not 1 hit per layer")
                    remaining.append(subGraph)
                    good_candidate = False
                    break
            
            if good_candidate:
                pval = KF_track_fit(sigma0, coords)
                if pval >= track_acceptance:
                    print("Good KF fit, P value:", pval)
                    extracted.append(candidate)
                else:
                    print("pval too small")
                    # TODO
            else:
                #TODO: run community detection
                print("run community detection...")

        else:

            candidate_to_remove_from_subGraph = []
            for n, candidate in enumerate(potential_tracks):
                print("processing sub:", n)

                # check for track fragments
                if len(candidate.nodes()) <= fragment : 
                    print("Too few nodes, track fragment")
                    continue

                # check for 1 hit per layer
                coords = list(nx.get_node_attributes(candidate,'coord_Measurement').values())
                coords = sorted(coords, reverse=True, key=lambda x: x[0])
                print("Last coord: ", coords[0])
                good_candidate = True
                for j in range(0, len(coords)-1):
                    if (np.abs(coords[j+1][0] - coords[j][0]) != 1):
                        print("Track", i, "bad candidate, not 1 hit per layer")
                        good_candidate = False
                        break
                
                if good_candidate:
                    pval = KF_track_fit(sigma0, coords)
                    if pval >= track_acceptance:
                        print("Good KF fit, P value:", pval)
                        extracted.append(candidate)
                        candidate_to_remove_from_subGraph.append(candidate)
                    else:
                        print("pval too small")
                        # TODO
                else:
                    #TODO: run community detection
                    print("run community detection...")
            
            # remove good candidates & save remaining network
            for good_candidate in candidate_to_remove_from_subGraph:
                nodes = good_candidate.nodes()
                subGraph.remove_nodes_from(nodes)
            remaining.append(subGraph)

                    

    # plot all extracted candidates, from previous iterations
    i = 0
    path = candidatesDir + str(i) + subgraph_path
    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        extracted.append(sub)
        i += 1
        path = candidatesDir + str(i) + subgraph_path
    plot_save_subgraphs(extracted, candidatesDir, "Extracted candidates")
    plot_save_subgraphs(remaining, remainingDir, "Remaining network")




if __name__ == "__main__":
    main()