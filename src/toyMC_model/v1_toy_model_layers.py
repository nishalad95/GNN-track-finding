import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import json
from modules.GNN_Measurement import GNN_Measurement
from modules.HitPairPredictor import HitPairPredictor
from utilities import helper as h
import pprint
import pickle
import itertools


def simulate_event(threshold, sigma0, outputDir):

    Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
    Nl = len(Lc) #number of layers
    start = 0

    # track initial & final positions
    y0 = np.array([-3,  1, -1,   3, -2,  2.5, -1.5, 3, 2, 2.5], dtype=float) # initial y
    yf = np.array([12, -4,  5, -14, -6, -24,   0, -16, -22, -10], dtype=float) # final y
    offset1 = np.random.rand(y0.size) * 5
    offset2 = np.random.rand(y0.size) * 5
    y0 += offset1
    yf += offset2

    # tau: track inclination, tau0 - gradient calculated from final & initial positions
    tau0 = (yf-y0)/(Lc[-1]-start)
    Ntr = len(y0) #number of simulated tracks

    # plotting the intial track toy model
    fig = plt.figure(figsize=(16, 9))
    ax1 = fig.add_subplot(1, 2, 1)
    major_ticks = np.arange(0, 11, 1)
    ax1.set_xticks(major_ticks)

    G = nx.Graph()
    nNodes = 0
    mcoll = [] #collections of track position measurements
    for l in range(Nl) : mcoll.append([])

    # keep track of num hits associated to each simulated track
    num_hits = {}

    # add hits to the graph network as nodes with node measurements
    for i in range(Ntr) :
        nhits = 0
        yc = y0[i] + tau0[i]*(Lc[:] - start)
        xc = Lc[:]
        node_labels = []
        for l in range(Nl) : 
            nu = sigma0*np.random.normal(0.0,1.0) #random error
            gm = GNN_Measurement(xc[l], yc[l] + nu, tau0[i], sigma0, label=i, n=nNodes)
            mcoll[l].append(gm)
            G.add_node(nNodes, GNN_Measurement=gm, 
                               xy=(xc[l], yc[l] + nu),
                               truth_particle=i)
            nNodes += 1
            nhits += 1
            node_labels.append(nNodes)
        ax1.scatter(xc, yc)
    ax1.set_title('Ground truth')
    ax1.grid()


    # generate hit pair predictions
    tau = 3.5
    hpp = HitPairPredictor(start, 7.0, tau) #max y0 and tau value
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xticks(major_ticks)
    
    nPairs = 0
    # plot hit-pair predictions as edges, add edges to the graph network
    for L1 in range(Nl) :
        for L2 in range(L1+1,Nl) :
            if L2-L1 > 2 : break #use the next 2 layers only
            for m1 in mcoll[L1] :
                for m2 in mcoll[L2] :
                    if m1.node == m2.node : continue
                    result = hpp.predict(m1, m2)
                    if result ==  0 : continue #pair (m1,m2) is rejected
                    nPairs += 1
                    ax2.plot([m1.x, m2.x],[m1.y,m2.y],alpha = 0.5)
                    edge = (m1.node, m2.node)
                    G.add_edge(*edge)
    print('Found', nPairs, 'hit pairs')

    # plot the ground truth and predicted hit pairs
    ax2.set_title('Hit pairs')
    ax2.grid()
    plt.tight_layout()
    plt.savefig(outputDir + '/simulated_tracks_hit_pairs.png', dpi=300)

    # TODO:
    # # compute track state estimates
    # Graphs = compute_track_state_estimates([G])
    # G = nx.to_directed(Graphs[0])

    # # remove all nodes with mean edge orientation above threshold
    # G = nx.Graph(G) # make a copy to unfreeze graph
    # filteredNodes = [node for node, attr in G.nodes(data=True) if attr['xy_edge_gradient_mean_var'][1] > threshold]
    # # TODO
    # for node in filteredNodes: G.remove_node(node)
    
    # # CCA: extract subgraphs
    # G = nx.to_directed(G)
    # subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

    # # compute track state estimates, priors and assign initial edge weightings
    # subGraphs = compute_track_state_estimates(subGraphs)
    # initialize_edge_activation(subGraphs)
    # compute_prior_probabilities(subGraphs)
    # compute_mixture_weights(subGraphs)

    # # for i, s in enumerate(subGraphs):
    # #     print("-------------------")
    # #     print("SUBGRAPH " + str(i))
    # #     for node in s.nodes(data=True):
    # #         pprint.pprint(node)
    # #     print("--------------------")
    # #     print("EDGE DATA:", s.edges.data(), "\n")

    # return subGraphs


def main():
    # parse command line args
    parser = argparse.ArgumentParser(description='track hit-pair simulator')
    parser.add_argument('-t', '--threshold', help='variance of edge orientation')
    parser.add_argument('-s', '--sigma', help='r.m.s of track position measurements')
    parser.add_argument('-o', '--output', help='output directory of simulation')
    args = parser.parse_args()
    
    threshold = float(args.threshold)
    sigma0 = float(args.sigma)
    outputDir = args.output

    subGraphs = simulate_event(threshold, sigma0, outputDir)

    #TODO: save subgraphs

if __name__ == "__main__":
    main()