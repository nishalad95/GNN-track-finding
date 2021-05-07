import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
import scipy as sp
from scipy.stats import norm
import scipy.stats
from numpy.linalg import inv
import random
import argparse
from GNN_Measurement import GNN_Measurement
from HitPairPredictor import HitPairPredictor
from plotting import *


# computes the following node and edge attributes: vertex degree, empirical mean & variance of edge orientation
# track state vector and covariance, mean state vector and mean covariance, adds attributes to network
def compute_track_state_estimates(GraphList, S):
    for G in GraphList:
        for node in G.nodes():
            gradients = []
            state_estimates = {}
            G.nodes[node]['degree'] = len(G.edges(node))
            m1 = (G.nodes[node]["GNN_Measurement"].x, G.nodes[node]["GNN_Measurement"].y)
                        
            for neighbor in nx.all_neighbors(G, node):
                m2 = (G.nodes[neighbor]["GNN_Measurement"].x, G.nodes[neighbor]["GNN_Measurement"].y)
                grad = (m1[1] - m2[1]) / (m1[0] - m2[0])
                gradients.append(grad)
                edge_state_vector = np.array([m1[1], grad])
                H = np.array([ [1, 0], [1/(m1[0] - m2[0]), 1/(m2[0] - m1[0])] ])
                covariance = H.dot(S).dot(H.T)
                covariance = np.array([covariance[0,0], covariance[0,1], covariance[1,0], covariance[1,1]])
                state_estimates[neighbor] = {'edge_state_vector': edge_state_vector, 'edge_covariance': covariance}
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList


def main():

    # parse command line args
    parser = argparse.ArgumentParser(description='track hit-pair simulator')
    parser.add_argument('-t', '--threshold', help='variance of edge orientation')
    parser.add_argument('-o', '--output', help='output directory of simulation')
    args = parser.parse_args()
    
    threshold = float(args.threshold)
    outputDir = args.output

    # define variables
    sigma0 = 0.5 #r.m.s of track position measurements
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements
    Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
    Nl = len(Lc) #number of layers
    start = 0
    y0 = np.array([-3,   1, -1,   3, -2,  2.5, -1.5]) #track positions at start, initial y
    yf = np.array([12, -4,  5, -14, -6, -24,   0]) #final track positions, final y

    # 100+ track sim
    #y0 = np.linspace(-3, 3, num=500)
    #yf = np.linspace(-900, 900, num=500)
    # random.shuffle(yf)

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

    # add hits to the graph network as nodes with node measurements
    for i in range(Ntr) :
        yc = y0[i] + tau0[i]*(Lc[:] - start)
        xc = Lc[:]
        for l in range(Nl) : 
            nu = sigma0*np.random.normal(0.0,1.0) #random error
            gm = GNN_Measurement(xc[l], yc[l] + nu, tau0[i], i, n=nNodes)
            mcoll[l].append(gm)
            G.add_node(nNodes, GNN_Measurement=gm, 
                               coord_Measurement=(xc[l], yc[l] + nu))
            nNodes += 1
        ax1.scatter(xc, yc)
    ax1.set_title('Ground truth')
    ax1.grid()

    # generate hit pair predictions
    tau = 3.5
    #tau = 100  # 100+ track sim
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
    plt.savefig(outputDir + 'simulated_tracks_hit_pairs.png', dpi=300)

    # compute track state estimates
    Graphs = compute_track_state_estimates([G], S)
    G = nx.to_directed(Graphs[0])

    # plot are save temperature network
    print("Saving temperature network to serialized form & adjacency matrix...")
    plot_save_temperature_network(G, 'degree', outputDir)

    # remove all nodes with edge orientation above threshold
    G = nx.Graph(G) # make a copy to unfreeze graph
    filteredNodes = [node for node, attr in G.nodes(data=True) if attr['edge_gradient_mean_var'][1] > threshold]
    # TODO
    # filteredNodes = [node for node, attr in G.nodes(data=True) if attr['multivariate_covariance'][1][1] > 0.05]
    for node in filteredNodes: G.remove_node(node)
    
    # extract subgraphs and update track state estimates
    G = nx.to_directed(G)
    subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    subGraphs = compute_track_state_estimates(subGraphs, S)
    
    # plot and save extracted subgraphs
    print("Saving subgraphs to serialized form & adjacency matrix...")
    title = "Weakly connected subgraphs extracted with variance of edge orientation <" + str(threshold)
    plot_save_subgraphs(subGraphs, outputDir, title)


if __name__ == "__main__":
    main()