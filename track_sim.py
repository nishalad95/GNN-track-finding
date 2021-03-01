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
import pickle
import random
from GNN_Measurement import GNN_Measurement
from HitPairPredictor import HitPairPredictor


# global variables
DIR = "simulation/"

def compute_track_state_estimates(GraphList, S):
    # computes the following node and edge attributes: vertex degree, empirical mean & variance of edge orientation
    # track state vector and covariance, mean state vector and mean covariance, adds attributes to network
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
                edge_state_vector = np.asarray([m1[1], grad])
                H = np.matrix([ [1, 0], [1/(m1[0] - m2[0]), 1/(m2[0] - m1[0])] ])
                covariance = H.dot(S).dot(H.T)
                state_estimates[neighbor] = {'edge_state_vector': edge_state_vector, 'edge_covariance': covariance}
            
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates
            #compute mean state vector
            mean_state_vector = [edge['edge_state_vector'] for edge in state_estimates.values()]
            mean_covariance = [edge['edge_covariance'] for edge in state_estimates.values()]
            G.nodes[node]['mean_state_vector'] = np.mean(mean_state_vector, axis=0)
            G.nodes[node]['mean_covariance'] = np.mean(mean_covariance, axis=0)
    return GraphList


# plot the graph network in the layers of the ID in the xy plane
def plot_graph_vertex_degree(G, attr, title, save=False):
    _, ax = plt.subplots(figsize=(12,10))
    # colour map based on attribute
    groups = set(nx.get_node_attributes(G, attr).values())
    mapping = dict(zip(sorted(groups),count()))
    nodes = G.nodes()
    colors = [mapping[nodes[n][attr]] for n in nodes()]
    pos = nx.get_node_attributes(G,'coord_Measurement')
    nx.draw_networkx_edges(G, pos, alpha=0.1)
    nc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=colors, 
                                node_size=100, cmap=plt.cm.hot, ax=ax)
    nx.draw_networkx_labels(G, pos)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 12, 1)
    ax.set_xticks(major_ticks)
    plt.colorbar(nc)
    plt.xlabel("layer in x axis")
    plt.ylabel("y coord")
    plt.title(title)
    plt.axis('on')
    plt.tight_layout()
    if save : plt.savefig(DIR + 'graph_vertex_degree.png', dpi=300)
    plt.show()


def plot_subgraphs(subGraphs, title, save=False):
    _, ax = plt.subplots(figsize=(12,10))
    for s in subGraphs:
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        pos=nx.get_node_attributes(s,'coord_Measurement')
        nx.draw_networkx_edges(s, pos, alpha=0.25)
        nx.draw_networkx_nodes(s, pos, node_color=color[0], node_size=75)
        nx.draw_networkx_labels(s, pos)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 12, 1)
    ax.set_xticks(major_ticks)
    plt.xlabel("layer in x axis")
    plt.ylabel("y coord")
    plt.title(title)
    plt.axis('on')
    if save : plt.savefig(DIR + 'subgraphs.png', dpi=300)
    plt.show()


def main():

    # define variables
    sigma0 = 0.5 #r.m.s of track position measurements
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements
    Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
    Nl = len(Lc) #number of layers
    start = 0
    y0 = np.array([-3,   1, -1,   3, -2,  2.5, -1.5]) #track positions at start, initial y
    yf = np.array([12, -4,  5, -14, -6, -24,   0]) #final track positions, final y
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
            G.add_node(nNodes, GNN_Measurement=gm, coord_Measurement=(xc[l], yc[l] + nu))
            nNodes += 1
        ax1.scatter(xc, yc)

    ax1.set_title('Ground truth')
    ax1.grid()

    # generate hit pair predictions
    hpp = HitPairPredictor(start, 7.0, 3.5) #max y0 and tau value
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xticks(major_ticks)
    nPairs = 0

    # plot hit-pair predictions as edges, add edges to the graph network
    for L1 in range(Nl) :
        for L2 in range(L1+1,Nl) :
            if L2-L1 > 2 : break #use the next 2 layers only
            for m1 in mcoll[L1] :
                # gaussians = {}
                for m2 in mcoll[L2] :
                    if m1.node == m2.node : continue
                    result = hpp.predict(m1, m2)
                    if result ==  0 : continue #pair (m1,m2) is rejected
                    nPairs += 1
                    ax2.plot([m1.x, m2.x],[m1.y,m2.y],alpha = 0.5)
                    edge = (m1.node, m2.node)
                    G.add_edge(*edge)
    print('found',nPairs,'hit pairs')

    # plot the ground truth and predicted hit pairs
    ax2.set_title('Hit pairs')
    ax2.grid()
    plt.tight_layout()
    plt.savefig(DIR + 'simulated_tracks_hit_pairs.png', dpi=300)
    plt.show()

    # compute track state estimates, plot & save graph - hot & cold regions
    Graphs = compute_track_state_estimates([G], S)
    G = nx.to_directed(Graphs[0])
    # for node in G.nodes(data=True) : print(node)
    title = "Simulated tracks as Graph network with degree of nodes plotted in colour"
    plot_graph_vertex_degree(G, 'degree', title, save=True)
    print("Saving graph to serialized file")
    nx.write_gpickle(G, DIR + "graph.gpickle")   # save in serial form
    A = nx.adjacency_matrix(G).todense()
    np.savetxt(DIR + 'graph_matrix.csv', A)  # save as adjacency matrix

    # remove all nodes with edge orientation above threshold
    threshold = 0.5
    G = nx.Graph(G) # make a copy to unfreeze graph
    filteredNodes = [node for node, attr in G.nodes(data=True) if attr['edge_gradient_mean_var'][1] > threshold]
    for node in filteredNodes: G.remove_node(node)
    
    # extract subgraphs and update track state estimates
    G = nx.to_directed(G)
    subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    subGraphs = compute_track_state_estimates(subGraphs, S)
    # print("first subgraph info ...")
    # for node in subGraphs[1].nodes(data=True) : print(node)
    
    # plot and save subgraphs
    title = "Filtered Graph Edge orientation var <" + str(threshold) + ", weakly connected subgraphs"
    plot_subgraphs(subGraphs, title, save=True)
    print("Saving subgraphs to serialized form....")
    for i, sub in enumerate(subGraphs):
        filename = DIR + str(i) + "_subgraph.gpickle"
        nx.write_gpickle(sub, filename)
        A = nx.adjacency_matrix(G).todense()
        np.savetxt(DIR + str(i) + "_subgraph_matrix.csv", A)



if __name__ == "__main__":
    main()