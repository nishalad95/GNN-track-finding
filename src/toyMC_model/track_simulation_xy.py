import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import json
from GNN_Measurement import GNN_Measurement as GNN_Measurement
from HitPairPredictor import HitPairPredictor as HitPairPredictor
from utilities import helper as h
import pprint
import pickle
import itertools
import random


def plot_network(GraphList, outputDir, ax, node_labels=True):
    for i, subGraph in enumerate(GraphList):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, 'xy')
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            edge_colors.append(color)
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('on')
    # plt.savefig('simulated_graph_network_xy_plane.png', dpi=300)
    # plt.show()



def simulate_event():

    # generate some tracks
    num_hits = 10
    radius = 10
    # angle_of_track = [1, 3, 6]      # num_tracks = len(angle_of_track) * 8
    # angle_of_track = [1, 6]
    angle_of_track = [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5]
    coords = np.array([])
    for i in angle_of_track:
        y = np.sqrt(radius**2 - i**2)    
        coords = np.append(coords, [i, y])
        coords = np.append(coords, [y, i])
        coords = np.append(coords, [-i, -y])
        coords = np.append(coords, [-y, -i])
        coords = np.append(coords, [i, -y])
        coords = np.append(coords, [-y, i])
        coords = np.append(coords, [-i, y])
        coords = np.append(coords, [y, -i])  
    dimensions = ( int(len(coords) / 2), 2)
    coords = coords.reshape(dimensions)
    start = np.zeros( (int(len(coords)), 2) )
    num_tracks = len(start)
    print("number of tracks", num_tracks)


    # create graph network
    G = nx.Graph()
    nNodes = 0
    sigma0 = 0.05        # measurement error r.m.s of track position measurement
    mu = 0.000001       # 10^-6 multiple scattering error - process noise


    y0 = []
    # plot the tracks
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    for n in range(num_tracks):
        dy = coords[n][1] - start[n][1]
        dx = coords[n][0] - start[n][0]
        gradient = dy/dx
        x = np.linspace(start[n][0], coords[n][0], num_hits)
        y = np.array([])
        # print("track number: ", n)
        for layer, i in enumerate(x):
            nu = sigma0 * np.random.normal(0.0,1.0) #random error
            y_new = (gradient * i) + nu
            y = np.append(y, y_new)
            # add nodes to graph network
            gm = GNN_Measurement(i, y_new, 0, 0, sigma0, mu, label=n, n=nNodes)
            # print("layer:", layer)
            G.add_node(nNodes, GNN_Measurement=gm,
                        xy=(i, y_new),
                        zr=(0,0),
                        xyzr=(i, y_new, 0, 0),
                        truth=n,
                        layer=layer)
            nNodes += 1
        ax1.scatter(x[1:], y[1:], s=5, label=n)
        # print("Y coordinates:\n", y[0])
        y0.append(y[0])
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid()
    # ax1.legend(loc="best",fontsize=6)
    ax1.set_title("XY Toy MC Model")


    # generate hit pair predictions
    ax2 = fig.add_subplot(1, 2, 2)
    nPairs = 0
    node_gm = nx.get_node_attributes(G, 'GNN_Measurement') # dictionary
    for node1, gm1 in node_gm.items():
        for node2, gm2 in node_gm.items():
            layer1 = G.nodes[node1]["layer"]
            layer2 = G.nodes[node2]["layer"]
            diff = np.abs(layer2 - layer1)
            if 0 < diff <= 2:
                dx = gm2.x - gm1.x
                dy = gm2.y - gm1.y
                if np.sqrt( dx**2 + dy**2 ) <= 3:
                    if node1 != node2: 
                        m = dy / dx
                        c = gm2.y - (m * gm2.x)
                        x_intercept = -1 * c / m
                        if (np.abs(c) <= 1.8) and (np.abs(x_intercept) <= 1.8):
                            nPairs += 1
                            ax2.plot([gm1.x, gm2.x],[gm1.y,gm2.y],alpha = 0.5)
                            # edge = (node1, node2)
                            G.add_edge(node1, node2, dx=dx, m=m)
    print('Found', nPairs, 'hit pairs')
    ax2.set_title('Hit pairs')
    ax2.grid()
    plt.tight_layout()
    # plt.savefig('simulated_tracks_hit_pairs_xy_plane.png', dpi=300)
    plt.show()

    # remove the collision point
    for i in range(num_tracks):
        G.remove_node(i*10)
    
    # plot graph network
    # G = nx.to_directed(G)   # freeze graph
    _, ax = plt.subplots(figsize=(10,8))
    plot_network([G], "", ax, node_labels=True)
    plt.show()

    # dx distribution 
    dx = nx.get_edge_attributes(G, 'dx').values()
    plt.hist(dx, bins=100)
    plt.show()

    # m distribution 
    m = nx.get_edge_attributes(G, 'm').values()
    plt.hist(m, bins=100)
    plt.show()


    # remove small dx nodes
    copyG = G.copy()
    for node1, node2, data in copyG.edges(data=True):
        if np.abs(data['dx']) > 0.75:
            if node1 in G: G.remove_node(node1)
            if node2 in G: G.remove_node(node2)


    # plot graph network
    G = nx.to_directed(G)   # freeze graph
    _, ax = plt.subplots(figsize=(10,8))
    plot_network([G], "", ax, node_labels=True)
    plt.show()

    # dx distribution 
    dx = nx.get_edge_attributes(G, 'dx').values()
    plt.hist(dx, bins=100)
    plt.show()

    # m distribution 
    m = nx.get_edge_attributes(G, 'm').values()
    plt.hist(m, bins=100)
    plt.show()

    plt.scatter(x, dx)
    plt.show()
    plt.scatter(x, m)
    plt.show()

    # connected component analysis
    subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    _, ax = plt.subplots(figsize=(10,8))
    for i in subGraphs:
        plot_network([i], "", ax, node_labels=True)
    plt.show()





def main():
    simulate_event()



if __name__ == "__main__":
    main()