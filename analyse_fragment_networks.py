import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import itertools
from collections import Counter


def plot_subgraphs(graph):
    _, ax = plt.subplots(figsize=(10,8))

    for i, subGraph in enumerate(graph):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, 'xy')
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis('on')
    plt.show()

def plot_subgraphs_zr(graph):
    _, ax = plt.subplots(figsize=(10,8))

    for i, subGraph in enumerate(graph):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, 'zr')
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel("z")
    plt.ylabel("r")
    plt.axis('on')
    plt.show()


# read in remaining
subGraphs = []
inputDir = "src/output/iteration_1/fragments/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of subgraphs:", len(subGraphs))

fragments = []
isolated_nodes = []
for s in subGraphs:
    if s.number_of_nodes() == 1:
        isolated_nodes.append(s)
    elif s.number_of_nodes() < 4:
        fragments.append(s)

print("number of isolated nodes: ", len(isolated_nodes))
percentage = len(isolated_nodes) * 100 / len(subGraphs)
print("percentage: ", percentage)
print("number of subgraphs with 2 or 3 nodes: ", len(fragments))
percentage = len(fragments) * 100 / len(subGraphs)
print("percentage: ", percentage)


# calculate the proportion of subgraphs that are track fragments
fragments = []
particle_ids = []
min_num_nodes = 4
for s in subGraphs:
    if s.number_of_nodes() < 3:
        fragments.append(s)

        # get the majority particle id from all nodes in the candidate to find out reconstructed particle id
        # get the hit dissociation to particle id for every node in each candidate
        # {node: {hit_id: [], particle_id: []} }, hits can be associated to more than 1 particle
        hit_dissociation = nx.get_node_attributes(s, 'hit_dissociation').values()
        gnn_particle_ids = []
        for hd in hit_dissociation:
            values = list(hd.values())
            gnn_particle_ids.append(values[1])
        gnn_particle_ids = list(itertools.chain(*gnn_particle_ids))
        freq_dist = Counter(gnn_particle_ids)
        particle_id = max(freq_dist, key=freq_dist.get)
        particle_ids.append(particle_id)

print("number of fragments:", len(fragments))
fraction = len(fragments) * 1.0 / len(subGraphs)
print("proportion of subgraphs that are track fragments:", fraction)

unique_particle_ids = list(set(particle_ids))
print("number of unique particle ids:", len(unique_particle_ids))

diff = len(fragments) - len(unique_particle_ids)
print("diff:", diff)


# # plot the graphs
plot_subgraphs(fragments)
plot_subgraphs_zr(fragments)

# for i, s in enumerate(fragments):
#     if i % 200 == 0 :
#         print(i, s)
#         plot_subgraphs([s])
#         plot_subgraphs_zr([s])
