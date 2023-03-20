# analyse remaining networks

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
from more_itertools import locate


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
filenames = []
inputDir = "src/output/iteration_2/remaining/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    # print("path: ", path)
    filenames.append(path)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of subgraphs:", len(subGraphs))

# plot the graphs
for i, s in enumerate(subGraphs):
    # if i % 55 == 0:
    if i <10:
        print(i, s)
        plot_subgraphs([s])
        plot_subgraphs_zr([s])

# extract the potentially good tracks
# 1) extract tracks with 2 nodes per layer in all layers - 1 pair in very close proximity
# 2) extract tracks with 1 node per layer in all layers and 2 nodes per layer in 1 layer
tracks_with_node_splitting = []
remaining_track_candidates = []
tracks_with_node_splitting_filenames = []
for i, s in enumerate(subGraphs):
    
    # only execute on non-track-fragments
    if (s.number_of_nodes() >= 4) and (s.number_of_nodes() <= 10000):
        # if i < 3:
        # plot_subgraphs([s])
        vivl_id_dict = nx.get_node_attributes(s, "vivl_id")
        node_nums = list(vivl_id_dict.keys())
        vivl_ids = list(vivl_id_dict.values())
        # get the freq distribution of the vivl_ids
        vivl_ids_freq = {x:vivl_ids.count(x) for x in vivl_ids}
        freq_count = list(vivl_ids_freq.values())

        # scenario 1)
        # check that there are exactly 2 nodes per layer in all layers
        if not any(count != 2 for count in freq_count):
            # print("Expect 2 nodes in each layer")
            # print("Here! subgraph: ",str(i))
            tracks_with_node_splitting.append(s)
            tracks_with_node_splitting_filenames.append(filenames[i])

        else:
            # print("Cannot process subgraph, leaving for further iterations")
            remaining_track_candidates.append(s)
    else:
        remaining_track_candidates.append(s)


print("Number of remaining subgraphs: ", len(remaining_track_candidates))
print("Number of candidates where track splitting could be possible: ", len(tracks_with_node_splitting))


# # plot the graphs
# for i, s in enumerate(tracks_with_node_splitting):
#     print(i, s)
#     plot_subgraphs([s])
#     plot_subgraphs_zr([s])