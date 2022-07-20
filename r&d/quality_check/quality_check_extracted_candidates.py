import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random


def plot_subgraphs(graph, key, axis1, axis2):
    _, ax = plt.subplots(figsize=(10,8))
    for i, subGraph in enumerate(graph):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, key)
        print("pos")
        print(pos)
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.axis('on')
    plt.show()


# read in all extracted candidates
subGraphs = []
filenames = []
inputDir = "src/output_test/iteration_1/candidates/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    filenames.append(path)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of subgraphs:", len(subGraphs))


print("\nRunning test 1 .....")
# check for atleast 4 hits per candidate
n=4 # minimum number of hits for good track candidate acceptance (>=n)
bad_subgraphs = []
for i, s in enumerate(subGraphs):
    num_nodes = s.number_of_nodes()
    if num_nodes < n:
        print("ERROR: Less than ", str(n), " number of hits in extracted candidate!")
        print("subgraph:", str(i) + subgraph_path)
        print("filename: ", filenames[i])
        bad_subgraphs.append(s)

if len(bad_subgraphs) > 0:
    print("OH NO! Track fragments detected! - test1")
    plot_subgraphs(bad_subgraphs)


print("\nRunning test 2 .....")
# check the ordering AND connection of nodes in xy coords
# increasing in r
for i, s in enumerate(subGraphs):
    xyzr_coords = nx.get_node_attributes(s, 'xyzr')
    # for debugging
    # if i == 0:
    edges = s.edges(data=True)
    list_xy_coords_nodes = list(xyzr_coords.items())
    list_xy_coords_nodes = sorted(list_xy_coords_nodes, reverse=True, key=lambda item: item[1][3]) # ( (node_num, (x,y,z,r)) )
    
    for j in range(len(list_xy_coords_nodes) - 1):
        node1 = list_xy_coords_nodes[j][0]
        node2 = list_xy_coords_nodes[j+1][0]
        if not s.has_edge(node1, node2) and not s.has_edge(node2, node1):
            print("theres a problem! - don't accept the track - test2: ", filenames[i])


print("\nRunning test 3 .....")
# check the ordering AND connection of nodes in rz coords
# increasing in z
for i, s in enumerate(subGraphs):
    xyzr_coords = nx.get_node_attributes(s, 'xyzr')
    # for debugging
    # if i == 0:
    edges = s.edges(data=True)
    list_xy_coords_nodes = list(xyzr_coords.items())
    list_xy_coords_nodes = sorted(list_xy_coords_nodes, reverse=True, key=lambda item: item[1][2])
    
    for j in range(len(list_xy_coords_nodes) - 1):
        node1 = list_xy_coords_nodes[j][0]
        node2 = list_xy_coords_nodes[j+1][0]
        if not s.has_edge(node1, node2) and not s.has_edge(node2, node1):
            print("theres a problem! - don't accept the track - test3: ", filenames[i])


tracks_with_node_merging = []

print("\nRunning test 4 .....")
layer_increment = 2.0
# check 1 hit per layer - increasing (or decreasing) in z
for i, s in enumerate(subGraphs):
    # for debugging
    # if i == 0:
    vivl_ids = nx.get_node_attributes(s, 'vivl_id')

    # sort according to layer id
    vivl_ids = list(vivl_ids.items())
    sorted_vivl_ids = sorted(vivl_ids, reverse=False, key=lambda item: item[1])

    # check that each layer id increases by 2 each time --> i.e. no holes
    layer_ids = [item[1][1] for item in sorted_vivl_ids]
    for j in range(len(layer_ids) - 1):
        if layer_ids[j+1] - layer_ids[j] > layer_increment:
            print("holes in the track!")

    # check that each node is connected in this order
    for j in range(len(sorted_vivl_ids) - 1):
        node1 = sorted_vivl_ids[j][0]
        node2 = sorted_vivl_ids[j+1][0]
        if not s.has_edge(node1, node2) and not s.has_edge(node2, node1):
            print("theres a problem! - don't accept the track - test4: ", filenames[i])
            tracks_with_node_merging.append(s)



plot_subgraphs(tracks_with_node_merging, "xy", "x", "y")
plot_subgraphs(tracks_with_node_merging, "zr", "z", "r")