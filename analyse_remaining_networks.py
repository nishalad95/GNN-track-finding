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
        # print("pos")
        # print(pos)
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
inputDir = "src/output/iteration_1/remaining/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of subgraphs:", len(subGraphs))

# # plot the graphs
# for i, s in enumerate(subGraphs):
#     # if i % 55 == 0:
#     if i <10:
#         print(i, s)
#         plot_subgraphs([s])
#         plot_subgraphs_zr([s])

# # calculate the proportion of remaining subgraphs that are track fragments
# fragments = []
# min_num_nodes = 4
# for s in subGraphs:
#     if s.number_of_nodes() < 4:
#         fragments.append(s)
# print("number of fragments:", len(fragments))
# fraction = len(fragments) * 1.0 / len(subGraphs)
# print("proportion of subgraphs that are track fragments:", fraction)


# extract the potentially good tracks
# 1) extract tracks with 2 nodes per layer in all layers - 1 pair in very close proximity
# 2) extract tracks with 1 node per layer in all layers and 2 nodes per layer in 1 layer
tracks_with_node_splitting = []
tracks_with_node_merging = []
remaining_track_candidates= []
for i, s in enumerate(subGraphs):
    
    # only execute on non-track-fragments
    if s.number_of_nodes() >= 4:
        # if i < 3:
        # plot_subgraphs([s])
        vivl_id_dict = nx.get_node_attributes(s, "vivl_id")
        module_id_dict = nx.get_node_attributes(s, "module_id")
        node_nums = list(vivl_id_dict.keys())
        vivl_ids = list(vivl_id_dict.values())
        # get the freq distribution of the vivl_ids
        vivl_ids_freq = {x:vivl_ids.count(x) for x in vivl_ids}
        freq_count = list(vivl_ids_freq.values())

        # print("\nAnalysing subgraph: ", str(i))
        # print("vivl_id_dict:\n", vivl_id_dict)
        # print("module_id_dict:\n", module_id_dict)
        # print("node_nums:\n", node_nums)
        # print("vivl_ids:\n", vivl_ids)
        # print("vivl_ids_freq:\n", vivl_ids_freq)
        # print("freq_count:\n", freq_count)

        # scenario 1)
        # check that there are exactly 2 nodes per layer in all layers
        if not any(count != 2 for count in freq_count):
            # print("Expect 2 nodes in each layer")
            # print("Here! subgraph: ",str(i))
            tracks_with_node_splitting.append(s)

        # scenario 2)
        # check there is 1 node per layer in all layers, apart from 1 layer with 2 nodes
        elif 2 in freq_count:
            # check there was only 1 occurence of '2'
            freq_count_remove_2 = list(filter(lambda x: x!= 2, freq_count))
            if len(freq_count) - len(freq_count_remove_2) <= 2:
                # print("There exists only 2 or fewer occurences of '2'")
                # check that all other values are equal to 1
                if not any(count != 1 for count in freq_count_remove_2):
                    # print("Expect 1 node in each layer")
                    # print("Here! subgraph: ",str(i))
                    tracks_with_node_merging.append(s)
                else:
                    # print("More than 1 node per layer, cannot process subgraph")
                    remaining_track_candidates.append(s)
            else:
                # print("More than 1 layer with 2 nodes, cannot process subgraph")
                remaining_track_candidates.append(s)
        else:
            # print("Cannot process subgraph, leaving for further iterations")
            remaining_track_candidates.append(s)
    else:
        remaining_track_candidates.append(s)


print("Number of remaining subgraphs: ", len(remaining_track_candidates))
print("Number of candidates where node merging could be possible: ", len(tracks_with_node_merging))
print("Number of candidates where track splitting could be possible: ", len(tracks_with_node_splitting))

# print("plotting tracks where node merging possible:")
# for i, s in enumerate(tracks_with_node_merging):
#     # if i % 5 == 0:
#     plot_subgraphs([s])

# print("plotting tracks where node splitting possible:")
# for s in tracks_with_node_splitting:
#     plot_subgraphs([s])

# print("plotting remaining candidate:")
# for i, s in enumerate(remaining_track_candidates):
#     if i % 5 == 0:
#         plot_subgraphs([s])


# handle tracks_with_node_merging - scenario 2
# plot the distribution of module ids and the coordinates for close proximity nodes
distances_close_proximity_nodes = []
coords_to_plot = []
for i, s in enumerate(tracks_with_node_merging):
    # if i < 3:
    vivl_id_dict = nx.get_node_attributes(s, "vivl_id")
    module_id_dict = nx.get_node_attributes(s, "module_id")
    node_nums = list(vivl_id_dict.keys())
    vivl_ids = list(vivl_id_dict.values())
    module_ids = list(module_id_dict.values())

    # Get duplicate vivl_ids from list using list comprehension + set() + count()
    duplicated_vivl_ids = list(set([tup for tup in vivl_ids if vivl_ids.count(tup) > 1]))
    
    # there could be more than 1 duplicated element
    for dup in duplicated_vivl_ids:
        # get the indexes which they appear at, and hence get the node numbers and module_ids
        indexes_of_repeated_items = list(locate(vivl_ids, lambda x: x == dup))
        nodes_of_interest = [node_nums[idx] for idx in indexes_of_repeated_items]
        moduleids_of_interest = [module_ids[idx] for idx in indexes_of_repeated_items]
        # print("nodes of interest:\n", nodes_of_interest)
        # print("moduleids_of_interest:\n", moduleids_of_interest)

        # check if only 2 nodes are presented for each duplicated item
        if len(nodes_of_interest) == 2:
            # compute the distance between the nodes
            node1 = nodes_of_interest[0]
            node2 = nodes_of_interest[1]
            node1_coords = s.nodes[node1]['xyzr']
            node2_coords = s.nodes[node2]['xyzr']
            distance = np.sqrt( (node1_coords[0] - node2_coords[0])**2 +
                                (node1_coords[1] - node2_coords[1])**2 +
                                (node1_coords[2] - node2_coords[2])**2 )
            # print("distance: ", distance)
            distances_close_proximity_nodes.append(distance)
            coords_to_plot.append(s.nodes[node1]['zr'])
            coords_to_plot.append(s.nodes[node2]['zr'])

plt.hist(distances_close_proximity_nodes, density=False, bins=50)
plt.xlabel("3d distance of close proximity nodes")
plt.ylabel("Frequency")
plt.show()

print(coords_to_plot)
zip(*coords_to_plot)

plt.scatter(*zip(*coords_to_plot))
plt.show()





# check the extracted subgraphs and see how many have 2 nodes in 2 or fewer layers and 1 node in the rest of the layers
# read in remaining
subGraphs = []
inputDir = "src/output/iteration_1/candidates/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of extracted candidates subgraphs:", len(subGraphs))

tracks_with_node_merging = []
for i, s in enumerate(subGraphs):
    vivl_id_dict = nx.get_node_attributes(s, "vivl_id")
    module_id_dict = nx.get_node_attributes(s, "module_id")
    node_nums = list(vivl_id_dict.keys())
    vivl_ids = list(vivl_id_dict.values())
    # get the freq distribution of the vivl_ids
    vivl_ids_freq = {x:vivl_ids.count(x) for x in vivl_ids}
    freq_count = list(vivl_ids_freq.values())
    

    if 2 in freq_count:
        print("2 frequency exists")
        # check there was only 1 occurence of '2'
        freq_count_remove_2 = list(filter(lambda x: x!= 2, freq_count))
        if len(freq_count) - len(freq_count_remove_2) <= 2:
            # print("There exists only 2 or fewer occurences of '2'")
            # check that all other values are equal to 1
            if not any(count != 1 for count in freq_count_remove_2):
                # print("Expect 1 node in each layer")
                # print("Here! subgraph: ",str(i))
                tracks_with_node_merging.append(s)


print("total number of tracks extracted which had node merging: ", len(tracks_with_node_merging))
print("plotting tracks where node merging possible:")
for i, s in enumerate(tracks_with_node_merging):
    if i % 5 == 0:
        plot_subgraphs([s])