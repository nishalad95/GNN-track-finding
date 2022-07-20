import networkx as nx
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random
from collections import Counter
from itertools import combinations
import itertools

# TODO: this will change when the barrel and endcap are both used
# Currently only using the endcap

def compute_3d_distance(coord1, coord2):
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )


def plot_subgraphs_in_plane(subGraph, key, axis1, axis2, i, node_labels=True):
    _, ax = plt.subplots(figsize=(10,8))
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
    pos=nx.get_node_attributes(subGraph, key)
    nodes = subGraph.nodes()
    edge_colors = []
    for u, v in subGraph.edges():
        if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
        else: edge_colors.append("#f2f2f2")
    nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
    nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=65)
    if node_labels:
        nx.draw_networkx_labels(subGraph, pos, font_size=8)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.axis('on')
    plt.show()
    # plt.savefig(str(i) + ".png", dpi=300)


# read in subgraph data
inputDir = "src/output/iteration_1/remaining/"
subgraph_path = ".gpickle"
subGraphs = []
os.chdir(".")
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)
    subGraphs.append(sub)

# TODO: in the extraction phase we need to add a check that no track can contain the same module id hit

module_id_differences = []
separations = []
for i, subGraph in enumerate(subGraphs):
    
    # don't process track fragments
    if subGraph.number_of_nodes() <= 4: continue

    # if i == 6 or i == 7:    # testing only
    print("\nProcessing subgraph ", str(i))

    # determine if a subgraph contains between 1 and 2 layers with multiple nodes
    node_in_volume_layer_id_dict = nx.get_node_attributes(subGraph, 'in_volume_layer_id')
    node_module_id_dict = nx.get_node_attributes(subGraph, 'module_id')
    counts = Counter(node_in_volume_layer_id_dict.values())
    layer_counts_dict = dict((k, v) for k, v in dict(counts).items() if int(v) == 2)
    num_layers_with_multiple_nodes = len(layer_counts_dict)

    if 1 <= num_layers_with_multiple_nodes <= 2:

        # get node indexes which are in the same layer
        for layer_id, count in layer_counts_dict.items():
            node_idx = [node for node in node_in_volume_layer_id_dict.keys() if layer_id == node_in_volume_layer_id_dict[node]]
            print("nodes that are close together:\n", node_idx)

            # only merge if there are 2 nodes in close proximity
            if len(node_idx) <= 2:

                # check that these 2 nodes have a common node in their neighbourhood
                node1 = node_idx[0]
                node2 = node_idx[1]
                node1_edges = subGraph.edges(node1)
                node2_edges = subGraph.edges(node2)
                node1_edges = list(itertools.chain.from_iterable(node1_edges))
                node2_edges = list(itertools.chain.from_iterable(node2_edges))
                node1_edges = filter(lambda val: val != node1, node1_edges)
                node2_edges = filter(lambda val: val != node2, node2_edges)
                print("node1 edges:\n", node1_edges)
                print("node2 edges:\n", node2_edges)
                common_nodes = list(set(node1_edges).intersection(node2_edges))
                print("common nodes:\n", common_nodes)

                if len(common_nodes) != 0:

                    # compute pairwise separation in 3D space
                    xyzr_coords = [subGraph.nodes[n]['xyzr'] for n in node_idx]
                    separation = [compute_3d_distance(a, b) for a, b in combinations(xyzr_coords, 2)]
                    separations.append(separation)

                    print("combinations:\n")
                    for a, b in combinations(xyzr_coords, 2):
                        print(a, b)
                        print(" ")
                        index = xyzr_coords.index(a)
                        print("index a: ", index)
                        print("node: ", node_idx[index])

                    plot_subgraphs_in_plane(subGraph, 'xy', 'x', 'y', i)

separations = list(itertools.chain(*separations))
print("number of remaining subgraphs that can be further extracted:", len(separations))

percentile95 = np.percentile(np.array(separations), 95)
print("Cut the data at 95th percentile: a total separation distance of:", percentile95)


plt.hist(separations, density=False, bins=200)
plt.xlabel("Distance between close proximity nodes in the same layer")
plt.ylabel("Frequency")
plt.show()
