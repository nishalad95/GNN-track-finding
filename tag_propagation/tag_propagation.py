import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
import pprint
import random
import copy

# private function
def __plot_subgraphs_in_plane(GraphList, key, axis1, axis2, node_labels, save_plot, title):
    _, ax = plt.subplots(figsize=(10,8))
    for i, subGraph in enumerate(GraphList):
        color = "#55d9a4" #["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, key)
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)
    plt.axis('on')
    if save_plot:
        plt.savefig(axis1 + axis2 + "_subgraphs_trackml_mod_without_isolated.png", dpi=300)
    # plt.show()


def plot_subgraphs(GraphList, node_labels=False, save_plot=False, title=""):
    # xy plane
    __plot_subgraphs_in_plane(GraphList, 'xy', "x", "y", node_labels, save_plot, title)
    # zr plane
    __plot_subgraphs_in_plane(GraphList, 'zr', "z", "r", node_labels, save_plot, title)



def plot_graph_by_tag(graph, iteration, frac_flipped=None):
    _, ax = plt.subplots(figsize=(10,8))
    pos = nx.get_node_attributes(graph, 'xy')
    colours = list(nx.get_node_attributes(graph, 'colour').values())
    nodes = graph.nodes()
    nx.draw_networkx_edges(graph, pos, alpha=0.75)
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_color=colours, node_size=50)
    nx.draw_networkx_labels(graph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('on')
    if frac_flipped is not None:
        plt.title("Tag propagation iteration: " + str(iteration) + " fraction of nodes flipped: " + str(frac_flipped))
    else:
        plt.title("Tag propagation iteration: " + str(iteration))
    plt.savefig("tag_propagation_" + str(iteration) + ".png", dpi=300)
    # plt.show()



# read graph data
endcap_graph = nx.read_gpickle("0_subgraph.gpickle")

# create an adjacency matrix A
A = nx.convert.to_dict_of_dicts(endcap_graph, edge_data=1)
pp = pprint.PrettyPrinter(depth=4)
print("Number of nodes: ", len(A))
print("Number of nodes: ", endcap_graph.number_of_nodes())

# TODO: parallelize this
# find isolated nodes: have no connections
# Adjacency matrix: connection k --> A[k] (key --> value)
num_isolated_nodes = 0
nodes_to_remove = []
for k in A.keys():
    if len(A[k]) == 0:
        if len(endcap_graph.in_edges(k)) == 0 and len(endcap_graph.out_edges(k)) == 0:
            num_isolated_nodes += 1
            nodes_to_remove.append(k)

# percentage of non-isolated nodes
frac_isolated_nodes = 100 - (num_isolated_nodes * 100 / len(A))
print("Number of isolated nodes: ", str(num_isolated_nodes))
print("Percentage of nodes to process: ", str(frac_isolated_nodes))

# remove isolated nodes
endcap_graph = nx.DiGraph(endcap_graph)
endcap_graph.remove_nodes_from(nodes_to_remove)
A = nx.convert.to_dict_of_dicts(endcap_graph)
print("Number of nodes after removing isolated nodes: ", endcap_graph.number_of_nodes())

# # plot the non-isolated nodes
# plot_subgraphs([endcap_graph], save_plot=True, node_labels=True)

# FORM a dictionary keys:[values] nodes:[neighbours to process which have radius larger than the node]
# Non-parallel version
A_nodes_to_process = A.copy()
for node, neighbours in A.items():
    node_radius = endcap_graph.nodes[node]['zr'][1]
    neighbours = list(neighbours.keys())
    for neighbour_node in neighbours:
        neighbour_radius = endcap_graph.nodes[neighbour_node]['zr'][1]
        # if the neighbour radius is smaller than the node radius, then remove it
        if neighbour_radius > node_radius:
            del A_nodes_to_process[node][neighbour_node]   
    # remove any nodes which have its number of connections to process length 0
    if len(A_nodes_to_process[node]) == 0:
        del A_nodes_to_process[node]

# A_nodes_to_process is always ALL the nodes to process each time
# We exclude the isolated nodes and only propagte tag information from nodes from one radial side
# This is so we take into account track direction
print("printing the new version of A:")
pp.pprint(A_nodes_to_process)


# plot the graph according to the first tag value in the list (should only be 1 tag at this point)
# associate all nodes to process with a random colour
for node in endcap_graph.nodes():
    colour = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
    endcap_graph.nodes[node]['colour'] = colour
iteration = 0
plot_graph_by_tag(endcap_graph, iteration)


# begin tag propagation
iteration = 1
threshold = 0.1
num_tags_flipped = []
frac_tags_flipped = []
frac_flipped = 1.0
total_number_of_nodes_to_process = len(A_nodes_to_process)
print("total number of nodes to process: ", str(total_number_of_nodes_to_process))
current_endcap_graph = copy.deepcopy(endcap_graph)
while frac_flipped > threshold:
    tags_flipped = 0
    for node, connections_dict in A_nodes_to_process.items():
        # get all neighbourhood tags
        all_neighbourhood_tags = []
        node_tag = endcap_graph.nodes[node]['tags'][-1]
        all_neighbourhood_tags.append(node_tag)
        # find the smallest neighbourhood tag including the node tag
        for neighbour in connections_dict.keys():
            neighbour_tag = endcap_graph.nodes[neighbour]['tags'][-1]
            all_neighbourhood_tags.append(neighbour_tag)
        smallest_tag = max(all_neighbourhood_tags)
        # add to the new graph - keep track of all previous tags
        current_endcap_graph.nodes[node]['tags'].append(smallest_tag)
        # get the correponding colour
        new_colour = current_endcap_graph.nodes[smallest_tag]['colour']
        current_endcap_graph.nodes[node]['colour'] = new_colour
        # check if the tag flipped in value
        if current_endcap_graph.nodes[node]['tags'][-2] != smallest_tag:
            tags_flipped += 1
    num_tags_flipped.append(tags_flipped)
    frac_flipped = tags_flipped / total_number_of_nodes_to_process
    frac_tags_flipped.append(frac_flipped)
    
    plot_graph_by_tag(current_endcap_graph, iteration, frac_flipped)
    iteration += 1
    endcap_graph = current_endcap_graph
    current_endcap_graph = copy.deepcopy(endcap_graph)

print("number of tags flipped per iteration:\n", num_tags_flipped)
print("fraction of tags flipped per iteration:\n", frac_tags_flipped)


# plot the fraction vs iteration number
x = np.arange(0, len(frac_tags_flipped), 1, dtype=int)
_, ax = plt.subplots(figsize=(10,8))
plt.scatter(x, frac_tags_flipped)
plt.xlabel("Iteration")
plt.ylabel("Fraction")
plt.title("Fraction of nodes whose tags flipped value")
plt.savefig("fraction_flipped.png", dpi=300)




def plot_separate_subgraph(subgraphs):
    _, ax = plt.subplots(figsize=(10,8))
    for G in subgraphs:
        color = G.graph['colour']
        pos=nx.get_node_attributes(G, 'xy')
        nodes = G.nodes()
        nx.draw_networkx_edges(G, pos)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=color, node_size=50)
        nx.draw_networkx_labels(G, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-200, 200])
    plt.ylim([-200, 200])
    plt.axis('on')
    plt.savefig("test_subgraph.png", dpi=300)
    # plt.show()


# select a few random graphs by colour to check
all_colours = nx.get_node_attributes(current_endcap_graph, 'colour')
unique_colours = list(set(all_colours.values()))
print("length of all colours: ", len(all_colours))
print("Unique colours: ", unique_colours)
print("Number of subgraphs: ", len(unique_colours))

colours = [unique_colours[0], unique_colours[1], unique_colours[2], unique_colours[10], unique_colours[20],
            unique_colours[30], unique_colours[40], unique_colours[50], unique_colours[100], unique_colours[200]]
print("colour to look for: ", colours)
separate_subgraphs = []
for colour in colours:
    nodes_same_colour = []
    G = nx.Graph(colour=colour)
    for node in current_endcap_graph.nodes(data=True):
        if node[1]['colour'] == colour:
            G.add_node(node[0], xy=node[1]['xy'])
    separate_subgraphs.append(G)


plot_separate_subgraph(separate_subgraphs)
