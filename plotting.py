import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
import numpy as np
import random

# plot the graph network in the layers of the ID in the xy plane
def plot_save_temperature_network(G, attr, outputDir):
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
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title("Track simulation with potential hit pairs, vertex degree indicated by colour")
    plt.axis('on')
    plt.tight_layout()
    plt.savefig(outputDir + "/temperature_network.png", dpi=300)

    # save to serialized form & adjacency matrix
    nx.write_gpickle(G, outputDir + "temperature_network.gpickle")   # save in serial form
    A = nx.adjacency_matrix(G).todense()
    np.savetxt(outputDir + 'temperature_network_matrix.csv', A)  # save as adjacency matrix


# plot the subgraphs extracted using threshold on nodes
def plot_save_subgraphs(subGraphs, outputDir, title):
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
    plt.xlim([0, 11])
    plt.ylim([-25, 15])
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title(title)
    plt.axis('on')
    plt.savefig(outputDir + "/subgraphs.png", dpi=300)

    # save to serialized form & adjacency matrix
    for i, sub in enumerate(subGraphs):
        save_network(outputDir + "/subgraphs/", i, sub)


# save network as serialized form & adjacency matrix
def save_network(directory, i, subGraph):
    filename = directory + str(i) + "_subgraph.gpickle"
    nx.write_gpickle(subGraph, filename)
    A = nx.adjacency_matrix(subGraph).todense()
    np.savetxt(directory + str(i) + "_subgraph_matrix.csv", A)
