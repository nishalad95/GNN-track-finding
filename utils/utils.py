import matplotlib.pyplot as plt
import networkx as nx
from itertools import count
import numpy as np
import random
import pprint


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
    plt.savefig(outputDir + "temperature_network.png", dpi=300)

    # save to serialized form & adjacency matrix
    nx.write_gpickle(G, outputDir + "temperature_network.gpickle")   # save in serial form
    A = nx.adjacency_matrix(G).todense()
    np.savetxt(outputDir + 'temperature_network_matrix.csv', A)  # save as adjacency matrix


# plot the subgraphs extracted using threshold on nodes
def plot_save_subgraphs(GraphList, outputFile, title, save=True):
    _, ax = plt.subplots(figsize=(12,10))
    colors = ["#bf6f2e", "#377fcc", "#78c953", "#c06de3"]
    for i, subGraph in enumerate(GraphList):
        color = colors[i % len(colors)]
        pos=nx.get_node_attributes(subGraph,'coord_Measurement')
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=75)
        nx.draw_networkx_labels(subGraph, pos)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 12, 1)
    ax.set_xticks(major_ticks)
    plt.xlim([0, 11])
    plt.ylim([-27, 15])
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title(title)
    plt.axis('on')
    plt.savefig(outputFile + "subgraphs.png", dpi=300)

    # save to serialized form & adjacency matrix
    if save:
        for i, sub in enumerate(GraphList):
            save_network(outputFile, i, sub)


# used for visualising the good extracted candidates & iteration num
def plot_save_subgraphs_iterations(GraphList, outputFile, title, save=True):
    _, ax = plt.subplots(figsize=(12,10))

    for subGraph in GraphList:
        iteration = int(subGraph.graph["iteration"])
        color = subGraph.graph["color"]
        pos=nx.get_node_attributes(subGraph,'coord_Measurement')
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=75, label=iteration)
        nx.draw_networkx_labels(subGraph, pos)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 12, 1)
    ax.set_xticks(major_ticks)
    plt.xlim([0, 11])
    plt.ylim([-27, 15])
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title(title)

    # plot legend & remove duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', title="iteration")    
    plt.axis('on')
    plt.savefig(outputFile + "subgraphs.png", dpi=300)

    # save to serialized form & adjacency matrix
    if save:
        for i, sub in enumerate(GraphList):
            save_network(outputFile, i, sub)


# save network as serialized form & adjacency matrix
def save_network(directory, i, subGraph):
    filename = directory + str(i) + "_subgraph.gpickle"
    nx.write_gpickle(subGraph, filename)


# plot the subgraphs extracted using threshold on nodes
def plot_subgraphs_merged_state(GraphList, outputFile, title):
    _, ax = plt.subplots(figsize=(12,10))
    colors = ["#bf6f2e", "#377fcc", "#78c953", "#c06de3"]
    merged_state_color = "#fce303"
    updated_state_color = "#8f8003"
    for i, subGraph in enumerate(GraphList):
        
        default_color = colors[i % len(colors)]
        color = []
        nodes = subGraph.nodes()
        for node in subGraph.nodes(data=True):
            node_attr = node[1]
            keys = node_attr.keys()

            if 'merged_state' in keys:
                color.append(merged_state_color)
            elif 'updated_track_states' in keys:
                color.append(updated_state_color)
            else:
                color.append(default_color)

        pos=nx.get_node_attributes(subGraph,'coord_Measurement')
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(default_color)
            else: edge_colors.append("#f2f2f2")

        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=75)
        nx.draw_networkx_labels(subGraph, pos)

    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    major_ticks = np.arange(0, 12, 1)
    ax.set_xticks(major_ticks)
    plt.xlim([0, 11])
    plt.ylim([-27, 15])
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title(title)
    plt.axis('on')
    plt.savefig(outputFile + "subgraphs_merged_state.png", dpi=300)



# computes the following node and edge attributes: vertex degree, empirical mean & variance of edge orientation
# track state vector and covariance, mean state vector and mean covariance, adds attributes to network
def compute_track_state_estimates(GraphList, sigma0):
    
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements
    
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
                key = neighbor # track state probability of A (node) conditioned on its neighborhood B
                state_estimates[key] = {'edge_state_vector': edge_state_vector, 
                                        'edge_covariance': covariance, 
                                        'coord_Measurement': m2
                                        }
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList


# used for initialization & update of priors
# assign prior probabilities/weights for neighbourhood of each node
def compute_prior_probabilities(GraphList, track_state_key):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        if len(nodes) == 1: continue
        
        for node in nodes:    
            node_num = node[0]
            node_attr = node[1]
            if track_state_key not in node_attr.keys(): continue

            track_state_estimates = node_attr[track_state_key]
            
            # compute number of ACTIVE neighbour nodes in each layer for given neighbourhood
            layer_neighbour_num_dict = {}
            for neighbour_num, _ in track_state_estimates.items():
                # inward edge coming into the node from the neighbour
                if subGraph[neighbour_num][node_num]['activated'] == 1:
                    layer = subGraph.nodes[neighbour_num]['GNN_Measurement'].x
                    if layer in layer_neighbour_num_dict.keys():
                        layer_neighbour_num_dict[layer].append(neighbour_num)
                    else:
                        layer_neighbour_num_dict[layer] = [neighbour_num]
        
            # assign prior probabilities to nodes in neighbourhood
            for _, neighbour_nums_list in layer_neighbour_num_dict.items():
                prior = 1/len(neighbour_nums_list)
                for neighbour_num in neighbour_nums_list:
                    track_state_estimates[neighbour_num]['prior'] = prior


def initialize_mixture_weights(GraphList):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        if len(nodes) == 1: continue
        for node in nodes:
            mixture_weight = 1/node[1]['degree']
            track_state_estimates = node[1]['track_state_estimates']
            for neighbour_num, v in track_state_estimates.items():
                v['mixture_weight'] = mixture_weight

                # add as an edge attribute - useful in community detection
                node_num = node[0]
                subGraph[neighbour_num][node_num]['mixture_weight'] = mixture_weight




def initialize_edge_activation(GraphList):
    for subGraph in GraphList:
        for e in subGraph.edges:
            attrs = {e: {"activated": 1}}
            nx.set_edge_attributes(subGraph, attrs)


#TODO: activate_edge() and deactivate_edge() functions


def query_node_degree_in_edges(subGraph, node_num):
    in_edges = subGraph.in_edges(node_num) # one direction only, not double counted
    # print("IN EDGES", in_edges)
    node_degree = 0
    for edge in in_edges:
        neighbour_num = edge[0]
        if (subGraph[neighbour_num][node_num]["activated"] == 1) : node_degree += 1
    return node_degree

def query_empirical_mean_var(subGraph, node_num):
    gradients = []
    m1 = (subGraph.nodes[node_num]["GNN_Measurement"].x, subGraph.nodes[node_num]["GNN_Measurement"].y)

    for neighbor in nx.all_neighbors(subGraph, node_num):
        if subGraph[neighbor][node_num]['activated'] == 1:
            m2 = (subGraph.nodes[neighbor]["GNN_Measurement"].x, subGraph.nodes[neighbor]["GNN_Measurement"].y)
            grad = (m1[1] - m2[1]) / (m1[0] - m2[0])
            gradients.append(grad)
    
    if len(gradients) > 0: return np.var(gradients)
    else: return None


# default weakly connected components CCA networkx implementation
def run_cca(GraphList):
    cca_subGraphs = []
    for subGraph in GraphList:
        subGraph = nx.to_directed(subGraph)
        for component in nx.weakly_connected_components(subGraph):
            cca_subGraphs.append(subGraph.subgraph(component).copy())
    print("NUMBER OF SUBGRAPHS ", len(cca_subGraphs))
    return cca_subGraphs
