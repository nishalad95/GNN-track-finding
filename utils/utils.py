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
def plot_save_subgraphs(GraphList, outputFile, title):
    _, ax = plt.subplots(figsize=(12,10))
    colors = ["#bf6f2e", "#377fcc", "#78c953", "#c06de3"]
    for i, subGraph in enumerate(GraphList):
        color = colors[i % len(colors)]
        # color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)])]
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
    plt.ylim([-25, 15])
    plt.xlabel("ID layer in x axis")
    plt.ylabel("y coordinate")
    plt.title(title)
    plt.axis('on')
    plt.savefig(outputFile + "subgraphs.png", dpi=300)

    # save to serialized form & adjacency matrix
    for i, sub in enumerate(GraphList):
        save_network(outputFile, i, sub)


# save network as serialized form & adjacency matrix
def save_network(directory, i, subGraph):
    filename = directory + str(i) + "_subgraph.gpickle"
    nx.write_gpickle(subGraph, filename)
    # A = nx.adjacency_matrix(subGraph).todense()
    # np.savetxt(directory + str(i) + "_subgraph_matrix.csv", A)


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
            if 'merged_state' in node_attr.keys() or 'updated_track_states' in node_attr.keys(): 
                if ('updated_track_states' in node_attr.keys()) and ('merged_state' not in node_attr.keys()):
                    color.append(updated_state_color)
                else:
                    color.append(merged_state_color)
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
    plt.ylim([-25, 15])
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
                key = (neighbor, node) # in_edge, track state going from neighbor to node, probability of A conditioned on its neighborhood B
                state_estimates[key] = {'edge_state_vector': edge_state_vector, 
                                            'edge_covariance': covariance, 
                                            'coord_Measurement': m2
                                            }
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList

# assign prior probabilities/weights for neighbourhood of each node
def compute_prior_probabilities(GraphList, track_state_key):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        if len(nodes) == 1: continue
        for node in nodes:
            
            node_attr = node[1]
            if track_state_key not in node_attr.keys(): continue

            track_state_estimates = node[1][track_state_key]
            
            # compute number of ACTIVE neighbour nodes in each layer for given neighbourhood
            layer_node_num_dict = {}
            for node_num, v in track_state_estimates.items():
                # print("NODE NUM", node_num)
                neighbor = node_num[0]
                central_node = node_num[1]
                if subGraph[neighbor][central_node]['activated'] == 1:
                    layer = subGraph.nodes[neighbor]['GNN_Measurement'].x
                    if layer in layer_node_num_dict.keys():
                        layer_node_num_dict[layer].append(node_num)
                    else:
                        layer_node_num_dict[layer] = [node_num]
        
            # assign prior probabilities to nodes in neighbourhood
            for _, node_nums_list in layer_node_num_dict.items():
                prior = 1/len(node_nums_list)
                for n in node_nums_list:
                    track_state_estimates[n]['prior'] = prior


def initialize_edge_activation(GraphList):
    for subGraph in GraphList:
        for e in subGraph.edges:
            attrs = {e: {"activated": 1}}
            nx.set_edge_attributes(subGraph, attrs)


def initialize_mixture_weights(GraphList):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        if len(nodes) == 1: continue
        for node in nodes:
            mixture_weight = 1/node[1]['degree']
            track_state_estimates = node[1]['track_state_estimates']
            for _, v in track_state_estimates.items():
                v['mixture_weight'] = mixture_weight


def update_mixture_weights(GraphList):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        edge_data = subGraph.edges.data()

        # mapping {edge going into node_num: number of activated edges}
        activated_incident_edges = {}
        for edges in edge_data:
            incident_node_num = edges[1]
            if incident_node_num in activated_incident_edges.keys():
                activated_incident_edges[incident_node_num] += edges[2]['activated']
            else:
                activated_incident_edges[incident_node_num] = edges[2]['activated']

        if len(nodes) == 1: continue
        for node in nodes:

            node_num = node[0]
            num_activated_edges = activated_incident_edges[node_num]
            
            if num_activated_edges == 0: mixture_weight = 0
            else: mixture_weight = 1 / num_activated_edges

            track_state_estimates = node[1]['track_state_estimates']
            for _, v in track_state_estimates.items():
                v['mixture_weight'] = mixture_weight


#TODO: activate_edge() and deactivate_edge() functions


# default weakly connected components CCA networkx implementation
def run_cca(GraphList):
    cca_subGraphs = []
    for subGraph in GraphList:
        subGraph = nx.to_directed(subGraph)
        for component in nx.weakly_connected_components(subGraph):
            cca_subGraphs.append(subGraph.subgraph(component).copy())
    print("NUMBER OF SUBGRAPHS ", len(cca_subGraphs))
    return cca_subGraphs
