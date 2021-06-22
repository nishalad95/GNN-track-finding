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
    plt.savefig(outputDir + "temperature_network.png", dpi=300)

    # save to serialized form & adjacency matrix
    nx.write_gpickle(G, outputDir + "temperature_network.gpickle")   # save in serial form
    A = nx.adjacency_matrix(G).todense()
    np.savetxt(outputDir + 'temperature_network_matrix.csv', A)  # save as adjacency matrix


# plot the subgraphs extracted using threshold on nodes
def plot_save_subgraphs(GraphList, outputFile, title):
    _, ax = plt.subplots(figsize=(12,10))
    for subGraph in GraphList:
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])]
        pos=nx.get_node_attributes(subGraph,'coord_Measurement')
        # edge_colors = [subGraph[u][v]['color'] for u,v in subGraph.edges()]
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1:
                edge_colors.append(color[0])
            else:
                edge_colors.append("#000000")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.25)
        nx.draw_networkx_nodes(subGraph, pos, node_color=color[0], node_size=75)
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
    A = nx.adjacency_matrix(subGraph).todense()
    np.savetxt(directory + str(i) + "_subgraph_matrix.csv", A)


# computes the following node and edge attributes: vertex degree, empirical mean & variance of edge orientation
# track state vector and covariance, mean state vector and mean covariance, adds attributes to network
def compute_track_state_estimates(GraphList):
    
    sigma0 = 0.5 #r.m.s of track position measurements
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
                                            'coord_Measurement': m2}
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList

# assign prior probabilities/weights for neighbourhood of each node
def compute_prior_probabilities(GraphList):
    for s in GraphList:
        nodes = s.nodes(data=True)
        if len(nodes) == 1: continue
        for node in nodes:
            track_state_estimates = node[1]['track_state_estimates']
            
            # compute number of neighbour nodes in each layer for given neighbourhood
            layer_node_num_dict = {}
            for node_num, v in track_state_estimates.items():
                layer = v['coord_Measurement'][0]
                if layer in layer_node_num_dict.keys():
                    layer_node_num_dict[layer].append(node_num)
                else:
                    layer_node_num_dict[layer] = [node_num]
        
            # assign prior probabilities to nodes in neighbourhood
            for _, node_nums_list in layer_node_num_dict.items():
                prior = 1/len(node_nums_list)
                for n in node_nums_list:
                    track_state_estimates[n]['prior'] = prior
    
    return GraphList

def initialize_edge_activation(GraphList):
    for subGraph in GraphList:
        for e in subGraph.edges:
            attrs = {e: {"activated": 1, "color": 'g'}}
            nx.set_edge_attributes(subGraph, attrs)
    return GraphList

#TODO: activate_edge() and deactivate_edge() functions

def run_cca(GraphList):
    cca_subGraphs = []
    for subGraph in GraphList:
        subGraph = nx.to_directed(subGraph)
        for component in nx.weakly_connected_components(subGraph):
            cca_subGraphs.append(subGraph.subgraph(component).copy())
    return cca_subGraphs

# Identify subgraphs by rerunning CCA & updating track state estimates
# plot the subgraphs to view the difference after clustering
# def run_cca(subGraphs, outputDir):
 
#     sg_outliers_removed = []
#     for s in subGraphs:
#         s = nx.to_directed(s)
#         for component in nx.weakly_connected_components(s):
#             sg_outliers_removed.append(s.subgraph(component).copy())
#     sg_outliers_removed = compute_track_state_estimates(sg_outliers_removed)
#     sg_outliers_removed = compute_prior_probabilities(sg_outliers_removed)

#     title = "Filtered Graph outlier edge removal using clustering with KL distance measure"
#     plot_save_subgraphs(sg_outliers_removed, outputDir, title)