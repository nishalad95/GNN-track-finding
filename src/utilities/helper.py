import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
from GNN_Measurement import GNN_Measurement as gnn
import pprint


def edge_length_xy(row):
    return np.sqrt(row.x**2 + row.y**2)

def get_volume_id(layer_id):
    return int(layer_id / 1000)

def get_in_volume_layer_id(layer_id):
    return int(layer_id % 100)


def initialize_edge_activation(GraphList):
    for subGraph in GraphList: nx.set_edge_attributes(subGraph, 1, "activated")


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
                    # layer = subGraph.nodes[neighbour_num]['GNN_Measurement'].x
                    layer = subGraph.nodes[neighbour_num]['in_volume_layer_id']
                    if layer in layer_neighbour_num_dict.keys():
                        layer_neighbour_num_dict[layer].append(neighbour_num)
                    else:
                        layer_neighbour_num_dict[layer] = [neighbour_num]
        
            # assign prior probabilities to nodes in neighbourhood
            for _, neighbour_nums_list in layer_neighbour_num_dict.items():
                prior = 1/len(neighbour_nums_list)
                for neighbour_num in neighbour_nums_list:
                    track_state_estimates[neighbour_num]['prior'] = prior


def query_node_degree_in_edges(subGraph, node_num):
    in_edges = subGraph.in_edges(node_num) # one direction only, not double counted
    node_degree = 0
    for edge in in_edges:
        neighbour_num = edge[0]
        if (subGraph[neighbour_num][node_num]["activated"] == 1) : node_degree += 1
    return node_degree


def compute_mixture_weights(GraphList):
    for subGraph in GraphList:
        nodes = subGraph.nodes(data=True)
        if len(nodes) == 1: continue
        for node in nodes:
            
            node_num = node[0]
            node_degree = query_node_degree_in_edges(subGraph, node_num)
            mixture_weight = 1/node_degree

            track_state_estimates = node[1]['track_state_estimates']
            for neighbour_num, v in track_state_estimates.items():
                v['mixture_weight'] = mixture_weight

                # add as an edge attribute - useful in community detection
                subGraph[neighbour_num][node_num]['mixture_weight'] = mixture_weight

# used in extrapolate_merged_states
def calculate_side_norm_factor(subGraph, node, updated_track_states):
    
    # split track state estimates into LHS & RHS
    node_num = node[0]
    node_attr = node[1]
    node_x_layer = node_attr['GNN_Measurement'].x
    left_nodes, right_nodes = [], []
    left_coords, right_coords = [], []

    for neighbour_num, _ in updated_track_states.items():
        neighbour_x_layer = subGraph.nodes[neighbour_num]['GNN_Measurement'].x

        # only calculate for activated edges
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            if neighbour_x_layer < node_x_layer: 
                left_nodes.append(neighbour_num)
                left_coords.append(neighbour_x_layer)
            else: 
                right_nodes.append(neighbour_num)
                right_coords.append(neighbour_x_layer)
    
    # store norm factor as node attribute
    left_norm = len(list(set(left_coords)))
    for left_neighbour in left_nodes:
        updated_track_states[left_neighbour]['side'] = "left"
        updated_track_states[left_neighbour]['lr_layer_norm'] = 1
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            updated_track_states[left_neighbour]['lr_layer_norm'] = left_norm

    right_norm = len(list(set(right_coords)))
    for right_neighbour in right_nodes:
        updated_track_states[right_neighbour]['side'] = "right"
        updated_track_states[right_neighbour]['lr_layer_norm'] = 1
        if subGraph[neighbour_num][node_num]['activated'] == 1:
            updated_track_states[right_neighbour]['lr_layer_norm'] = right_norm


# used in extrapolate_merged_states
def reweight(subGraphs, track_state_estimates_key):
    print("Reweighting Gaussian mixture...")
    reweight_threshold = 0.1

    for subGraph in subGraphs:
        if len(subGraph.nodes()) == 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            if track_state_estimates_key in node_attr.keys():
                print("\nReweighting node:", node_num)
                updated_track_states = node_attr[track_state_estimates_key]
                
                calculate_side_norm_factor(subGraph, node, updated_track_states)
                
                # compute reweight denominator
                reweight_denom = 0
                for neighbour_num, updated_state_dict in updated_track_states.items():
                    if subGraph[neighbour_num][node_num]['activated'] == 1:
                        reweight_denom += (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'])
                
                # compute reweight
                for neighbour_num, updated_state_dict in updated_track_states.items():
                    if subGraph[neighbour_num][node_num]['activated'] == 1:
                        reweight = (updated_state_dict['mixture_weight'] * updated_state_dict['likelihood'] * updated_state_dict['prior']) / reweight_denom
                        reweight /= updated_state_dict['lr_layer_norm']
                        print("REWEIGHT:", reweight)
                        print("side:", updated_state_dict['side'])  
                        updated_state_dict['mixture_weight'] = reweight

                        # add as edge attribute
                        subGraph[neighbour_num][node_num]['mixture_weight'] = reweight
                    
                        # reactivate/deactivate
                        if reweight < reweight_threshold:
                            subGraph[neighbour_num][node_num]['activated'] = 0
                            print("deactivating edge: (", neighbour_num, ",", node_num, ")")
                        else:
                            subGraph[neighbour_num][node_num]['activated'] = 1
                            print("reactivating edge: (", neighbour_num, ",", node_num, ")")



def compute_track_state_estimates(GraphList, sigma0, mu):
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements
    
    for G in GraphList:
        for node in G.nodes():
            gradients = []
            state_estimates = {}
            # (x, y)
            node_gnn = G.nodes[node]["GNN_Measurement"]
            m1 = (node_gnn.x, node_gnn.y)
            # (z, r)
            # m1 = (node_gnn.z, node_gnn.r)
                        
            for neighbor in nx.all_neighbors(G, node):
                # (x, y)
                neighbour_gnn = G.nodes[neighbor]["GNN_Measurement"]
                m2 = (neighbour_gnn.x, neighbour_gnn.y)
                # (z, r)
                # m2_zr = (neighbour_gnn.z, neighbour_gnn.r)

                grad = (m1[1] - m2[1]) / (m1[0] - m2[0])
                gradients.append(grad)
                edge_state_vector = np.array([m1[1], grad])
                H = np.array([ [1, 0], [1/(m1[0] - m2[0]), 1/(m2[0] - m1[0])] ])
                covariance = H.dot(S).dot(H.T)
                covariance = np.array([covariance[0,0], covariance[0,1], covariance[1,0], covariance[1,1]])

                key = neighbor # track state probability of A (node) conditioned on its neighborhood B
                state_estimates[key] = {'edge_state_vector': edge_state_vector, 
                                        'edge_covariance': covariance, 
                                        'xy': m2,
                                        # 'zr' : m2_zr
                                        }
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList


def construct_graph(graph, nodes, edges, truth, sigma0, mu):
    # group truth particle ids to node index
    group = truth.groupby('node_idx')
    grouped_pid = group.apply(lambda row: row['particle_id'].unique())
    grouped_pid = pd.DataFrame({'node_idx':grouped_pid.index, 'particle_id':grouped_pid.values})
    grouped_pid['single_particle_id'] = grouped_pid['particle_id'].str[0]

    # add nodes
    for i in range(len(nodes)):
        row = nodes.iloc[i]
        node_idx = int(row.node_idx)
        x, y, z, r = row.x, row.y, row.z, row.r
        volume_id, in_volume_layer_id = row.volume_id, row.in_volume_layer_id
        label =  grouped_pid.loc[grouped_pid['node_idx'] == node_idx]['single_particle_id'].item()  # MC truth label (particle id)
        gm = gnn.GNN_Measurement(x, y, z, r, sigma0, mu, label=label, n=node_idx)
        graph.add_node(node_idx, GNN_Measurement=gm, 
                            xy=(x, y),                              # all attributes here for development only - can be abstracted away in GNN_Measurement
                            zr=(z, r),
                            xyzr=(x,y,z,r),                         # temporary - used in extract track candidates                             
                            volume_id = volume_id,
                            in_volume_layer_id = in_volume_layer_id,
                            vivl_id = (volume_id, in_volume_layer_id),
                            truth_particle=label)

    # add bidirectional edges
    for i in range(len(edges)):
        row = edges.iloc[i]
        node1, node2 = row.node1, row.node2
        graph.add_edge(node1, node2)
        graph.add_edge(node2, node1)


def load_metadata(event_path, max_volume_region):
    # graph nodes
    nodes = pd.read_csv(event_path + "nodes.csv")
    nodes = nodes.loc[nodes['layer_id'] <= max_volume_region]
    nodes['r'] = nodes.apply(lambda row: edge_length_xy(row), axis=1)
    nodes['volume_id'] = nodes.apply(lambda row: get_volume_id(row.layer_id), axis=1) 
    nodes['in_volume_layer_id'] = nodes.apply(lambda row: get_in_volume_layer_id(row.layer_id), axis=1)

    # graph edges
    edges = pd.read_csv(event_path + "edges.csv")
    new_header = edges.iloc[0] #grab the first row for the header
    edges = edges[1:] #take the data less the header row
    edges.columns = new_header #set the header row as the df header
    edges['node2'] = edges.apply(lambda row: row.name[0], axis=1)
    edges['node1'] = edges.apply(lambda row: row.name[1], axis=1)
    edges = edges.astype({'node2': 'int32', 'node1': 'int32'})
    edges = edges.loc[(edges['node2'] < max_volume_region) & (edges['node1'] < max_volume_region)]

    # return as dataframes
    return nodes, edges


def load_save_truth(event_path, truth_event_path, truth_event_file):
    # truth event
    hits_particles = pd.read_csv(truth_event_path + "truth.csv")
    particles_nhits = pd.read_csv(truth_event_path + "particles.csv")

    # nodes to hits
    nodes_hits = pd.read_csv(event_path + "nodes_to_hits.csv")
    truth = nodes_hits[['node_idx', 'hit_id']]
    hit_ids = truth['hit_id']
    particle_ids = np.array([])
    for hid in hit_ids:
        pid = hits_particles.loc[hits_particles['hit_id'] == hid]['particle_id'].item()
        particle_ids = np.append(particle_ids, pid)
    truth['particle_id'] = particle_ids

    # number of hits for each truth particle
    nhits = np.array([])
    particle_ids = truth['particle_id']
    for pid in particle_ids:
        try:
            n = particles_nhits.loc[particles_nhits['particle_id'] == pid]['nhits'].item()
        except ValueError:
            n = 0
        nhits = np.append(nhits, n)
    truth['nhits'] = nhits

    # save truth
    truth.to_csv(truth_event_file, index=False)


# save network as serialized form & adjacency matrix
def save_network(directory, i, subGraph):
    filename = directory + str(i) + "_subgraph.gpickle"
    nx.write_gpickle(subGraph, filename)


def __plot_subgraphs_in_plane(GraphList, outputDir, key, axis1, axis2, node_labels, save_plot, title):
    _, ax = plt.subplots(figsize=(10,8))
    for i, subGraph in enumerate(GraphList):
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, key)
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=5)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)
    plt.axis('on')
    if save_plot:
        plt.savefig(outputDir + axis1 + axis2 + "_subgraphs_trackml_mod.png", dpi=300)


def plot_subgraphs(GraphList, outputDir, node_labels=False, save_plot=False, title=""):
    # xy plane
    __plot_subgraphs_in_plane(GraphList, outputDir, 'xy', "x", "y", node_labels, save_plot, title)
    # zr plane
    __plot_subgraphs_in_plane(GraphList, outputDir, 'zr', "z", "r", node_labels, save_plot, title)



def __plot_save_subgraphs_iterations_in_plane(GraphList, extracted_pvals, outputFile, title,
                                                    key, axis1, axis2, node_labels, save_plot):

    _, ax = plt.subplots(figsize=(12,10))
    for subGraph in GraphList:
        iteration = int(subGraph.graph["iteration"])
        color = subGraph.graph["color"]
        pos=nx.get_node_attributes(subGraph, key)
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=65, label=iteration)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=8)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title(title)
    # plot legend & remove duplicate entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper left', title="iteration")    
    plt.axis('on')
    if save_plot:
        plt.savefig(outputFile + "subgraphs_"+ key +".png", dpi=300)

    # save extracted candidate track quality information
    for i, sub in enumerate(GraphList):
        save_network(outputFile, i, sub) # save network to serialized form

    f = open(outputFile + "pvals.csv", 'w')
    writer = csv.writer(f)
    for pval in extracted_pvals:
        writer.writerow([pval])
    f.close()


# used for visualising the good extracted candidates & iteration num
def plot_save_subgraphs_iterations(GraphList, extracted_pvals, outputFile, title, node_labels=True, save_plot=True):
    #xy plane
    __plot_save_subgraphs_iterations_in_plane(GraphList, extracted_pvals, outputFile, title, 'xy', "x", "y", node_labels, save_plot)
    #zr plane
    __plot_save_subgraphs_iterations_in_plane(GraphList, extracted_pvals, outputFile, title, 'zr', "z", "r", node_labels, save_plot)



# # used for visualising the good extracted candidates & iteration num
# def plot_save_subgraphs_iterations(GraphList, extracted_pvals, outputFile, title, node_labels=True, save_plot=True):
#     # xy plot
#     _, ax = plt.subplots(figsize=(12,10))
#     for subGraph in GraphList:
#         iteration = int(subGraph.graph["iteration"])
#         color = subGraph.graph["color"]
#         pos=nx.get_node_attributes(subGraph,'xy')
#         edge_colors = []
#         for u, v in subGraph.edges():
#             if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
#             else: edge_colors.append("#f2f2f2")
#         nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
#         nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=65, label=iteration)
#         if node_labels:
#             nx.draw_networkx_labels(subGraph, pos, font_size=8)
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     plt.xlabel("x coordinate")
#     plt.ylabel("y coordinate")
#     plt.title(title)
#     # plot legend & remove duplicate entries
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), loc='upper left', title="iteration")    
#     plt.axis('on')
#     if save_plot:
#         plt.savefig(outputFile + "subgraphs_xy.png", dpi=300)

#     # rz plot
#     _, ax = plt.subplots(figsize=(12,10))
#     for subGraph in GraphList:
#         iteration = int(subGraph.graph["iteration"])
#         color = subGraph.graph["color"]
#         pos=nx.get_node_attributes(subGraph,'zr')
#         edge_colors = []
#         for u, v in subGraph.edges():
#             if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
#             else: edge_colors.append("#f2f2f2")
#         nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
#         nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=65, label=iteration)
#         if node_labels:    
#             nx.draw_networkx_labels(subGraph, pos, font_size=8)
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     plt.xlabel("z")
#     plt.ylabel("r")
#     plt.title(title)
#     # plot legend & remove duplicate entries
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), loc='upper left', title="iteration")    
#     plt.axis('on')
#     if save_plot:
#         plt.savefig(outputFile + "subgraphs_rz.png", dpi=300)

#     # save extracted candidate track quality information
#     for i, sub in enumerate(GraphList):
#         save_network(outputFile, i, sub) # save network to serialized form

#     f = open(outputFile + "pvals.csv", 'w')
#     writer = csv.writer(f)
#     for pval in extracted_pvals:
#         writer.writerow([pval])
#     f.close()