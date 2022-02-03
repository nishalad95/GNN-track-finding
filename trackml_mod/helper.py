import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
from modules.GNN_Measurement import *
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
                    # layer = subGraph.nodes[neighbour_num]['GNN_Measurement_3D'].x
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


def compute_track_state_estimates(GraphList, sigma0, mu):
    S = np.matrix([[sigma0**2, 0], [0, sigma0**2]]) # covariance matrix of measurements
    
    for G in GraphList:
        for node in G.nodes():
            gradients = []
            state_estimates = {}
            # m1 = (G.nodes[node]["GNN_Measurement_3D"].x, G.nodes[node]["GNN_Measurement_3D"].y)
            # (z, r)
            m1 = (G.nodes[node]['r_z_coords'][0], G.nodes[node]['r_z_coords'][1])
                        
            for neighbor in nx.all_neighbors(G, node):
                m2_xy = (G.nodes[neighbor]["GNN_Measurement_3D"].x, G.nodes[neighbor]["GNN_Measurement_3D"].y)
                # (z, r)
                m2 = (G.nodes[neighbor]["r_z_coords"][0], G.nodes[neighbor]["r_z_coords"][1])
                grad = (m1[1] - m2[1]) / (m1[0] - m2[0])
                gradients.append(grad)
                edge_state_vector = np.array([m1[1], grad])
                H = np.array([ [1, 0], [1/(m1[0] - m2[0]), 1/(m2[0] - m1[0])] ])
                covariance = H.dot(S).dot(H.T)
                covariance = np.array([covariance[0,0], covariance[0,1], covariance[1,0], covariance[1,1]])
                covariance += mu # add process noise
                key = neighbor # track state probability of A (node) conditioned on its neighborhood B
                state_estimates[key] = {'edge_state_vector': edge_state_vector, 
                                        'edge_covariance': covariance, 
                                        'coord_Measurement': m2_xy,
                                        'r_z_coords' : m2
                                        }
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))
            G.nodes[node]['track_state_estimates'] = state_estimates

    return GraphList


def construct_graph(graph, nodes, edges, truth, sigma0):
    tau = np.nan # track inclination from final and initial stating position - not needed

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
        gm = GNN_Measurement(x, y, tau, sigma0, label=label, n=node_idx)
        gm_3D = GNN_Measurement_3D(x, y, z, tau, sigma0, label=label, n=node_idx)
        graph.add_node(node_idx, GNN_Measurement=gm, 
                            GNN_Measurement_3D=gm_3D, 
                            coord_Measurement=(x, y),
                            coord_Measurement_3d=(x, y, z),
                            r_z_coords=(z, r),
                            volume_id = volume_id,
                            in_volume_layer_id = in_volume_layer_id,
                            truth_particle=label,)

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


def plotGraph(graph, filename):
    # plot network in xy
    _, ax = plt.subplots(figsize=(12,10))
    nodes = graph.nodes()
    pos = nx.get_node_attributes(graph,'coord_Measurement')
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=5, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.4, ax=ax)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Nodes & Edges extracted from TrackML generated data")
    # plt.savefig("images/xy_" + filename, dpi=300)
    # plt.show()

    # plot network in rz
    _, ax = plt.subplots(figsize=(12,10))
    pos = nx.get_node_attributes(graph,'r_z_coords')
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=5, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.4, ax=ax)
    plt.xlabel("z")
    plt.ylabel("r")
    plt.title("Nodes & Edges extracted from TrackML generated data")
    # plt.savefig("images/rz_" + filename, dpi=300)
    # plt.show()


def plot_subgraphs_in_plane(GraphList, outputDir, key, axis1, axis2):
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
    plt.xlabel(axis1)
    plt.ylabel(axis2)
    plt.title("Nodes & Edges extracted from TrackML generated data")
    plt.axis('on')
    # plt.savefig(outputDir + axis1 + axis2 + "_subgraphs_trackml_mod.png", dpi=300)
    # plt.show()

def plot_subgraphs(GraphList, outputDir):
    # xy plane
    plot_subgraphs_in_plane(GraphList, outputDir, 'coord_Measurement', "x", "y")
    # zr plane
    plot_subgraphs_in_plane(GraphList, outputDir, 'r_z_coords', "z", "r")