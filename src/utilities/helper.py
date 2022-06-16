import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
from GNN_Measurement import GNN_Measurement as gnn
import pprint
from math import *


def edge_length_xy(row):
    return np.sqrt(row.x**2 + row.y**2)

def get_volume_id(layer_id):
    return int(layer_id / 1000)

def get_in_volume_layer_id(layer_id):
    return int(layer_id % 100)

# def get_module_id(node_idx, truth_df):
#     return truth_df.loc[truth_df.node_idx == node_idx]['module_id']

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


# TODO: some of these functions are in the rotation of a track during the extraction
# they can be used from these functions instead - duplication of code
def compute_3d_distance(coord1, coord2):
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a


# get angle to the positive x axis in radians
def getAngleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    return angle_trunc(atan2(deltaY, deltaX))


def rotate_track(coords, separation_3d_threshold=None):
    # coords are ordered from outermost to innermost -> use innermost edge
    p1 = coords[-1]
    p2 = coords[-2]

    # if nodes p1 and p2 are too close, use the next node
    if separation_3d_threshold is not None:    
        distance = compute_3d_distance(p1, p2)
        if distance < separation_3d_threshold:
            p2 = coords[-3]

    # rotate counter clockwise, first edge to be parallel with x axis
    angle = 2*pi - getAngleBetweenPoints(p1, p2)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_coords = []
    for c in coords:
        x, y = c[0], c[1]
        x_new = x * np.cos(angle) - y * np.sin(angle)    # x_new = xcos(angle) - ysin(angle)
        y_new = x * np.sin(angle) + y * np.cos(angle)    # y_new = xsin(angle) + ycos(angle) 
        rotated_coords.append((x_new, y_new)) 
    return rotated_coords

def compute_track_state_estimates(GraphList, sigma0):
    sigma0 = 4.0        # 4.0mm larger error in m_0 due to beamspot error
    sigmaA = 0.1        # 0.1mm
    sigmaB = 0.1        # 0.1mm
    S = np.array([  [sigma0**2,         0,                  0], 
                    [0,                 sigmaA**2,          0], 
                    [0,                 0,                  sigmaB**2]])        # edge covariance matrix
    
    m_0 = 0.0   # measurements for parabolic model
    m_A = 0.0
    for i, G in enumerate(GraphList):
        for node in G.nodes():
            gradients = []
            track_state_estimates = {}
            
            # create a list of node & neighbour coords including the origin
            node_gnn = G.nodes[node]["GNN_Measurement"]
            m_node = (node_gnn.x, node_gnn.y)
            coords = [(0.0, 0.0), m_node]
            keys = [-1, node]

            for neighbor in nx.all_neighbors(G, node):
                neighbour_gnn = G.nodes[neighbor]["GNN_Measurement"]
                m_neighbour = (neighbour_gnn.x, neighbour_gnn.y)
                coords.append(m_neighbour)
                keys.append(neighbor)

                dy = m_node[1] - m_neighbour[1]
                dx = m_node[0] - m_neighbour[0]
                grad = dy / dx
                gradients.append(grad)
            
            # [neighbour1, neighbour2, ..., node, (0.0, 0.0)]
            coords.reverse()
            keys.reverse()

            # rotate the node & its neighbours to local node-specific coordinate system
            rotated_coords = rotate_track(coords)

            # translate all coords such that the node in question becomes the new origin
            x_trans = rotated_coords[-2][0]
            y_trans = rotated_coords[-2][1]
            transformed_coords = []
            for rc in rotated_coords:
                tx = rc[0] - x_trans
                ty = rc[1] - y_trans
                tc = (tx, ty)
                transformed_coords.append(tc)
            # print("TRANSLATED COORDS:\n", transformed_coords)

            # for each neighbour connection obtain the measurement vector in the new axis
            # [m_0, m_A, m_B] m_0 the old origin, m_A the new origin, m_B the neighbour
            x_0 = transformed_coords[-1][0]
            transformed_neighbour_coords = transformed_coords[:-2]
            keys = keys[:-2]
            for tnc, key in zip(transformed_neighbour_coords, keys):
                x_B = tnc[0]
                m_B = tnc[1]
                measurement_vector = [m_0, m_A, m_B]
                H = np.array([  [x_0**2,         x_0,          1], 
                                [0,                0,          1], 
                                [x_B**2,         x_B,          1]])

                # compute track state parameters & covariance
                H_inv = np.linalg.inv(H)    # invert H matrix to obtain measurement matrix
                track_state_vector = H_inv.dot(measurement_vector)
                covariance = H_inv.dot(S).dot(H_inv.T)
                track_state_estimates[key] = {  'edge_state_vector': track_state_vector, 
                                                'edge_covariance': covariance }

            # TODO: debugging
            # if i == 0:
            #     print("track state estimates:\n", track_state_estimates)
            # store all track state estimates at the node
            G.nodes[node]['track_state_estimates'] = track_state_estimates
            # (mean, variance) of edge orientation - needed for KL distance in clustering
            G.nodes[node]['edge_gradient_mean_var'] = (np.mean(gradients), np.var(gradients))

    return GraphList



def __get_particle_id(row, df):
    particle_ids = []
    for hit_id in row:
        particle_id = df.loc[df.hit_id == hit_id]['particle_id'].item()
        particle_ids.append(particle_id)
    return particle_ids



def construct_graph(graph, nodes, edges, truth, sigma0, mu):
    # TODO: 'truth_particle' attribute needs to be updated - clustering calculation uses 1 particle id, currently needs updating
    # group truth particle ids to node index
    group = truth.groupby('node_idx')
    grouped_pid = group.apply(lambda row: row['particle_id'].unique())
    grouped_pid = pd.DataFrame({'node_idx':grouped_pid.index, 'particle_id':grouped_pid.values})
    grouped_pid['single_particle_id'] = grouped_pid['particle_id'].str[0]

    # node to hit dissociation: group truth hits to node index
    grouped_hit_id = group.apply(lambda row: row['hit_id'].unique())
    grouped_hit_id = pd.DataFrame({'node_idx':grouped_hit_id.index, 'hit_id':grouped_hit_id.values})
    grouped_hit_id['particle_id'] = grouped_hit_id['hit_id'].apply(lambda row: __get_particle_id(row, truth))
    grouped_hit_id['hit_dissociation'] = grouped_hit_id.apply(lambda row: {"hit_id": row['hit_id'], "particle_id":row['particle_id']}, axis=1)

    grouped_module_ids = group['module_id'].unique()

    # add nodes
    for i in range(len(nodes)):
        row = nodes.iloc[i]
        node_idx = int(row.node_idx)
        x, y, z, r = row.x, row.y, row.z, row.r
        volume_id, in_volume_layer_id= row.volume_id, row.in_volume_layer_id

        # get module ids linked to this node (multiple here)
        module_id = grouped_module_ids[node_idx]
        
        # TODO: update the following
        label = grouped_pid.loc[grouped_pid['node_idx'] == node_idx]['single_particle_id'].item()  # MC truth label (particle id)
        
        hit_dissociation = grouped_hit_id.loc[grouped_hit_id['node_idx'] == node_idx]['hit_dissociation'].item()
        
        gm = gnn.GNN_Measurement(x, y, z, r, sigma0, mu, label=label, n=node_idx)
        graph.add_node(node_idx, GNN_Measurement=gm, 
                            xy=(x, y),                              # all attributes here for development only - can be abstracted away in GNN_Measurement
                            zr=(z, r),
                            xyzr=(x,y,z,r),                         # temporary - used in extract track candidates                             
                            volume_id = volume_id,
                            in_volume_layer_id = in_volume_layer_id,
                            vivl_id = (volume_id, in_volume_layer_id),
                            module_id = module_id,
                            truth_particle=label,   # TODO: update the following
                            hit_dissociation=hit_dissociation)
    
    # add bidirectional edges
    for i in range(len(edges)):
        row = edges.iloc[i]
        node1, node2 = row.node1, row.node2
        # if both nodes in network, add edge
        if (node1 in graph.nodes()) and (node2 in graph.nodes()):
            graph.add_edge(node1, node2)
            graph.add_edge(node2, node1)



# TODO: rename to load_nodes_edges
def load_metadata(event_path, max_volume_region):
    # graph nodes
    nodes = pd.read_csv(event_path + "nodes.csv")
    # select nodes in region of interest
    nodes = nodes.loc[nodes['layer_id'] <= max_volume_region]
    nodes['r'] = nodes.apply(lambda row: edge_length_xy(row), axis=1)
    nodes['volume_id'] = nodes.apply(lambda row: get_volume_id(row.layer_id), axis=1) 
    nodes['in_volume_layer_id'] = nodes.apply(lambda row: get_in_volume_layer_id(row.layer_id), axis=1)
    # nodes['module_id'] = nodes.apply(lambda row: get_module_id(), axis=1)

    # graph edges
    # select all edges - TODO: select only edges where node1 and node2 contained in nodes df (above)
    # TODO: much faster way of doing this
    edges = pd.read_csv(event_path + "edges.csv")
    new_header = edges.iloc[0] #grab the first row for the header
    edges = edges[1:] #take the data less the header row
    edges.columns = new_header #set the header row as the df header
    edges['node2'] = edges.apply(lambda row: row.name[0], axis=1)
    edges['node1'] = edges.apply(lambda row: row.name[1], axis=1)
    edges = edges.astype({'node2': 'int32', 'node1': 'int32'})

    # return as dataframes
    return nodes, edges


def load_save_truth(event_path, truth_event_path, truth_event_file):
    # truth event
    hits_particles = pd.read_csv(truth_event_path + "truth.csv")
    particles_nhits = pd.read_csv(truth_event_path + "particles.csv")
    hits_module_id = pd.read_csv(truth_event_path + "hits.csv")

    # nodes to hits
    nodes_hits = pd.read_csv(event_path + "nodes_to_hits.csv")
 
    # for every node and hit_id mapping, get the particle_id & module_id
    truth = pd.DataFrame(columns = ['node_idx', 'hit_id', 'particle_id', 'module_id', 'nhits'])
    truth['node_idx'] = nodes_hits['node_idx']
    truth['hit_id'] = nodes_hits['hit_id']
    for index, row in truth.iterrows():
        hit_id = row.hit_id
        truth['particle_id'][index] = hits_particles.loc[hits_particles.hit_id == hit_id].particle_id.item()
        truth['module_id'][index] = hits_module_id.loc[hits_module_id.hit_id == hit_id].module_id.item()

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


# private function
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
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=4)
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



# private function
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
        nx.draw_networkx_nodes(subGraph, pos, node_color=color, node_size=50, label=iteration)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=4)
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
