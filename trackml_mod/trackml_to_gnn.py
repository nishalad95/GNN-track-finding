import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
from GNN_Measurement import GNN_Measurement

def plotGraph(graph):
    # plot network in xy
    _, ax = plt.subplots(figsize=(12,10))
    pos = nx.get_node_attributes(graph,'coord_Measurement')
    nx.draw_networkx_nodes(graph, pos, node_size=5)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    # nx.draw_networkx_labels(graph, pos)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Nodes & Edges extracted from TrackML generated data")
    #plt.savefig("xy_pixel78_trackml_mod.png", dpi=300)
    plt.show()

    # plot network in rz
    _, ax = plt.subplots(figsize=(12,10))
    pos = nx.get_node_attributes(graph,'r_z_coords')
    nx.draw_networkx_nodes(graph, pos, node_size=5)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    # nx.draw_networkx_labels(graph, pos)
    plt.xlabel("r")
    plt.ylabel("z")
    plt.title("Nodes & Edges extracted from TrackML generated data")
    #plt.savefig("rz_pixel78_trackml_mod.png", dpi=300)
    plt.show()

    # plot xyz
    _, ax = plt.subplots(figsize=(12,10))
    pos = nx.get_node_attributes(graph,'coord_Measurement_3d')
    nx.draw_networkx_nodes(graph, pos, node_size=5)
    plt.show()


# graph nodes
df_nodes = pd.read_csv("generated_events/event_1_filtered_graph_nodes.csv")
# select graph nodes in 7 & 8 pixel volumes
pixel_nodes_7_8 = df_nodes.loc[(df_nodes['layer_id'] >= 7000) & (df_nodes['layer_id'] <= 8999)] # look at barrel & endcap (left) only
pixel_nodes_7_8['r'] = pixel_nodes_7_8.apply(lambda row: np.sqrt(row.x**2 + row.y**2), axis = 1)
# print(pixel_nodes_7_8)

# graph edges
df_edges = pd.read_csv("generated_events/event_1_filtered_graph_edges.csv")
# set new header
new_header = df_edges.iloc[0] #grab the first row for the header
df_edges = df_edges[1:] #take the data less the header row
df_edges.columns = new_header #set the header row as the df header
# convert edge string tuple to ints
df_edges['node2i'] = df_edges.apply(lambda row: row.name[0], axis=1)
df_edges['node1i'] = df_edges.apply(lambda row: row.name[1], axis=1)
df_edges = df_edges.astype({'node2i': 'int32', 'node1i': 'int32'})
# select graph edges in 7 & 8 pixel volumes
pixel_edges_7_8 = df_edges.loc[(df_edges['node2i'] >= 7000) & (df_edges['node2i'] <= 8999) & 
                                (df_edges['node1i'] >= 7000) & (df_edges['node1i'] <= 8999)] 
# print(pixel_edges_7_8)


# create a graph network
# for every line in the df, compute gnn_measurement object
# label = MC truth track label (particle reference)
# n = node index
# gm = GNN_Measurement(xpos, ypos, dy_dx, sigma0, label=i, n=nNodes)
# G.add_node(nNodes, GNN_Measurement=gm, 
#                                coord_Measurement=(xc[l], y_pos),
#                                truth_particle=[i],
#                                color=color[i])  
G = nx.Graph()
sigma0 = 0.5
tau = 0 # TODO
for i in range(len(pixel_nodes_7_8)):
    row = pixel_nodes_7_8.iloc[i]
    x = row.x
    y = row.y
    z = row.z
    r = row.r
    n = row.node_idx
    label = i
    gm = GNN_Measurement(x, y, tau, sigma0, label=label, n=n)
    G.add_node(n, GNN_Measurement=gm, 
                  coord_Measurement=(x, y),
                  coord_Measurement_3d=(x, y, z),
                  r_z_coords=(z, r))

for i in range(len(pixel_edges_7_8)):
    row = pixel_edges_7_8.iloc[i]
    edge = (row.node2i, row.node1i)
    G.add_edge(*edge)


# plot the graph network
plotGraph(G)