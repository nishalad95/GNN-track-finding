from community import community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
import igraph
# import louvain
import leidenalg as la
import os
import glob
from utilities.helper import *
import pprint



def community_detection(subGraph, fragment):

    edge_data = subGraph.edges.data()
    if len(edge_data) == 0: 
        print("zero edges in subgraph")
        return [], []

    # # make a copy & remove deactive edges
    sub = subGraph.copy()
    edges_to_remove = []
    for u, v in sub.edges():
        if sub[u][v]['activated'] == 0:
            edges_to_remove.append((u,v))  
    for u, v in edges_to_remove:
        sub.remove_edge(u, v)
  
    # # run a CCA and assign each subgraph a different partition - forms the intial partition ???
    # sub = nx.to_directed(G)
    # subGraphs = [sub.subgraph(c).copy() for c in nx.weakly_connected_components(sub)]

    # convert to igraph
    isub = igraph.Graph.from_networkx(sub)
    coords = isub.vs()["xy"]
    gnn_meas = isub.vs()["GNN_Measurement"]
    isub.es["weight"] = isub.es()["mixture_weight"]

    print("weights in igraph:\n", isub.es["weight"])
    labels = [n.node for n in gnn_meas]
    weights = isub.es["weight"]

    print("igraph: ", isub)
    print("gnn measurement", gnn_meas, len(gnn_meas))
    print("labels", labels, len(labels))
    print("weights", weights, len(weights))

    # print("Adjacency matrix:\n", isub.get_adjacency())

    # igraph.plot(isub, layout=coords, vertex_label=labels) #edge_label = weights)

    # find the partition
    partition = la.find_partition(isub, 
                                la.ModularityVertexPartition, 
                                weights=weights, 
                                n_iterations=-1)

                                
    
    
    igraph.plot(partition, layout=coords, vertex_label=labels)
    membership = np.array(partition.membership)
    counts = Counter(membership)

    valid_communities = []
    vc_coords = []
    all_node_nums = subGraph.nodes()
    for community, freq in counts.items():
        # check for no track fragments
        if freq > fragment:
            idx = np.where(membership == community)[0]
            community_nodes = [labels[i] for i in idx]
            # print(community_nodes)

            # check for 1 hit per layer
            vivl_id_values = nx.get_node_attributes(subGraph,'vivl_id').values()
            if len(vivl_id_values) == len(set(vivl_id_values)):
                print("good candidate - one hit per layer")
                good_candidate = True
            else:
                good_candidate = False

            # extract community
            if good_candidate:
                subCopy = subGraph.copy()
                nodes_to_remove = np.setdiff1d(all_node_nums, community_nodes)
                subCopy.remove_nodes_from(nodes_to_remove)
                valid_communities.append(subCopy)
                vc_coords.append(coords)

    return valid_communities, vc_coords




# ## TESTING 

# def plot_community(subGraph, partition, title):
#     _, ax = plt.subplots(figsize=(12,10))

#     pos=nx.get_node_attributes(subGraph,'xy')
#     cmap = cm.get_cmap('viridis', max(partition.values()))
#     nx.draw_networkx_nodes(subGraph, pos, partition.keys(), node_size=75,
#                             cmap=cmap, node_color=list(partition.values()))
#     nx.draw_networkx_edges(subGraph, pos, alpha=0.75)
#     nx.draw_networkx_labels(subGraph, pos)
#     ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
#     major_ticks = np.arange(0, 12, 1)
#     ax.set_xticks(major_ticks)
#     plt.xlim([0, 11])
#     plt.ylim([-27, 15])
#     plt.xlabel("ID layer in x axis")
#     plt.ylabel("y coordinate")
#     plt.title(title)
#     plt.axis('on')
#     plt.show()


inputDir = "src/output/iteration_1/remaining/"
fragment = 4
subgraph_path = "_subgraph.gpickle"
subGraphs = []
os.chdir(".")
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)
    # nx.set_edge_attributes(sub, 1, "activated")
    subGraphs.append(sub)

# print("SubGraphs:", subGraphs)
# subGraphs = subGraphs[0:5]

all_valid_communities = []
for s in subGraphs:
    valid_communities, _ = community_detection(s, fragment)
    print("valid_communities", valid_communities)
    if len(valid_communities) > 0:
        all_valid_communities.append(valid_communities)

print("number of track candidates extracted via community detection:", len(all_valid_communities))
# plot_subgraphs(all_valid_communities, "src/output/iteration_1/", node_labels=True, save_plot=True, title="Track candidates extracted via community detection")