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
from utils.utils import *
import pprint


# def plot_community(subGraph, partition, title):
#     _, ax = plt.subplots(figsize=(12,10))

#     pos=nx.get_node_attributes(subGraph,'coord_Measurement')
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


# inputDir = "output_test1/iteration_2/remaining/"
inputDir = "output/iteration_2/remaining/"
fragment = 4

subgraph_path = "_subgraph.gpickle"
subGraphs = []
os.chdir(".")
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)

    edges_to_remove = []
    for u, v in sub.edges():
        if sub[u][v]['activated'] == 0:
            edges_to_remove.append((u,v))
    
    for u, v in edges_to_remove:
        sub.remove_edge(u, v)

    subGraphs.append(sub)


# print(subGraphs)
# # subGraphs = subGraphs[:1]
# # print(subGraphs)


# for G in subGraphs:
#     #first compute the best partition
#     # weight = "mixture_weight"
#     weight = "likelihood"
#     partition = community_louvain.best_partition(G, weight=weight)

#     # compute the best partition
#     partition = community_louvain.best_partition(G, weight=weight)


#     # draw the graph
#     plot_community(G, partition, "")
#     print()
#     for node in G.nodes(data=True):
#         pprint.pprint(node[1])


valid_communities = []

for sub in subGraphs:
    print("\n\n")
    edge_data = sub.edges.data()

    if len(edge_data) == 0: continue

    sub = igraph.Graph.from_networkx(sub)
    coords = sub.vs()["coord_Measurement"]
    gnn_meas = sub.vs()["GNN_Measurement"]
    weights = sub.es()["mixture_weight"]
    labels = [n.node for n in gnn_meas]

    # print("labels:", labels)
    # print("coords:", coords)
    # print("edge weights:", weights)
    igraph.plot(sub, layout=coords, vertex_label=labels)

    partition = la.find_partition(sub, 
                                la.ModularityVertexPartition, 
                                weights="mixture_weight", 
                                n_iterations=-1)
    
    igraph.plot(partition, layout=coords, vertex_label=labels)
    membership = np.array(partition.membership)
    print("\nMEMBERSHIP")
    print(membership)
    print("LABELS")
    print(labels)
    counts = Counter(membership)
    print("counts:", counts)

    for community, freq in counts.items():
        # check for no track fragments
        if freq > fragment:
            idx = np.where(membership == community)[0]
            nodes = [labels[i] for i in idx]
            print(nodes)

            # check for 1 hit per layer