import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint

# temporary - read in the extracted candidates from this stage and previous iterations
# ^ this is already in the above file
i = 0
candidatesDir = "../output/iteration_2/candidates/"
# candidatesDir = "./output/iteration_2/remaining/"
subgraph_path = "_subgraph.gpickle"
path = candidatesDir + str(i) + subgraph_path
extracted = []
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    extracted.append(sub)
    i += 1
    path = candidatesDir + str(i) + subgraph_path

# print("extracted candidates:\n", extracted)

# for each extracted track candidate, for each node, collate all the mixture weights
for i, s in enumerate(extracted):
    # # all bidirectional edges
    print("\nEdge date for track candidate:", i)
    print(s.edges(data=True))
    weights_dict = nx.get_edge_attributes(s, 'mixture_weight')
    # weights = list(weights_dict.values())
    # # print("weights_dict", weights_dict)
    # plt.hist(weights, bins='auto')
    # plt.title("Extracted track candidate " + str(i))
    # plt.savefig("node_weight_dist/extracted/candidate_" + str(i) + "_total_dist.png", dpi=300)


    # for inward facing edges
    total_num_nodes = len(s.nodes())
    fig = plt.figure(figsize=(30, 5))
    for j, node in enumerate(s.nodes()):
        node_weights = []
        inward_edges = s.in_edges(node)
        print("node:", node)
        print("inward edges:\n", inward_edges)
        for edge in inward_edges:
            print(s.get_edge_data(*edge))
            if s.get_edge_data(*edge)['activated'] == 1:
                w = weights_dict[edge]
                node_weights.append(w)
                print("w:", w)

        ax1 = fig.add_subplot(1, total_num_nodes, j+1)
        ax1.hist(node_weights)
        ax1.set_title("node " + str(node))
    
    plt.savefig("node_weight_dist/extracted/candidate_" + str(i) + "_node_dist.png", dpi=300)
    # plt.savefig("node_weight_dist/remaining/candidate_" + str(i) + "_node_dist.png", dpi=300)