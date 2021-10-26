import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
import csv


i = 0
remainingDir = "./output/iteration_2/remaining/"
subgraph_path = "_subgraph.gpickle"
path = remainingDir + str(i) + subgraph_path
remaining_networks = []

with open(r'remaining_edge_reweights.csv', 'a') as f:
    writer = csv.writer(f)


    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        remaining_networks.append(sub)
        i += 1
        path = remainingDir + str(i) + subgraph_path

        # for each remaining network
        for i, s in enumerate(remaining_networks):
            for node_1, node_2, edge_attributes in s.edges(data=True):
                
                truth_1 = s.nodes[node_1]['truth_particle']
                truth_2 = s.nodes[node_2]['truth_particle']
                
                truth = 0
                if truth_1 == truth_2:
                    truth = 1

                # print("edge:(", node_1, ", ", node_2, ")")
                # print("truth:", truth)
                # print("mixture_weight:", edge_attributes['mixture_weight'])

                writer.writerow([truth, edge_attributes['mixture_weight']])