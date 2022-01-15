import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os
import pprint
import csv
from sklearn import cluster
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets import make_blobs
from itertools import cycle
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch


i = 0
remainingDir = ["output/iteration_1/network/", "output/iteration_2/network/"]
outputfiles = ["shared_hit_identification/statistics_with_truth_1particle/iteration1.txt", "shared_hit_identification/statistics_with_truth_1particle/iteration2.txt"]
subgraph_path = "_subgraph.gpickle"
networks = []

for j, directory in enumerate(remainingDir):

    path = directory + str(i) + subgraph_path

    while os.path.isfile(path):
        sub = nx.read_gpickle(path)
        networks.append(sub)
        i += 1
        path = directory + str(i) + subgraph_path

    for s in networks:

        for node, node_attr in s.nodes(data=True):
            mixture_weights, theta_rad, theta_deg, gradient, truth = [], [], [], [], []
            
            print("\nProcessing node num:", node)

            # get inward facing edges to the node
            node_x = node_attr['coord_Measurement'][0]
            node_y = node_attr['coord_Measurement'][1]
            inward_edges = s.in_edges(node)
            outward_edges = s.out_edges(node)
            print("inward facing edges:", inward_edges)

            # for every edge, compute edge angle and extract mixture weight
            active_inward_edges = []
            for edge in inward_edges:
                edge_attr = s.get_edge_data(*edge)
                if edge_attr['activated'] == 1:
                    active_inward_edges.append(edge)


            print("active inward edges:", active_inward_edges)

            if len(active_inward_edges) <= 1: 
                print("Skipping node: ", node)
                continue

            
            print("calculating gradients")
            for edge in active_inward_edges:
                edge_attr = s.get_edge_data(*edge)

                print("edge:", edge)
                print("edge[0]", edge[0], "edge[1],", edge[1])
                print("edge_attr", edge_attr)

                weight = edge_attr['mixture_weight']
                neighbour_attr = s.nodes[edge[0]]

                neighbour_x = neighbour_attr['coord_Measurement'][0]
                neighbour_y = neighbour_attr['coord_Measurement'][1]
                dy_dx = (neighbour_y - node_y) / (neighbour_x - node_x)
                theta_r = np.arctan(dy_dx)
                theta_d = np.degrees(theta_r)

                print("mixture weight:", weight)
                print("neighbour xy:", neighbour_x, neighbour_y)
                print("node xy", node_x, node_y)
                print("dy/dx", dy_dx)
                # print("theta radians:", theta_r)
                print("theta degrees:", theta_d)

                mixture_weights.append(weight)
                # theta_rad.append(theta_r)
                theta_deg.append(theta_d)
                gradient.append(dy_dx)

                # compute truth value of edge
                # due to hit merging - if common value exists then MC truth = 1
                truth_node = node_attr['truth_particle']
                truth_neighbour = neighbour_attr['truth_particle']
                # common_truth = bool(set(truth_node) & set(truth_neighbour))
                edge_truth = 0
                # if common_truth: edge_truth = 1
                if truth_node[0] == truth_neighbour[0]: edge_truth = 1
                truth.append(edge_truth)


            # Compute agglomerative hierarchical clustering
            df = pd.DataFrame(columns =['mixture_weights', 'gradient'])
            df['mixture_weights'], df['gradient'] = mixture_weights, gradient
            df['active_inward_edges'] , df['edge_truth'] = active_inward_edges, truth

            print("DATAFRAME:")
            print(df)

            df = df.loc[df['edge_truth'] == 1]
            
            # linkage = ['ward', 'complete', 'average', 'single']
            linkage = ['average']
            labels_list = df['active_inward_edges'].tolist()

            df = df[['mixture_weights', 'gradient']]

            for link in linkage:
                Z = sch.linkage(df, method=link)
                # print("Z matrix using linkage: " + link)
                # print("cluster1, cluster2, dist, no. of elements")
                # print(Z)
                max_distance = np.amax(Z[:,2])
                # print("maximum distance: ", str(max_distance))
                # save the data
                with open(outputfiles[j], 'a') as f:
                    f.write(str(max_distance)+"\n")
