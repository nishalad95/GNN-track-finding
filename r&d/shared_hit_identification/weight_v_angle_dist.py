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
remainingDir = "../output_test5/iteration_1/network/"
subgraph_path = "_subgraph.gpickle"
path = remainingDir + str(i) + subgraph_path
networks = []

while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    networks.append(sub)
    i += 1
    path = remainingDir + str(i) + subgraph_path


for i, s in enumerate(networks):

    for node, node_attr in s.nodes(data=True):
        mixture_weights, theta_rad, theta_deg, gradient = [], [], [], []
        
        print("\nProcessing node num:", node)

        # get inward facing edges to the node
        node_x = node_attr['xy'][0]
        node_y = node_attr['xy'][1]
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
            neighbour_x = s.nodes[edge[0]]['xy'][0]
            neighbour_y = s.nodes[edge[0]]['xy'][1]
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

        
        # # plot distribution as subplots
        # # theta in radians
        # plt.figure()
        # plt.scatter(theta_rad, mixture_weights)
        # plt.xlabel("theta radians")
        # plt.ylabel("mixture weight")
        # plt.title('mixture weights vs theta for node ' + str(node))
        # plt.savefig("weight_v_theta_r/node_" + str(node) + ".png", dpi=300)

        # # theta in degrees
        plt.figure()
        plt.scatter(theta_deg, mixture_weights)
        plt.xlabel("theta degrees")
        plt.ylabel("mixture weight")
        plt.title('mixture weights vs theta for node ' + str(node))
        plt.savefig("weight_v_theta_d/node_" + str(node) + ".png", dpi=300)


        # Compute agglomerative hierarchical clustering
        df = pd.DataFrame(columns =['mixture_weights', 'gradient'])
        df['mixture_weights'], df['gradient'], df['active_inward_edges'] = mixture_weights, gradient, active_inward_edges
        
        linkage = ['ward', 'complete', 'average', 'single']
        labels_list = df['active_inward_edges'].tolist()

        df = df[['mixture_weights', 'gradient']]

        for link in linkage:
            plt.figure()
            Z = sch.linkage(df, method=link)
            print("Z matrix using linkage: " + link)
            print("cluster1, cluster2, dist, no. of elements")
            print(Z)
            max_distance = np.amax(Z[:,2])
            print("maximum distance: ", str(max_distance))
            dendogram_ward = sch.dendrogram(Z, labels=labels_list)
            plt.title("Dendogram: agglomerative clustering for node" + str(node))
            plt.ylabel("Cluster distance using linkage:" + link)
            plt.xlabel("Inward edge connection")
            plt.savefig("agglomerative/dendograms/" + link + "/node_" + str(node) + ".png", dpi=300)





        # # Clustering - Compute Affinity Propagation
        # num_points = len(mixture_weights)
        # coords = np.empty((num_points, 2))
        
        # coords[:,0] = mixture_weights
        # coords[:,1] = theta_deg

        # af = AffinityPropagation(random_state=0).fit(coords)
        # cluster_centers_indices = af.cluster_centers_indices_
        # labels = af.labels_

        # n_clusters_ = len(cluster_centers_indices)
        # print("num of clusters:", n_clusters_)
        # print("cluster_centers_indices", cluster_centers_indices)
        # print("labels", labels)

        # # Plot result
        # plt.figure(1)
        # plt.clf()

        # colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
        # for k, col in zip(range(n_clusters_), colors):
        #     class_members = labels == k
        #     cluster_center = coords[cluster_centers_indices[k]]
        #     plt.plot(coords[class_members, 1], coords[class_members, 0], col + ".")
        #     plt.plot(
        #         cluster_center[1],
        #         cluster_center[0],
        #         "o",
        #         markerfacecolor=col,
        #         markeredgecolor="k",
        #         markersize=14,
        #     )
        #     for x in coords[class_members]:
        #         plt.plot([cluster_center[1], x[1]], [cluster_center[0], x[0]], col)

        # plt.xlabel("theta degrees")
        # plt.ylabel("mixture weight")
        # plt.title("Estimated number of clusters: %d" % n_clusters_)
        # plt.savefig("clusters/node_" + str(node) + ".png", dpi=300)
