import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
import pandas as pd
import networkx as nx
import random


# private function
def __plot_subgraphs_in_plane(GraphList, outputDir, key, axis1, axis2, node_labels, save_plot, title):
    for i, subGraph in enumerate(GraphList):

        # if i == 49:
        _, ax = plt.subplots(figsize=(10,8))
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
        pos=nx.get_node_attributes(subGraph, key)
        nodes = subGraph.nodes()
        edge_colors = []
        for u, v in subGraph.edges():
            if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
            else: edge_colors.append("#f2f2f2")
        nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
        nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=65)
        if node_labels:
            nx.draw_networkx_labels(subGraph, pos, font_size=8)
        ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
        plt.xlabel(axis1)
        plt.ylabel(axis2)
        plt.title(title)
        plt.axis('on')
        if save_plot:
            plt.savefig(outputDir + axis1 + axis2 + "_" + str(i) + "_subgraphs_trackml_mod.png", dpi=300)


def plot_subgraphs(GraphList, outputDir, node_labels=False, save_plot=False, title=""):
    # xy plane
    __plot_subgraphs_in_plane(GraphList, outputDir, 'xy', "x", "y", node_labels, save_plot, title)
    # zr plane
    # __plot_subgraphs_in_plane(GraphList, outputDir, 'zr', "z", "r", node_labels, save_plot, title)


# read in subgraph data
subgraph_path = "_subgraph.gpickle"
inputDir = "src/output/iteration_1/remaining/"
subGraphs = []
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)

    # if len(sub.nodes()) >= 4:
    subGraphs.append(sub)

    i += 1
    path = inputDir + str(i) + subgraph_path

print("number of remaining subgraphs:", len(subGraphs))

outputDir="remaining/minCurv_0.3_134/"
plot_subgraphs(subGraphs, outputDir, node_labels=True, save_plot=True)