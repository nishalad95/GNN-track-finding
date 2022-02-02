import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
from modules.GNN_Measurement import *
from helper import *
import pprint
import time
import os
import glob



def main(outputDir):

    event_path = "generated_events/event_1_filtered_graph_"
    truth_event_path = "truth/event000001000-"
    truth_event_file = truth_event_path + "nodes-particles-id.csv"
    max_volume_region = 8000 # first consider endcap volume 7 only

    nodes, edges = load_metadata(event_path, max_volume_region)
    # load_save_truth(event_path, truth_event_path, truth_event_file) #  only need to execute once
    truth = pd.read_csv(truth_event_file)

    # create a graph network
    endcap_graph = nx.DiGraph()
    sigma0 = 0.5
    construct_graph(endcap_graph, nodes, edges, truth, sigma0)

    # plot the network
    # plotGraph(endcap_graph, "endcap7_trackml_mod.png")
    print("Endcap volume 7 graph network:")
    print("Number of edges:", endcap_graph.number_of_edges())
    print("Number of nodes:", endcap_graph.number_of_nodes())

    # compute track state estimates
    endcap_graph = compute_track_state_estimates([endcap_graph], sigma0)

    # # print node information
    # for i, s in enumerate(endcap_graph):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     print("-------------------")
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")

    # maybe we don't need to filter any nodes
    # # remove all nodes with mean edge orientation above threshold
    # # see how the algorithm behaves in the cold region and just the endcap
    # threshold = 0.8
    endcap_graph = nx.Graph(endcap_graph[0])
    # filteredNodes = [(node, attr['coord_Measurement'])for node, attr in endcap_graph.nodes(data=True) if attr['edge_gradient_mean_var'][1] > threshold]
    # print("Removing nodes with variance of edge orientation greater than: ", threshold)
    # for (node, _) in filteredNodes: 
    #     endcap_graph.remove_node(node)
    #     # print("removing node:", node)
    # print("Number of nodes removed:", len(filteredNodes))

    # out-of-the-box CCA: extract subgraphs
    endcap_graph = nx.to_directed(endcap_graph)
    subGraphs = [endcap_graph.subgraph(c).copy() for c in nx.weakly_connected_components(endcap_graph)]
    
    # compute track state estimates, priors and assign initial edge weightings
    subGraphs = compute_track_state_estimates(subGraphs, sigma0)
    initialize_edge_activation(subGraphs)
    compute_prior_probabilities(subGraphs, 'track_state_estimates')
    compute_mixture_weights(subGraphs)

    print("Number of subgraphs..", len(subGraphs))
    # plot_subgraphs(subGraphs)

    # save the subgraphs
    # outputDir = "output/track_sim/network/"
    for i, sub in enumerate(subGraphs):
        save_network(outputDir, i, sub)


def print_graph_stats(inputDir):
    # read in subgraph data
    # inputDir = "output/track_sim/network/"
    subgraph_path = "_subgraph.gpickle"
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    num_nodes = 0
    num_edges = 0
    for subGraph in subGraphs:
        num_nodes += len(subGraph.nodes())
        num_edges += len(subGraph.edges())

    print("num_nodes: ", num_nodes)
    print("node_edges: ", num_edges)



if __name__ == "__main__":
    start_time = time.time()
    
    outputDir = "output/track_sim/network/"
    main(outputDir)
    
    end_time = time.time()
    duration = end_time - start_time
    print("Execution time (s), trackml_to_gnn.py: ", duration)

    print_graph_stats(outputDir)