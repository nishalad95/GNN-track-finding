import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
import argparse
from utilities import helper as h
import pprint
import os
import glob


def print_graph_stats(inputDir):
    # read in subgraph data
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


def main():

    # TODO: command line args, change these into command line arguments with full path directory
    parser = argparse.ArgumentParser(description='Convert trackml csv to GNN')
    parser.add_argument('-o', '--outputDir', help="Full directory path of where to save graph networks")
    # parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-m', '--sigma_ms', help="uncertainty due to multiple scattering, process noise")
    parser.add_argument('-n', '--eventNetwork', help="Full directory path to event nodes, edges & nodes-to-hits")
    parser.add_argument('-t', '--eventTruth', help="Full directory path to event truth from TrackML")
    
    # TODO: temporary, only considering endcap volume 7
    max_volume_region = 8000 # first consider endcap volume 7 only

    args = parser.parse_args()
    outputDir = args.outputDir
    # sigma0 = float(args.error)         # r.m.s measurement error
    sigma_ms = float(args.sigma_ms)                # process error - due to multiple scattering
    
    # TODO: the following will get moved to .sh file
    # event_1 network corresponds to event000001000 truth
    event_network = args.eventNetwork + "/event_3_filtered_graph_"
    event_truth = args.eventTruth + "/event000001002-"
    event_truth_file = event_truth + "full-mapping-minCurv-0.3-800.csv"

    # load truth information & metadata on events
    nodes, edges = h.load_nodes_edges(event_network, max_volume_region)
    h.load_save_truth(event_network, event_truth, event_truth_file) #  only need to execute once
    truth = pd.read_csv(event_truth_file)

    # create a graph network
    endcap_graph = nx.DiGraph()
    endcap_graph = h.construct_graph(endcap_graph, nodes, edges, truth, sigma_ms)

    # debugging
    # print("-----------------------------------------")
    # print("ENDCAP GRAPH:")
    # print("-----------------------------------------")
    # for node in endcap_graph.nodes(data=True):
    #     pprint.pprint(node)
    # print("-----------------------------------------")
    # print("EDGE DATA:", endcap_graph.edges.data(), "\n")
    # print("-----------------------------------------")

    print("Endcap volume 7 graph network:")
    print("Number of edges:", endcap_graph.number_of_edges())
    print("Number of nodes:", endcap_graph.number_of_nodes())

    # temporary: can remove after
    f = open("parabolic_param_a.txt", "w")
    f.write("Before CCA \n")
    f.close()

    # compute track state estimates, extract subgraphs: out-of-the-box CCA
    endcap_graph = h.compute_track_state_estimates([endcap_graph])
    endcap_graph = nx.Graph(endcap_graph[0])
    endcap_graph = nx.to_directed(endcap_graph)

    # temporary: can remove later
    print("Number of edges again:", endcap_graph.number_of_edges())
    print("Number of nodes again:", endcap_graph.number_of_nodes())

    subGraphs = [endcap_graph.subgraph(c).copy() for c in nx.weakly_connected_components(endcap_graph)]
    
    # temporary: can remove after
    f = open("parabolic_param_a.txt", "a")
    f.write("After CCA \n")
    f.close()
    
    subGraphs = h.compute_track_state_estimates(subGraphs)
    h.initialize_edge_activation(subGraphs)
    h.compute_prior_probabilities(subGraphs, 'track_state_estimates')
    h.compute_mixture_weights(subGraphs)

    print("Number of subgraphs..", len(subGraphs))
    h.plot_subgraphs(subGraphs, outputDir, title="Nodes & Edges subgraphs from TrackML generated data")
    
    # save the subgraphs
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)

    print_graph_stats(outputDir)




if __name__ == "__main__":    
    main()