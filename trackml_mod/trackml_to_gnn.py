import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import networkx as nx
import random
import argparse
from modules.GNN_Measurement import *
from helper import *
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
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    parser.add_argument('-m', '--mu', help="uncertainty due to multiple scattering, process noise")
    event_path = "trackml_mod/generated_events/event_1_filtered_graph_"
    truth_event_path = "trackml_mod/truth/event000001000-"
    truth_event_file = truth_event_path + "nodes-particles-id.csv"
    max_volume_region = 8000 # first consider endcap volume 7 only

    args = parser.parse_args()
    outputDir = args.outputDir         # output/track_sim/network/
    sigma0 = float(args.error)         # r.m.s measurement error
    mu = float(args.mu)                # process error - due to multiple scattering

    # load truth information & metadata on events
    nodes, edges = load_metadata(event_path, max_volume_region)
    # load_save_truth(event_path, truth_event_path, truth_event_file) #  only need to execute once
    truth = pd.read_csv(truth_event_file)

    # create a graph network
    endcap_graph = nx.DiGraph()
    construct_graph(endcap_graph, nodes, edges, truth, sigma0)

    # plot the network
    # plotGraph(endcap_graph, "endcap7_trackml_mod.png")
    print("Endcap volume 7 graph network:")
    print("Number of edges:", endcap_graph.number_of_edges())
    print("Number of nodes:", endcap_graph.number_of_nodes())

    # compute track state estimates, extract subgraphs: out-of-the-box CCA
    endcap_graph = compute_track_state_estimates([endcap_graph], sigma0, mu)
    endcap_graph = nx.Graph(endcap_graph[0])
    endcap_graph = nx.to_directed(endcap_graph)
    subGraphs = [endcap_graph.subgraph(c).copy() for c in nx.weakly_connected_components(endcap_graph)]
    subGraphs = compute_track_state_estimates(subGraphs, sigma0, mu)
    initialize_edge_activation(subGraphs)
    compute_prior_probabilities(subGraphs, 'track_state_estimates')
    compute_mixture_weights(subGraphs)

    print("Number of subgraphs..", len(subGraphs))
    # plot_subgraphs(subGraphs, outputDir)

    # save the subgraphs
    for i, sub in enumerate(subGraphs):
        save_network(outputDir, i, sub)

    print_graph_stats(outputDir)




if __name__ == "__main__":    
    main()