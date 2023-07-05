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
import time


def main():

    parser = argparse.ArgumentParser(description='Convert trackml csv to GNN')
    parser.add_argument('-o', '--outputDir', help="Full directory path of where to save graph networks")
    parser.add_argument('-n', '--eventNetwork', help="Full directory path to event nodes, edges & nodes-to-hits")
    parser.add_argument('-t', '--eventTruth', help="Full directory path to event truth from TrackML")
    parser.add_argument('-a', '--min_volume', help="Minimum volume integer number in TrackML model to consider")
    parser.add_argument('-z', '--max_volume', help="Maximum volume integer number in TrackML model to consider")
    parser.add_argument('-e', '--sigma0xy', help="sigma0 rms of track position measurements in xy plane")
    parser.add_argument('-r', '--sigma0rz', help="sigma0 rms of track position measurements in rz plane")
    parser.add_argument('-m', '--sigma0rz2', help="sigma0 rms of track position measurements in rz plane - orientation of barrel and endcap layer")
    parser.add_argument('-b', '--endcapboundary', help="endcap boundary z coordinate - orientation of barrel and endcap layer")


    args = parser.parse_args()
    outputDir = args.outputDir
    min_volume = int(args.min_volume)
    max_volume = int(args.max_volume)
    sigma0xy = float(args.sigma0xy)
    sigma0rz = float(args.sigma0rz)
    sigma0rz2 = float(args.sigma0rz2)
    endcap_boundary = float(args.endcapboundary)

    # TODO: the following will get moved to .sh file
    # event_1 network corresponds to event000001000 truth
    event_network = args.eventNetwork + "/event_1_filtered_graph_"
    event_truth = args.eventTruth + "/event000001000-"
    event_truth_file = event_truth + "full-mapping-minCurv-0.3-800.csv"
    
    # # averaging over many events
    # # event_2 network corresponds to event000001001 truth
    # event_network = args.eventNetwork + "/event_4_filtered_graph_"
    # event_truth = args.eventTruth + "/event000001003-"
    # event_truth_file = event_truth + "full-mapping-minCurv-0.3-800.csv"

    start = time.time()

    # load truth information & metadata on events
    nodes, edges = h.load_nodes_edges(event_network, min_volume, max_volume)
    # NOTE: only need to execute the following once - aggregating all truth information into 1 file
    h.load_save_truth(event_network, event_truth, event_truth_file)
    truth = pd.read_csv(event_truth_file)

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion initial load: "+ str(total_time))
    start = time.time()

    # create a graph network
    pixel_graph_network = nx.DiGraph()
    print("here")
    pixel_graph_network = h.construct_graph(pixel_graph_network, nodes, edges, truth)
    print("Graph network info before processing:")
    print("Number of edges:", pixel_graph_network.number_of_edges())
    print("Number of nodes:", pixel_graph_network.number_of_nodes())

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion construct graph: "+ str(total_time))
    start = time.time()

    pixel_graph_network = nx.DiGraph(pixel_graph_network)

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion to_directed: "+ str(total_time))
    start = time.time()

    # extract subgraphs: out-of-the-box CCA
    subGraphs = [pixel_graph_network.subgraph(c).copy() for c in nx.weakly_connected_components(pixel_graph_network)]

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion CCA: "+ str(total_time))
    start = time.time()
    
    # compute track state estimates, priors and weights
    subGraphs = h.compute_track_state_estimates(subGraphs, sigma0xy, sigma0rz, sigma0rz2, endcap_boundary)
    h.initialize_edge_activation(subGraphs)
    h.compute_prior_probabilities(subGraphs, 'track_state_estimates')
    h.compute_mixture_weights(subGraphs, 'track_state_estimates')

    # add node degree as attribute
    for s in subGraphs:
        for node_num, _ in s.nodes(data=True):
            degree = h.query_node_degree_in_edges(s, node_num)
            s.nodes[node_num]['degree'] = degree

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion compute track state estimates: "+ str(total_time))
    start = time.time()

    print("Number of subgraphs..", len(subGraphs))
    
    # save the subgraphs
    for i, sub in enumerate(subGraphs):
        h.save_network(outputDir, i, sub)

    end = time.time()
    total_time = end - start
    print("Time taken in event_conversion write to file: "+ str(total_time))
    start = time.time()




if __name__ == "__main__":    
    main()