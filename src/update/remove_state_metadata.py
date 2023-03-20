import os, glob
import numpy as np
import networkx as nx
import argparse
from utilities import helper as h
import pprint
import math
import csv


def main():

    # parse command line args
    parser = argparse.ArgumentParser(description='extract track candidates')
    parser.add_argument('-r', '--remain', help='output directory to save remaining network')
    args = parser.parse_args()

    # set variables
    remaining = args.remain

    # read in subgraph data
    subgraph_path = "_subgraph.gpickle"
    subGraphs = []
    os.chdir(".")
    for file in glob.glob(remaining + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        subGraphs.append(sub)

    # Remove the state metadata for connections that are no longer connected 
    # Looking for in edges to be active edges!
    for graph in subGraphs:
        for _, node in enumerate(graph.nodes(data=True)):
            node_num = node[0]
            node_attr = node[1]

            track_state_key = "track_state_estimates"
            if "updated_track_states" in node_attr.keys():
                track_state_key = "updated_track_states"
            states = node_attr[track_state_key]
            state_neighbours = list(states.keys())

            neighbours = [n for n in graph.neighbors(node_num)]
            for sn in state_neighbours:
                if sn not in neighbours: 
                    # remove this metadata from the state dictionary
                    state_dict = graph.nodes[node_num][track_state_key]                           
                    graph.nodes[node_num][track_state_key].pop(sn)
                    graph.nodes[node_num][track_state_key] = state_dict

    # update the priors and weights
    h.compute_prior_probabilities(subGraphs, 'track_state_estimates')
    h.compute_prior_probabilities(subGraphs, 'updated_track_states')
    h.reweight(subGraphs, 'updated_track_states')

    # save remaining and track fragments
    for i, sub in enumerate(subGraphs):
        h.save_network(remaining, i, sub)



if __name__ == "__main__":
    main()
