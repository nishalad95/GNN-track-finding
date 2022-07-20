import networkx as nx
import numpy as np
import glob
from utils import query_node_degree_in_edges


# read in subgraph data
events = []
for file in glob.glob("output/track_sim/sigma0.5/100_events.gpickle"):
    event = nx.read_gpickle(file)
    events.append(event)

print("length of total events:", len(events))
print("events:", events)

for idx, event in enumerate(events):

    print("processing event:", str(idx))
    subGraphs = list(event.values())[0]
    for n, subGraph in enumerate(subGraphs):

        # print("processing the subGraph:", subGraph)
        if n != 1: continue

        for node in subGraph.nodes(data=True):
            node_num = node[0]
            node_attr = node[1]

            emp_var = node_attr['edge_gradient_mean_var'][1]
            num_edges = query_node_degree_in_edges(subGraph, node_num)
            if num_edges <= 1: continue

            # convert attributes to arrays
            track_state_estimates = node_attr["track_state_estimates"]
            print("track state estimates:\n", track_state_estimates)