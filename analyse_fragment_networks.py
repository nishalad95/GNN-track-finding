import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import random
import itertools
from collections import Counter

# read in remaining
subGraphs = []
inputDir = "src/output/iteration_1/fragments/"
subgraph_path = "_subgraph.gpickle"
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path
print("Intial total no. of subgraphs:", len(subGraphs))

fragments = []
isolated_nodes = []
for s in subGraphs:
    if s.number_of_nodes() == 1:
        isolated_nodes.append(s)
    elif s.number_of_nodes() < 4:
        fragments.append(s)

print("number of isolated nodes: ", len(isolated_nodes))
percentage = len(isolated_nodes) * 100 / len(subGraphs)
print("percentage: ", percentage)
print("number of subgraphs with 2 or 3 nodes: ", len(fragments))
percentage = len(fragments) * 100 / len(subGraphs)
print("percentage: ", percentage)


# calculate the proportion of subgraphs that are track fragments
fragments = []
particle_ids = []
min_num_nodes = 4
for s in subGraphs:
    if s.number_of_nodes() < 4:
        fragments.append(s)

        # get the majority particle id from all nodes in the candidate to find out reconstructed particle id
        # get the hit dissociation to particle id for every node in each candidate
        # {node: {hit_id: [], particle_id: []} }, hits can be associated to more than 1 particle
        hit_dissociation = nx.get_node_attributes(s, 'hit_dissociation').values()
        gnn_particle_ids = []
        for hd in hit_dissociation:
            values = list(hd.values())
            gnn_particle_ids.append(values[1])
        gnn_particle_ids = list(itertools.chain(*gnn_particle_ids))
        freq_dist = Counter(gnn_particle_ids)
        particle_id = max(freq_dist, key=freq_dist.get)
        particle_ids.append(particle_id)

print("number of fragments:", len(fragments))
fraction = len(fragments) * 1.0 / len(subGraphs)
print("proportion of subgraphs that are track fragments:", fraction)

unique_particle_ids = list(set(particle_ids))
print("number of unique particle ids:", len(unique_particle_ids))

diff = len(fragments) - len(unique_particle_ids)
print("diff:", diff)



