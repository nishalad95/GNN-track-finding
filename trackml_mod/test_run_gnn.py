import networkx as nx
import os
import glob
from helper import plot_subgraphs

# read in subgraph data
subGraphs = []
os.chdir(".")
inputDir = "output/track_sim/network/"
subgraph_path = "_subgraph.gpickle"
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)
    subGraphs.append(sub)

subGraphs = subGraphs[:100]
plot_subgraphs(subGraphs)