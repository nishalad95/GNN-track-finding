import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os
import argparse
import random
from utilities import helper as h
import argparse

# parse input arguments
parser = argparse.ArgumentParser(description='plot all extracted candidates')
parser.add_argument('-i', '--iterations', help='Total number of iterations')
args = parser.parse_args()
iteration_num = int(args.iterations)


# read in all candidates
extracted = []
candidatesDir = "src/output/iteration_" + str(iteration_num) + "/candidates/"
subgraph_path = "_subgraph.gpickle"
i=0
path = candidatesDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    extracted.append(sub)
    i+=1
    path = candidatesDir + str(i) + subgraph_path


# plot and save all candidates from all iterations
h.plot_save_subgraphs_iterations(extracted, candidatesDir, "Extracted candidates", node_labels=True, save_plot=True)