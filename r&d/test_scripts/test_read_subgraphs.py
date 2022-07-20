import matplotlib.pyplot as plt
import numpy as np
import csv
import glob
import os
import pandas as pd
import networkx as nx
import random
import pprint

# read in subgraph data
subgraph_path = "_subgraph.gpickle"
inputDir = "src/trackml_mod/output/iteration_2/candidates/"
subGraphs = []
i = 0
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    subGraphs.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path

for i, s in enumerate(subGraphs):
    if i <= 10 :
        print(s)
        for node in s.nodes(data=True):
            pprint.pprint(node)