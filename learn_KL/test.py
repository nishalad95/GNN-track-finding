import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from utils import *
import pprint
import pickle
import csv

inputDir = "output/track_sim/sigma0.1/5000_events.gpickle"

open_file = open(inputDir, "rb")
events = pickle.load(open_file)

for _, subGraphs in events.items():
        # if i != 0 : continue
        for n, subGraph in enumerate(subGraphs):
            # if n != 1: continue

            # print("subgraph:", subGraph, type(subGraph))

            for node in subGraph.nodes(data=True):
            #     node_num = node[0]
                node_attr = node[1]

                if node_attr['edge_gradient_mean_var'][1] >= 0.6:
                    pprint.pprint(node)