import numpy as np
import matplotlib.pyplot as plt
import os, glob, csv
import pandas as pd
import networkx as nx
import random
from math import *


separation_3d_threshold = 10
axis_lim = 180


def compute_3d_distance(coord1, coord2):
    x1, y1, z1 = coord1[0], coord1[1], coord1[2]
    x2, y2, z2 = coord2[0], coord2[1], coord2[2]
    return np.sqrt( (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2 )


def angle_trunc(a):
    while a < 0.0:
        a += pi * 2
    return a

# get angle to the positive x axis in radians
def getAngleBetweenPoints(p1, p2):
    deltaY = p2[1] - p1[1]
    deltaX = p2[0] - p1[0]
    return angle_trunc(atan2(deltaY, deltaX))

def rotate_track(coords):
    p1 = coords[-1]
    p2 = coords[-2]

    # if nodes p1 and p2 are too close, use the next node
    distance = compute_3d_distance(p1, p2)
    if distance < separation_3d_threshold:
        p2 = coords[-3]

    angle = 2*pi - getAngleBetweenPoints(p1, p2)
    # angle_degrees = angle * 180/pi
    # print("angle of rotation: ", angle_degrees)

    # rotate counter clockwise
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotated_coords = []
    for c in coords:
        x, y = c[0], c[1]
        x_new = x * cos_angle - y * sin_angle   # x_new = xcos(angle) - ysin(angle)
        y_new = x * sin_angle + y * cos_angle   # y_new = xsin(angle) + ycos(angle) 
        rotated_coords.append((x_new, y_new)) 
    return rotated_coords



def plot_subgraph(subGraph, pos=None):
    _, ax = plt.subplots(figsize=(10,8))
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6) ])][0]
    if pos == None: 
        pos=nx.get_node_attributes(subGraph, 'xy')
        print("pos:\n", pos)
    nodes = subGraph.nodes()
    edge_colors = []
    for u, v in subGraph.edges():
        if subGraph[u][v]['activated'] == 1: edge_colors.append(color)
        else: edge_colors.append("#f2f2f2")
    nx.draw_networkx_edges(subGraph, pos, edge_color=edge_colors, alpha=0.75)
    nx.draw_networkx_nodes(subGraph, pos, nodelist=nodes, node_color=color, node_size=50)
    nx.draw_networkx_labels(subGraph, pos, font_size=4)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('on')
    plt.xlim([-axis_lim, axis_lim])
    plt.ylim([-axis_lim, axis_lim])
    plt.show()



# read in subgraph data
inputDir = "src/output/iteration_1/network/"
subgraph_path = "_subgraph.gpickle"
subGraphs = []
os.chdir(".")
for file in glob.glob(inputDir + "*" + subgraph_path):
    sub = nx.read_gpickle(file)
    subGraphs.append(sub)


# select only every 10th subgraph
for i, subGraph in enumerate(subGraphs):
    
    if i % 5 == 0:
        # plot original track
        plot_subgraph(subGraph)

        # rotate track
        coords = list(nx.get_node_attributes(subGraph, 'xyzr').values())
        coords = sorted(coords, reverse=True, key=lambda xyzr: xyzr[3])
        coords = rotate_track(coords)

        # plot rotated track
        plt.subplots(figsize=(10,8))
        plt.scatter(*zip(*coords))
        plt.xlim([-axis_lim, axis_lim])
        plt.ylim([-axis_lim, axis_lim])
        plt.show()