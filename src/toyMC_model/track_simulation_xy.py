import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import json
from GNN_Measurement import GNN_Measurement as GNN_Measurement
from HitPairPredictor import HitPairPredictor as HitPairPredictor
from utilities import helper as h
import pprint
import pickle
import itertools


def simulate_event():

    # generate some tracks
    num_hits = 10
    radius = 10
    angle_of_track = [1, 3, 6]      # num_tracks = len(angle_of_track) * 8
    # angle_of_track = [6]
    coords = np.array([])
    for i in angle_of_track:
        y = np.sqrt(radius**2 - i**2)    
        coords = np.append(coords, [i, y])
        coords = np.append(coords, [y, i])
        coords = np.append(coords, [-i, -y])
        coords = np.append(coords, [-y, -i])
        coords = np.append(coords, [i, -y])
        coords = np.append(coords, [-y, i])
        coords = np.append(coords, [-i, y])
        coords = np.append(coords, [y, -i])
    
    dimensions = ( int(len(coords) / 2), 2)
    coords = coords.reshape(dimensions)
    start = np.zeros( (int(len(coords)), 2) )
    num_tracks = len(start)
    # print("start\n", start, "end\n", coords)
    # print("num of tracks:", num_tracks)

    # create graph network
    G = nx.Graph()
    nNodes = 0
    sigma0 = 0.3        # measurement error r.m.s of track position measurement
    mu = 0.000001       # 10^-6 multiple scattering error - process noise

    # plot the tracks
    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    for n in range(num_tracks):
        gradient = (coords[n][1] - start[n][1]) / (coords[n][0] - start[n][0])
        x = np.linspace(start[n][0], coords[n][0], num_hits)
        y = np.array([])
        # print("track number: ", n)
        for layer, i in enumerate(x):
            nu = sigma0 * np.random.normal(0.0,1.0) #random error
            y_new = (gradient * i) + nu
            y = np.append(y, y_new)
            # add nodes to graph network
            gm = GNN_Measurement(i, y_new, 0, 0, sigma0, mu, label=n, n=nNodes)
            # print("layer:", layer)
            G.add_node(nNodes, GNN_Measurement=gm,
                        xy=(i, y_new),
                        zr=(0,0),
                        xyzr=(i, y_new, 0, 0),
                        truth=n,
                        layer=layer)
            nNodes += 1
        ax1.scatter(x[1:], y[1:], s=5, label=n)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid()
    ax1.legend(loc="best",fontsize=6)
    ax1.set_title("XY Toy MC Model")

    # generate hit pair predictions
    hpp = HitPairPredictor(0, 0.5, 3.5) #max y0 and tau value
    ax2 = fig.add_subplot(1, 2, 2)
    nPairs = 0

    node_gm = nx.get_node_attributes(G, 'GNN_Measurement') # dictionary
    for node1, gm1 in node_gm.items():
        for node2, gm2 in node_gm.items():
            if G.nodes[node2]["layer"] - G.nodes[node1]["layer"] > 2: break
            dx = gm2.x - gm1.x
            dy = gm2.y - gm1.y
            if np.sqrt( dx**2 + dy**2 ) > radius: break
            
            if node1 == node2: continue

            result = hpp.predict(gm1, gm2)
            if result == 0: continue # pair (node1, node2) rejected
            nPairs += 1
            ax2.plot([gm1.x, gm2.x],[gm1.y,gm2.y],alpha = 0.5)
            edge = (node1, node2)
            G.add_edge(*edge)
    print('Found', nPairs, 'hit pairs')
    
    ax2.set_title('Hit pairs')
    ax2.grid()
    plt.tight_layout()
    plt.savefig('simulated_tracks_hit_pairs_xy_plane.png', dpi=300)
    plt.show()

    




def main():
    simulate_event()



if __name__ == "__main__":
    main()