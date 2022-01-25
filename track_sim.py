from networkx.generators.small import truncated_tetrahedron_graph
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import argparse
import json
from numpy.lib.function_base import average
from modules.GNN_Measurement import *
from modules.HitPairPredictor import HitPairPredictor
from utils.utils import *
import pprint

def plot_unique_labels(ax):
    handles,labels=ax.get_legend_handles_labels() #get existing legend item handles and labels
    i=np.arange(len(labels)) #make an index for later
    filter=np.array([]) 
    unique_labels=list(set(labels)) #find unique labels
    for ul in unique_labels: #loop through unique labels
        filter=np.append(filter,[i[np.array(labels)==ul][0]]) #find the first instance of this label and add its index to the filter
    handles=[handles[int(f)] for f in filter] #filter out legend items to keep only the first instance of each repeated label
    labels=[labels[int(f)] for f in filter]
    ax.legend(handles,labels)   

def main():

    # parse command line args
    parser = argparse.ArgumentParser(description='track hit-pair simulator')
    parser.add_argument('-t', '--threshold', help='variance of edge orientation')
    parser.add_argument('-o', '--output', help='output directory of simulation')
    parser.add_argument('-e', '--error', help="rms of track position measurements")
    args = parser.parse_args()
    
    threshold = float(args.threshold)
    outputDir = args.output
    sigma0 = float(args.error) #r.m.s of track position measurements

    Lc = np.array([1,2,3,4,5,6,7,8,9,10]) #detector layer coordinates along x-axis
    Nl = len(Lc) #number of layers
    start = 0
    y0 = np.array([-3,  1, -1,   3, -2,  2.5, -1.5]) #track positions at start, initial y
    yf = np.array([12, -4,  5, -14, -6, -24,   0]) #final track positions, final y

    # tau: track inclination, tau0 - gradient calculated from final & initial positions
    tau0 = (yf-y0)/(Lc[-1]-start)
    Ntr = len(y0) #number of simulated tracks
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(Ntr)]

    # plotting the intial track toy model
    fig = plt.figure(figsize=(16, 18))
    ax1 = fig.add_subplot(2, 2, 1)
    major_ticks = np.arange(0, 11, 1)
    ax1.set_xticks(major_ticks)

    G = nx.Graph()
    nNodes = 0
    mcoll = [] #collections of track position measurements
    for l in range(Nl) : mcoll.append([])

    # add hits to the graph network as nodes with node measurements
    print("Simulating hits & tracks with rms position measurement error: ", str(sigma0))
    for i in range(Ntr) :
        yc = y0[i] + tau0[i]*(Lc[:] - start)
        xc = Lc[:]
        node_labels = []
        for l in range(Nl) : 
            nu = sigma0 * np.random.normal(0.0,1.0) #random error
            y_pos = yc[l] + nu
            gm = GNN_Measurement(xc[l], y_pos, tau0[i], sigma0, label=i, n=nNodes)
            mcoll[l].append(gm)
            G.add_node(nNodes, GNN_Measurement=gm, 
                               coord_Measurement=(xc[l], y_pos),
                               mc_coord=(xc[l], yc[l]),
                               truth_particle=[i],
                               color=color[i])      # for plotting only
            nNodes += 1
            node_labels.append(nNodes)
        ax1.scatter(xc, yc, label=i, color=color[i])
    ax1.set_title('Ground truth')
    ax1.grid()
    plt.legend()

    print("number of nodes before hit merging:", G.number_of_nodes())

    # hit merging of close proximity hits
    for l in range(1, Nl+1, 1):
        
        # print("\nLAYER:", l)
        # sort the nodes in each layer by their y coord measurement
        layerNodes = [node for node in G.nodes(data=True) if node[1]['coord_Measurement'][0]==l]
        sortedLayerNodes = [node for node in sorted(layerNodes, key=lambda x: x[1]['coord_Measurement'][1])]
        
        # extract node nums & y coord measurement
        node_nums = []
        sorted_nodes_ypos = []
        for node in sorted(sortedLayerNodes, key=lambda x: x[1]['coord_Measurement'][1]):
            node_nums.append(node[0])
            sorted_nodes_ypos.append(node[1]['coord_Measurement'][1])
        # print("node nums:", node_nums)
        # print("sorted y pos:\n", l, "\n", sorted_nodes_ypos)

        # separate into bins and merge nodes
        bin_step = sigma0 * np.sqrt(12)
        bin_min = np.min(sorted_nodes_ypos)
        bin_max = np.max(sorted_nodes_ypos) + bin_step
        ybins = np.arange(bin_min, bin_max, bin_step)

        # loop through each ybin
        # print("ybins", ybins)
        for i in range(len(ybins) - 1):
            nodes_to_merge = []
            truth_particles = []
            bin_centre = (ybins[i] + ybins[i+1]) / 2          
            for j, ypos in enumerate(sorted_nodes_ypos):
                if ybins[i] <= ypos <= ybins[i+1]:
                    node_num = sortedLayerNodes[j][0]
                    truth = sortedLayerNodes[j][1]["truth_particle"]
                    nodes_to_merge.append(node_num)
                    truth_particles.append(truth[0])
            
            # remove all nodes in this bin & add a merged node
            # print("nodes to merge:", nodes_to_merge)
            # print("truth particles to merge:", truth_particles)
            if len(nodes_to_merge) > 1: 
                print("merging nodes:", nodes_to_merge, "in layer: ", l)
                for k, node_num in enumerate(nodes_to_merge):
                    if k == 0 :
                        G.nodes[node_num]['mc_coord'] = (l, bin_centre)
                        nu = sigma0 * np.random.normal(0.0,1.0) #random error
                        # add measurement error
                        y_pos = bin_centre + nu
                        # print("nu: ", nu)
                        # print("bin centre:", bin_centre)
                        # print("new y pos", y_pos)
                        G.nodes[node_num]['GNN_Measurement'].y = y_pos
                        G.nodes[node_num]['coord_Measurement'] = (l, y_pos)
                        G.nodes[node_num]['truth_particle'] = truth_particles
                    else:
                        print("removing node:" , node_num)
                        G.remove_node(node_num)
                
    print("number of nodes after hit merging:", G.number_of_nodes())

    # plot the network
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_xticks(major_ticks)
    for _, d in G.nodes(data=True):
        xc, yc = d['mc_coord'][0], d['mc_coord'][1]
        truth = d['truth_particle'][0]
        node_color = d['color']
        ax2.scatter(xc, yc, color=node_color, label=truth)
    ax2.set_title('Merged ground truth')
    ax2.grid()
    plot_unique_labels(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_xticks(major_ticks)
    for _, d in G.nodes(data=True):
        xc, yc = d['coord_Measurement'][0], d['coord_Measurement'][1]
        truth = d['truth_particle'][0]
        node_color = d['color']
        ax3.scatter(xc, yc, color=node_color, label=truth)
    ax3.set_title('Merged hits with measurement error')
    ax3.grid()
    plot_unique_labels(ax3)
     
    # # # save the num of hits for simulated tracks
    # # with open(outputDir + 'truth_hit_data.txt', 'w') as file:
    # #     file.write(json.dumps(num_hits))

    # generate hit pair predictions
    tau = 4.0
    hpp = HitPairPredictor(start, 7.0, tau) #max y0 and tau value
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_xticks(major_ticks)
    nPairs = 0

    # plot hit-pair predictions as edges, add edges to the graph network
    for L1 in range(Nl) :
        for L2 in range(L1+1,Nl) :
            if L2-L1 > 2 : break #use the next 2 layers only
            # if L2-L1 >= 2 : break #use the next layer only
            for m1 in mcoll[L1] :
                for m2 in mcoll[L2] :
                    # check if node exists in graph
                    if (G.has_node(m1.node) and G.has_node(m2.node)):
                        if m1.node == m2.node : continue
                        result = hpp.predict(m1, m2)
                        if result ==  0 : continue #pair (m1,m2) is rejected
                        nPairs += 1
                        ax4.plot([m1.x, m2.x],[m1.y,m2.y],alpha = 0.5)
                        edge = (m1.node, m2.node)
                        G.add_edge(*edge)
    print('Found', nPairs, 'hit pairs')

    # plot the ground truth and predicted hit pairs
    ax4.set_title('Hit pairs')
    ax4.grid()
    
    plt.tight_layout()
    plt.savefig(outputDir + 'simulated_tracks_hit_pairs.png', dpi=300)


    # compute track state estimates
    Graphs = compute_track_state_estimates([G], sigma0)
    G = nx.to_directed(Graphs[0])

    # plot are save temperature network
    print("Saving temperature network to serialized form & adjacency matrix...")
    plot_save_temperature_network(G, 'degree', outputDir)

    # remove all nodes with mean edge orientation above threshold
    G = nx.Graph(G) # make a copy to unfreeze graph
    filteredNodes = [(node, attr['coord_Measurement'])for node, attr in G.nodes(data=True) if attr['edge_gradient_mean_var'][1] > threshold]
    
    # # ensure no holes - for development purposes only
    # for i in range(Ntr):
    #     start_value = i * len(Lc)
    #     end_value = start_value + len(Lc) - 1
    #     values_to_check = np.linspace(start_value, end_value, num=len(Lc), dtype=int)
    #     intersection = sorted(list(set(values_to_check).intersection(filteredNodes)))
    #     # print("intersection", intersection)
    #     for j in range(len(intersection) - 1):
    #         if np.abs(intersection[j+1] - intersection[j]) != 1:
    #             for k in intersection[j+1:]:
    #                 if k in filteredNodes:
    #                     filteredNodes.remove(k)
    #     # print("filtered nodes", filteredNodes)
    
    print("removing nodes with variance of edge orientation greater than: ", threshold)
    for (node, _) in filteredNodes: 
        G.remove_node(node)
        # print("removing node:", node)
    
    # CCA: extract subgraphs
    G = nx.to_directed(G)
    subGraphs = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    
    # compute track state estimates, priors and assign initial edge weightings
    subGraphs = compute_track_state_estimates(subGraphs, sigma0)
    initialize_edge_activation(subGraphs)
    compute_prior_probabilities(subGraphs, 'track_state_estimates')
    compute_mixture_weights(subGraphs)
    
    # for i, s in enumerate(subGraphs):
    #     print("-------------------")
    #     print("SUBGRAPH " + str(i))
    #     print("-------------------")
    #     print("EDGE DATA:")
    #     for connection in s.edges.data():
    #         print(connection)
    #     print("-------------------")
    #     for node in s.nodes(data=True):
    #         pprint.pprint(node)
    #     print("--------------------")

    
    # plot and save extracted subgraphs
    print("Saving subgraphs to serialized form")
    title = "Weakly connected subgraphs extracted with variance of edge orientation <" + str(threshold)
    plot_save_subgraphs(subGraphs, outputDir + "/network/", title)


if __name__ == "__main__":
    main()