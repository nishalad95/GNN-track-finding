# assign each hit a weight:
    # weights should be 1/total num of hits in that event (including the nodes we removed)
    # sum of weights = 1 --> normalized

# assign each MC track to a particle
    # tracks are uniquely matched to particles using double majority rule
        # check whether we have managed to reconstruct all simulated tracks
        # for a given track, the matching particle is the one to which the absolute majority (strictly more that 50%) of the track points belong
        # the track should have the absolute majority of the points of the matching particle.
        # If any of these constraints is not met, the score for this track is zero

########################################

# focus on efficiency first

# input - cca_output reconstructed tracks, and current KL threshold, total number of simulated tracks
# calculate tracking efficiency function for each subgraph
# calculate purity
# print metrics
# adjust KL threshold


########################################

import os, json, glob
import networkx as nx
# import numpy as np

def calibrate():

    # simulated tracks
    with open("output/track_sim/sim_nhits.txt") as json_file:
        sim_nhits_dict = json.load(json_file)
    num_sim_tracks = len(sim_nhits_dict)
    min_nhits = 4

    # reconstructed tracks
    inputDir = "output/iteration_1/outlier_removal/"
    subgraph_path = "_subgraph.gpickle"
    recon_subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        recon_subGraphs.append(sub)


    # verify which simulated tracks have been reconstructed
    num_succ_sim_recon = 0
    for i in range(num_sim_tracks):
        if sim_nhits_dict[str(i)] > min_nhits:
            # print("SIM", i)
            for r in recon_subGraphs:
                # perform matching between recon and sim tracks
                label_count = {}    # dict simulated track label: count of nodes belonging to track label
                for node in r.nodes(data=True):
                    sim_track_label = node[1]["GNN_Measurement"].track_label

                    if sim_track_label in label_count.keys():
                        label_count[sim_track_label] = label_count.get(sim_track_label) + 1
                    else:
                        label_count[sim_track_label] = 1

                sim_track_label_list = list(label_count.keys())
                recon_count_list = list(label_count.values())
                total_num_hits = sum(recon_count_list)
                max_sim_track_label = max(recon_count_list)

                ratio = max_sim_track_label / total_num_hits
                idx = recon_count_list.index(max_sim_track_label)
                particle = sim_track_label_list[idx]
                
                if (particle == i) and (ratio > 0.5):
                    # print("MATCH")
                    num_succ_sim_recon += 1
                    break

        # print("total num succ sim recon", num_succ_sim_recon)

    efficiency = num_succ_sim_recon / num_sim_tracks
    return efficiency


# particle_idx = np.linspace(0, num_sim_tracks-1, num_sim_tracks, dtype=int)
# reconstructed = []
# purity = []
# print(particle_idx)

# # read in subgraph data
# subGraphs = []
# os.chdir(".")
# for file in glob.glob(inputDir + "*" + subgraph_path):
#     sub = nx.read_gpickle(file)
#     subGraphs.append(sub)

# # calculate track reconstruction efficiency
# for subGraph in subGraphs:
    
#     # only execute for subgraphs with 4 hits or more
#     if len(subGraph.nodes()) < 4 : continue
    
#     # dictionary: k - simulated track label, v - count of nodes belonging to that track label
#     recon_track_label = {}
#     for node in subGraph.nodes(data=True):
#         sim_track_label = node[1]["GNN_Measurement"].track_label

#         if sim_track_label in recon_track_label.keys():
#             recon_track_label[sim_track_label] = recon_track_label.get(sim_track_label) + 1
#         else:
#             recon_track_label[sim_track_label] = 1

#     keys_list = list(recon_track_label.keys())
#     values_list = list(recon_track_label.values())
#     total_num_modes = sum(values_list)
#     max_sim_track_label = max(values_list)

#     if max_sim_track_label / total_num_modes > 0.5:
#         idx = values_list.index(max_sim_track_label)
#         particle = keys_list[idx]
#         reconstructed.append(particle)
#         purity.append(max_sim_track_label / total_num_modes)

# reconstructed = np.unique(reconstructed)
# efficiency = len(reconstructed) / num_sim_tracks
# print("efficiency:", efficiency)
# print("purity", purity)
# average_purity = np.average(purity)
# print("average purity", average_purity)