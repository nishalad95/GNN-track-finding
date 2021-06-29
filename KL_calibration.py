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

def compute_track_recon_eff(outputDir):

    # simulated tracks
    with open("output/track_sim/truth_hit_data.txt") as json_file:
        sim_nhits_dict = json.load(json_file)
    num_sim_tracks = len(sim_nhits_dict)
    min_nhits = 4

    # reconstructed tracks
    inputDir = outputDir #"output/iteration_1/outlier_removal/"
    subgraph_path = "_subgraph.gpickle"
    recon_subGraphs = []
    os.chdir(".")
    for file in glob.glob(inputDir + "*" + subgraph_path):
        sub = nx.read_gpickle(file)
        recon_subGraphs.append(sub)

    # TODO: temporary, used for weights, need to change later
    num_remaining_hits = 0
    for r in recon_subGraphs:
        for node in r.nodes(): num_remaining_hits += 1
    print("Num remaining hits:", num_remaining_hits)


    # verify which simulated tracks have been reconstructed
    score = 0
    num_succ_sim_recon = 0
    for i in range(num_sim_tracks):
        sim_nhits = sim_nhits_dict[str(i)]['num_hits']
        sim_nodes = sim_nhits_dict[str(i)]['node_labels']
        if sim_nhits > min_nhits:
            for r in recon_subGraphs:
                # perform matching between recon and sim tracks
                match = False
                label_count = {}    # dict - k: simulated track label, v: count of nodes belonging to track label
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
                    match = True
                    num_succ_sim_recon += 1
                    break

            # compute score via track & particle purities
            if match:
                num_nodes_recon_track = len(r.nodes())
                num_nodes_sim_track = len(sim_nodes)
                intersection = list(set(r.nodes()) & set(sim_nodes))
                track_purity = len(intersection) / num_nodes_recon_track
                particle_purity = len(intersection) / num_nodes_sim_track
                # print("PURITIES", track_purity, particle_purity)

                if track_purity > 0.5 and particle_purity > 0.5:
                    # TODO: currently using equal weights for all nodes, can change to lower weighting for inner hits
                    score += len(intersection) * (1/num_remaining_hits)


    efficiency = num_succ_sim_recon / num_sim_tracks
    return efficiency, score