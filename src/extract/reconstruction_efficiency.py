import csv
import pandas as pd
import numpy as np
from statistics import mode
import os
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from collections import Counter
import argparse
import matplotlib.pyplot as plt


# parse command line args
parser = argparse.ArgumentParser(description='extract track candidates')
parser.add_argument('-t', '--eventTruth', help="Full directory path to event truth from TrackML")
parser.add_argument('-o', '--output', help='output directory to save data')
parser.add_argument('-a', '--min_volume', help="Minimum volume integer number in TrackML model to consider")
parser.add_argument('-z', '--max_volume', help="Maximum volume integer number in TrackML model to consider")
parser.add_argument('-i', '--iterations', help='Total number of iterations')

args = parser.parse_args()
outputDir = args.output
event_truth = args.eventTruth
min_volume = int(args.min_volume)
max_volume = int(args.max_volume)
iteration_num = int(args.iterations)
inputDir = outputDir + "/iteration_" + str(iteration_num) + "/candidates/"

# every hit has an associated particle id: from the truth information file for hits
# full info about particles itself is contained in the particles file


#########################################
# Identify reference tracks from MC truth
#########################################

reference_path = event_truth + "/event000001000-"
# reference_path = event_truth + "/event000001003-"

# identify the particle ids that have pT > 1GeV, units GeV/c
particles = pd.read_csv(reference_path + "particles.csv", sep=',')
particles = particles.assign(pT=lambda row: (np.sqrt(row.px**2 + row.py**2)))
# all_particles = particles
pt_cut = 1.0  # units of momentum GeV/c
particles = particles.loc[particles.pT >= pt_cut]
particle_ids = particles.particle_id.to_list()

# go through the truth file for hits & identify hits corresponding to each particle_id
truth = pd.read_csv(reference_path + "truth.csv", sep=',')
hit_ids = truth.loc[truth.particle_id.isin(particle_ids)]
hit_ids = hit_ids.hit_id.to_list()

# TODO: TEMPORARY - subset of hits contained in the volume of interest
# extract the subset of hits contained in the volume of interest
hits = pd.read_csv(reference_path + "full-mapping-minCurv-0.3-800.csv", sep=',')
pixel_hits = hits.loc[(hits.hit_id.isin(hit_ids)) 
                        & (hits.volume_id >= min_volume)
                        & (hits.volume_id <= max_volume)]
print("Total number of reference tracks in range ", str(min_volume), " to ", str(max_volume))
print(len(pixel_hits.particle_id.unique()))


# Compute number of reference tracks: Apply a cut on the num of distinct layers for all the hits of a particle
# due to the many-to-1 hits-to-node mapping, we require a cut on the number of distinct layers
num_distinct_layers_required = 4
num_reference_tracks = 0
reference_tracks_dict = {}          # key:value particle_id:num_distinct_layers_endcap
unique_particle_ids = pixel_hits.particle_id.unique()
for i, p in enumerate(unique_particle_ids):
    ref_track = pixel_hits.loc[pixel_hits.particle_id == p]
    # calculate number of distinct layers there are in this reference track
    unique_vivl_ids = set(list(zip(ref_track.volume_id, ref_track.layer_id)))
    num_layers_queried = len(unique_vivl_ids)
    if num_layers_queried >= num_distinct_layers_required:
        # good reference track
        # check 'one-hit-per-module' constraint
        one_hit_per_module = ref_track[ref_track.duplicated(['volume_id', 'layer_id', 'module_id'], keep=False)]
        if len(one_hit_per_module) == 0:
            # good reference track found
            num_reference_tracks += 1
            reference_tracks_dict[p] = ref_track.hit_id.to_list()   # add to dictionary
        else:
            # TODO: handle this case
            print(">1 hit per module!! TODO: will need to handle this case!")
            print("Need to check for same volume id & layer id, but one-hit-per-module")
            print(one_hit_per_module)
    # else:
    #     print("Too few number of distinct layers in reference track: ", num_layers_queried)

print("Number of good reference tracks according to our criteria:\n", num_reference_tracks)




# #########################################################
# # Look at what tracks have been successfully reconstructed
# # Track matching stage:
# #########################################################

track_candidates = []
i = 0
subgraph_path = "_subgraph.gpickle"
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    track_candidates.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path

print("inputDir:\n", inputDir)
print("number of track candidates to process:", len(track_candidates))


# # distribution of reconstructed tracks pTs:
# pT_dist_before = []

num_reconstructed_tracks = 0
track_purities = np.array([])
particle_purities = np.array([])
reconstructed_particle_ids_set = set()

# truth_num_hits = hits
truth_num_hits = pixel_hits
print("truth_num_hits:\n", truth_num_hits)

for i, track in enumerate(track_candidates):
    # if i == 20:   # temporary for debugging
    print("------- Processing track candidate ", str(i), " -------")
    
    # get the majority particle id from all nodes in the candidate to find out reconstructed particle id
    # get the hit dissociation to particle id for every node in each candidate
    # {node: {hit_id: [], particle_id: []} }, hits can be associated to more than 1 particle
    hit_dissociation = nx.get_node_attributes(track, 'hit_dissociation').values()
    gnn_particle_ids = []
    for hd in hit_dissociation:
        values = list(hd.values())
        gnn_particle_ids.append(values[1])
    gnn_particle_ids = list(itertools.chain(*gnn_particle_ids))
    freq_dist = Counter(gnn_particle_ids)
    reconstructed_particle_id = max(freq_dist, key=freq_dist.get)
    n_good = freq_dist[reconstructed_particle_id] * 1.0

    # pT_value_before = all_particles.loc[all_particles.particle_id == reconstructed_particle_id].pT.item()
    # print("pT value before:", pT_value_before)
    # pT_dist_before.append(pT_value_before)
    # print("pT_dist_before", pT_dist_before)

    print("gnn particle ids:\n", gnn_particle_ids)
    print("freq dist:\n", freq_dist)
    print("reconstructed particle id:\n", reconstructed_particle_id)
    print("n_good:\n", n_good)

    # check if reconstructed track particle_id appears in reference tracks
    if reconstructed_particle_id in reference_tracks_dict.keys():
        print("HURRAY! Reconstructed particle id exists in reference tracks particles!")
        reference_hits = reference_tracks_dict[reconstructed_particle_id]
        if n_good >= 0.5 * len(reference_hits):
            print("track reconstructed by gnn! particle_id: ", reconstructed_particle_id)
            
            # calculate the purity
            track_purity = n_good / len(gnn_particle_ids)
            # nhits_df = truth_num_hits.loc[truth_num_hits.particle_id == reconstructed_particle_id]['nhits_endcap_volume7']
            # we want the total number of hits generated by this particle in the region we are considering (pixel)
            nhits_df = truth_num_hits.loc[truth_num_hits.particle_id == reconstructed_particle_id]
            nhits_in_region = len(nhits_df)
            particle_purity = n_good / nhits_in_region
            print("particle purity: ", particle_purity, " track purity: ", track_purity)

            if (track_purity >= 0.5) and (particle_purity >= 0.5):
                
                # reference tracks can be reconstructed more than once - check no double counting
                if reconstructed_particle_id not in reconstructed_particle_ids_set:
                    reconstructed_particle_ids_set.add(reconstructed_particle_id)
                    track_purities = np.append(track_purities, track_purity)
                    particle_purities = np.append(particle_purities, particle_purity)
                    num_reconstructed_tracks += 1
                else:
                    print("OH NO! Track reconstructed more than once, moving onto next candidate")
            
            else:
                print("OH NO! purity not high enough")
        
        else: 
            print("OH NO! n_good < 0.5*len(reference_hits)")
    else:
        print("OH NO! Reconstructed particle id does not exist in reference tracks particles!")
        print("Particle id:", reconstructed_particle_id)

np.savetxt(outputDir + "/extracted_track_purities.csv", track_purities, delimiter=",")
np.savetxt(outputDir + "/extracted_particle_purities.csv", particle_purities, delimiter=",")

# print("Reconstructed particle ids set:\n", reconstructed_particle_ids_set)

# # distribution of reconstructed tracks pTs before the pT cut
# plt.hist(pT_dist_before, bins=100)
# plt.ylabel("Frequency")
# plt.xlabel("pT (GeV)")
# plt.title("pT distribution of all extracted candidates")
# # plt.savefig(outputDir + "/pt_distribution_extracted_candidates.png", dpi=300)
# plt.show()

# distribution of reconstructed tracks pTs after the purity calc & pT cut
# pT_dist = []
# for i, r in enumerate(reconstructed_particle_ids_set):
#     pT_dist.append(particles.loc[particles.particle_id == r].pT.item())
# plt.hist(pT_dist, bins=30)
# plt.title("pT distribution of extracted candidates after processing purity & pT cut")
# plt.ylabel("Frequency")
# plt.xlabel("pT (GeV)")
# plt.savefig(outputDir + "/pt_distribution_extracted_candidates.png", dpi=300)

efficiency = num_reconstructed_tracks * 100 / num_reference_tracks
efficiency= "{:.3f}".format(efficiency)
print("\n-------------------------------------------------------")
print("Total num of reconstructed tracks:", num_reconstructed_tracks)
print("Total num of reference tracks:", num_reference_tracks)
print("Track reconstruction efficiency: ", efficiency, "%")
print("-------------------------------------------------------")
