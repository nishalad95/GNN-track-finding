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


# parse command line args
parser = argparse.ArgumentParser(description='extract track candidates')
parser.add_argument('-t', '--eventTruth', help="Full directory path to event truth from TrackML")
parser.add_argument('-o', '--output', help='output directory to save data')
args = parser.parse_args()
outputDir = args.output
event_truth = args.eventTruth
# for evaluating the GNN algorithm
# TODO: will chance in future
inputDir = "src/output/iteration_2/candidates/"


# every hit has an associated particle id: from the truth information file for hits
# full info about particles itself is contained in the particles file

#########################################
# obtain reference tracks from MC truth
#########################################

reference_path = event_truth + "/event000001000-"

# identify the ids of particles that have pT > 1GeV, units GeV/c
particles = pd.read_csv(reference_path + "particles.csv", sep=',')
particles = particles.assign(pT=lambda row: (np.sqrt(row.px**2 + row.py**2)))
pt_cut = 1.  # units of momentum GeV/c
particles = particles.loc[particles.pT >= pt_cut]
particle_ids = particles.particle_id.to_list()

# go through the truth file for hits & identify hits cooresponding to each particle_id
truth = pd.read_csv(reference_path + "truth.csv", sep=',')
hit_ids = truth.loc[truth.particle_id.isin(particle_ids)]
hit_ids = hit_ids.hit_id.to_list()

# extract the subset of hits contained in the volume of interest (i.e. only endcap for now)
hits = pd.read_csv(reference_path + "hits.csv", sep=',')
hits['particle_id'] = truth['particle_id']
left_endcap_hits = hits.loc[(hits.hit_id.isin(hit_ids)) & (hits.volume_id == 7)]
print("Number of left endcap hits > 1 GeV:", len(left_endcap_hits))

# plot
left_endcap_hits = left_endcap_hits.assign(r=lambda row: (np.sqrt(row.x**2 + row.y**2)))
# x = left_endcap_hits.x.to_list()
# y = left_endcap_hits.y.to_list()
# z = left_endcap_hits.z.to_list()
# r = left_endcap_hits.r.to_list()
# plt.scatter(x, y)
# plt.show()
# plt.scatter(z, r)
# plt.show()


# Compute number of reference tracks: Apply a cut on the num of hits
# require >= 4 hits per track (in the volume of interest) to be a reference track
num_hits = 4
unique_particle_ids = left_endcap_hits.particle_id.unique()
num_reference_tracks = 0
reference_tracks_dict = {}          # key:value particle_id:[hit_ids]
for i, p in enumerate(unique_particle_ids):
    ref_track = left_endcap_hits.loc[left_endcap_hits.particle_id == p]
    if len(ref_track) >= num_hits:
        
        # check the "one-hit-per-module" constraint
        #TODO: need to check for same volume too here!!
        one_hit_per_module = ref_track[ref_track.duplicated(['layer_id','module_id'], keep=False)]
        if len(one_hit_per_module) == 0:
            # good reference track found
            num_reference_tracks += 1
            reference_tracks_dict[p] = ref_track.hit_id.to_list()   # add to dictionary
            # if i < 1:
            #     print("ref track:\n", ref_track)
            #     print("hit_ids in reference track")
            #     print(reference_tracks_dict[p])
            #     x = ref_track.x.to_list()
            #     y = ref_track.y.to_list()
            #     z = ref_track.z.to_list()
            #     r = ref_track.r.to_list()
            #     plt.scatter(x, y, s=10)
            #     plt.xlabel("x")
            #     plt.ylabel("y")
            #     plt.show()
            #     plt.scatter(z, r, s=10)
            #     plt.xlabel("z")
            #     plt.ylabel("r")
            #     plt.show()
        else:
            # TODO: handle this case
            print(">1 hit per module!! TODO: will need to handle this case!")
            print("need to check for same volume id & layer id, but one-hit-per-module")
            print(one_hit_per_module)

print("Number of reference tracks:", num_reference_tracks)
# print("reference tracks:", reference_tracks_dict)
# print("reference tracks particle ids:\n", reference_tracks_dict.keys())

# #########################################################
# # Look at what tracks have been succesfully reconstructed
# #########################################################


# TODO
track_candidates = []
i = 0
subgraph_path = "_subgraph.gpickle"
path = inputDir + str(i) + subgraph_path
while os.path.isfile(path):
    sub = nx.read_gpickle(path)
    track_candidates.append(sub)
    i += 1
    path = inputDir + str(i) + subgraph_path



num_reconstructed_tracks = 0
track_purities = np.array([])
particle_purities = np.array([])
truth_num_hits = pd.read_csv(reference_path + "nodes-particles-id.csv", sep=',')

for i, track in enumerate(track_candidates):
    print("---\nProcessing track candidate ", str(i), "\n---")
    # {node: {hit_id: [], particle_id: []} }
    hit_dissociation = nx.get_node_attributes(track, 'hit_dissociation').values()

    # if i == 0:   # temporary for debugging
    gnn_particle_ids = []
    for hd in hit_dissociation:
        values = list(hd.values())
        gnn_particle_ids.append(values[1])

    gnn_particle_ids = list(itertools.chain(*gnn_particle_ids))
    freq_dist = Counter(gnn_particle_ids)
    reconstructed_particle_id = max(freq_dist, key=freq_dist.get)
    n_good = freq_dist[reconstructed_particle_id]
    
    # compute track purity and particle purity
    track_purity = n_good / len(gnn_particle_ids)
    track_purities = np.append(track_purities, track_purity)
    nhits = truth_num_hits.loc[truth_num_hits.particle_id == reconstructed_particle_id]['nhits'].iloc[0]
    particle_purity = n_good / nhits
    particle_purities = np.append(particle_purities, particle_purity)
    print("Track purity:", track_purity)
    print("Particle purity:", particle_purity)

    # compute number of reconstructed tracks
    if track_purity >= 0.5:     # good reconstructed track
        # check if reconstructed track particle_id appears in reference tracks
        if reconstructed_particle_id in reference_tracks_dict.keys():
            print("HURRAY! Reconstructed particle id exists in reference tracks particles!")
            reference_hits = reference_tracks_dict[reconstructed_particle_id]
            if n_good >= 0.5 * len(reference_hits):
                print("track reconstructed by gnn! particle_id: ", reconstructed_particle_id)
                num_reconstructed_tracks += 1
            else: print("n_good < 0.5*len(reference_hits)")
        else:
            print("OH NO! Reconstructed particle id does not exist in reference tracks particles!")
            print("Particle id:", reconstructed_particle_id)

np.savetxt(outputDir + "/extracted_track_purities.csv", track_purities, delimiter=",")
np.savetxt(outputDir + "/extracted_particle_purities.csv", particle_purities, delimiter=",")


print("num of reconstructed tracks:", num_reconstructed_tracks)
efficiency = num_reconstructed_tracks * 100 / num_reference_tracks
efficiency= "{:.3f}".format(efficiency)
print("----------------------------")
print("Track reconstruction efficiency: ", efficiency, "%")
print("----------------------------")
